import torch
import torch.nn as nn

from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, dinp=784, dhid=400, drep=20, cuda=False):
        # dhid = dimension of hidden layer
        # drep = dimension of hidden representation
        super(VAE, self).__init__()

        self.dinp = dinp
        self.cuda = cuda
        self.fc1 = nn.Linear(dinp, dhid)
        self.fc21 = nn.Linear(dhid, drep)
        self.fc22 = nn.Linear(dhid, drep)
        self.fc3 = nn.Linear(drep, dhid)
        self.fc4 = nn.Linear(dhid, dinp)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dinp))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

class CVAE(nn.Module):
    def __init__(self, dinp=784, dhid=400, drep=20, dcond=0, cuda=False):
        # dhid = dimension of hidden layer
        # drep = dimension of hidden representation
        super(CVAE, self).__init__()

        self.dinp = dinp
        self.cuda = cuda
        self.fc1 = nn.Linear(dinp+dcond, dhid)
        self.fc21 = nn.Linear(dhid, drep)
        self.fc22 = nn.Linear(dhid, drep)
        self.fc3 = nn.Linear(drep+dcond, dhid)
        self.fc4 = nn.Linear(dhid, dinp)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, cond):
        size = x.size()
        inp = torch.cat((x.view(size[0], -1), cond), 1)
        mu, logvar = self.encode(inp)
        z = self.reparametrize(mu, logvar)
        z = torch.cat((z, cond), 1)
        return self.decode(z), mu, logvar
