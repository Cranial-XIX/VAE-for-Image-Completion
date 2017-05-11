from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython.display import Image
import random

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pretrained', type=int, default=0,
                    help='Whether to use a pretrained model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # print (x)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # print ('logvar',logvar.size())
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        # print ('std', std.size(), 'mu', mu.size())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        # print ('x', x.size(), 'x_modified',x.view(-1, 784).size())
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def reconstruction_loss(recon_x, x):
    BCE = reconstruction_

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        # print ('data',data.size()[0])
        # print ('element', data[0,0,2,2])

        newData = data
        # Random choose a 6 x 6 mask in the image and set it to 0
        for i in xrange(newData.size()[0]):
            row = random.randint(3,25)
            col = random.randint(3,25)
            # print (i, '\t', row, '\t', col)
            # print (data[i,0,row-3:row+3, col-3:col+3])
            newData[i,0,row-3:row+3, col-3:col+3] = 0.0

        newData = Variable(newData)
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(newData)
        # print (data.size())
        # print (recon_batch.size())
        # print ('recon',recon_batch.size())
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        if args.cuda:
            data = data.cuda()

        for i in xrange(data.size()[0]):
            row = random.randint(3,25)
            col = random.randint(3,25)
            # print (i, '\t', row, '\t', col)
            # print (data[i,0,row-3:row+3, col-3:col+3])
            data[i,0,row-3:row+3, col-3:col+3] = 0.0

        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



if args.pretrained == 0:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    torch.save({'state_dict': model.state_dict()}, "model_dict")
else:
    pre = torch.load("model_dict")
    model.load_state_dict(pre['state_dict'])

def compare(a, b):
    to_image = transforms.ToPILImage()
    to_image(a).show()
    to_image(b).show()

def show(a):
    to_image = transforms.ToPILImage()
    to_image(a).show()

for batch_idx, (data, _) in enumerate(train_loader):
    testImg = data[1]
    row = random.randint(3,25)
    col = random.randint(3,25)
    testImg[0,row-3:row+3, col-3:col+3] = 0.0

    testImg = Variable(testImg)
    test_img = Variable(data[1])
    break

'''
# pick the first image as test image
max_idx = 100000
mu, var = model.encode(testImg.view(-1, 784))
z_mu = model.reparametrize(mu, var)
min = 10000000
scale = 0.15

for i in xrange(max_idx):
    z_std = Variable(torch.FloatTensor(1,20).normal_())
    z = z_mu + scale * z_std
    recon = model.decode(z_mu)
    loss = reconstruction_function(recon, test_img)
    if loss < min:
        min = loss
        min_recon = recon
'''


mu, var = model.encode(testImg.view(-1, 784))
z_mu = model.reparametrize(mu, var)
# min_recon = model.decode(z_mu)
max_idx = 1000
min = 10000000
scale = 0.1
for i in xrange(max_idx):
    z_std = Variable(torch.FloatTensor(1,20).normal_())
    z = z_mu + scale * z_std
    recon = model.decode(z_mu)
    loss = reconstruction_function(recon, test_img)
    if loss < min:
        min = loss
        min_recon = recon


compare(test_img.view(1, 28, -1).data, min_recon.view(1, 28, -1).data)

'''
i = 0
max_idx = 2
for batch_idx, (data, _) in enumerate(train_loader):
    if i >= max_idx:
        break
    i += 1
    original = Variable(data[0])
    recon, m, v = model(original)
    compare(original.view(1, 28, -1).data, recon.view(1, 28, -1).data)
'''





