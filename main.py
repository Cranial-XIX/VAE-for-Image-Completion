from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from IPython.display import Image
from model import VAE, CVAE
from torch.autograd import Variable
from torchvision import datasets, transforms

# Reading command line arguments
parser = argparse.ArgumentParser(description='VAE for Image Autocompletion')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--mode', default="train",
                    help='Whether to use a pretrained model')
parser.add_argument('--model', default="VAE",
                    help='Which model to use for autocompletion: VAE, VAE_INC, CVAE_LB, CVAE_INC')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# arguments for using cuda on GPU
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# training instances
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# testing instances
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data', train=False, 
        transform=transforms.ToTensor()
    ),
    batch_size=args.batch_size, shuffle=True, **kwargs)

isVAE = True
# initialize the model
if args.model == "VAE" or args.model == "VAE_INC":
    model = VAE()
elif args.model == "CVAE_LB":
    isVAE = False
    model = CVAE(784+10, 400, 20, False)
else:
    isVAE = False
    model = CVAE(784*2, 400, 20, False)

if args.cuda:
    model.cuda()

# Binary Cross-Entropy loss
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    '''
    the loss consists of two parts, the binary Cross-Entropy loss for
    the reconstruction loss and the KL divergence loss for the variational
    inference.
    '''
    BCE = reconstruction_function(recon_x, x)
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# one-hot embedding for label representation
labelembed = Variable(torch.eye(10, 10))

def min(a, b):
    return a if a < b else b

def max(a, b):
    return a if a > b else b

def occludeimg(data):
    # Randomily choose a center that is on the object
    # of the image and mask out a 6*6 square around it
    occluded = data.clone()
    size = occluded.size()
    for i in range(size[0]):
        nonzeros = torch.nonzero(newData[i,0])
        choice = random.randint(len(nonzeros))
        row = nonzeros[choice][0]
        col = nonzeros[choice][1]
        left = max(0, col-3)
        right = min(27, col+3)
        top = max(0, row-3)
        bot = min(27, row+3)
        incomplete[i,0,top:bot, left:right] = 0.0
    return occluded

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (gldimg, label) in enumerate(train_loader):
        # determine the input for the model
        gldimg = Variable(gldimg)
        if args.model == "VAE" or args.model == "CVAE_INC":
            inp = gldimg
        else:
            inp = Variable(occludeimg(gldimg))

        # if it is conditional VAE, we need to compute the condition
        if args.model == "CVAE_LB":
            cond = Variable(labelembed.index_select(0, label))
        elif args.model == "CVAE_INC":
            cond = Variable(occludeimg(gldimg))

        # make it cuda variable if allowed
        if args.cuda:
            inp = inp.cuda()
            gldimg = gldimg.cuda()
            cond = cond.cuda()

        optimizer.zero_grad()
        # forward
        if isVAE:
            # VAE
            recon_batch, mu, logvar = model(inp)
        else:
            # CVAE
            recon_batch, mu, logvar = model(inp, cond)

        # calculate loss
        loss = loss_function(recon_batch, gldimg, mu, logvar)
        # backprop
        loss.backward()

        train_loss += loss.data[0]
        # take a gradient step
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(gldimg), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(gldimg)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        if args.cuda:
            data = data.cuda()

        newData = data
        for i in xrange(newData.size()[0]):
            row = random.randint(3,25)
            col = random.randint(3,25)
            # print (i, '\t', row, '\t', col)
            # print (data[i,0,row-3:row+3, col-3:col+3])
            newData[i,0,row-3:row+3, col-3:col+3] = 0.0

        newData = Variable(newData, volatile=True)
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(newData)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def getfile():
    if args.model == "VAE":
        return "VAE_model"
    elif args.model == "VAE_INC":
        return "VAE_INC_model"
    elif args.model == "CVAE_LB":
        return "CVAE_LB_model"
    elif args.model == "CVAE_INC":
        return "CVAE_INC_model"
    else:
        print ("INVALID MODEL TYPE!")

## training or loading model
if args.mode == "train":
    savefile = getfile()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    torch.save({'state_dict': model.state_dict()}, savefile)
elif args.mode == "inpainting":
    loadfile = getfile()
    pre = torch.load(loadfile)
    model.load_state_dict(pre['state_dict'])

# some helper function to show up images
def compare(a, b):
    to_image = transforms.ToPILImage()
    to_image(a).show()
    to_image(b).show()

def show(a):
    to_image = transforms.ToPILImage()
    to_image(a).show()


# visualize result by comparing convered img and original img
for batch_idx, (data, _) in enumerate(train_loader):

    sampleNumber = 15
    test_img = Variable(data[sampleNumber])
    testImg = data[sampleNumber]

    row = random.randint(3,25)
    col = random.randint(3,25)
    testImg[0,row-3:row+3, col-3:col+3] = 0.0

    testImg = Variable(testImg)
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
# z_mu = model.reparametrize(mu, var)
# min_recon = model.decode(z_mu)
max_idx = 1000
min = 10000000
scale = 0.1

for i in range(max_idx):
    z_mu = model.reparametrize(mu, var)
    # z_std = Variable(torch.FloatTensor(1,20).normal_())
    # z = z_mu + scale * z_std
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





