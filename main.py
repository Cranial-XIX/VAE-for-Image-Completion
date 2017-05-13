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
                    help='Choose a mode: train or inpainting')
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

half_occludesize = 4
def occludeimg(data):
    # Randomily choose a center that is on the object
    # of the image and mask out a 6*6 square around it
    occluded = data.clone()
    size = occluded.size()
    for i in range(size[0]):
        nonzeros = torch.nonzero(occluded[i,0])
        choice = random.randint(0, len(nonzeros)-1)
        row = nonzeros[choice][0]
        col = nonzeros[choice][1]
        left = max(0, col-half_occludesize)
        right = min(27, col+half_occludesize)
        top = max(0, row-half_occludesize)
        bot = min(27, row+half_occludesize)
        occluded[i,0, top:bot, left:right] = 0.0
    return occluded

# used for testing
def occludeimg_and_returncenter(data):
    # Randomily choose a center that is on the object
    # of the image and mask out a 6*6 square around it
    occluded = data.clone()
    size = occluded.size()
    nonzeros = torch.nonzero(occluded[0])
    choice = random.randint(0, len(nonzeros)-1)
    row = nonzeros[choice][0]
    col = nonzeros[choice][1]
    left = max(0, col-half_occludesize)
    right = min(27, col+half_occludesize)
    top = max(0, row-half_occludesize)
    bot = min(27, row+half_occludesize)
    occluded[0, top:bot, left:right] = 0.0
    return occluded, (top, bot, left, right)

def occludeimg_with_center(data, center):
    # Randomily choose a center that is on the object
    # of the image and mask out a 6*6 square around it
    top, bot, left, right = center
    occluded = data.clone()
    occluded[0, top:bot, left:right] = 0.0
    return occluded

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (gldimg, label) in enumerate(train_loader):
        # determine the input for the model

        if args.model == "VAE" or args.model == "CVAE_INC":
            inp = Variable(gldimg)
        else:
            inp = Variable(occludeimg(gldimg))

        # if it is conditional VAE, we need to compute the condition
        if args.model == "CVAE_LB":
            cond = Variable(labelembed.index_select(0, Variable(label)))
        elif args.model == "CVAE_INC":
            cond = Variable(occludeimg(gldimg))

        gldimg = Variable(gldimg)
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
    for batch_idx, (gldimg, label) in enumerate(test_loader):
        # determine the input for the model
        if args.model == "VAE" or args.model == "CVAE_INC":
            inp = Variable(gldimg, volatile=True)
        else:
            inp = Variable(occludeimg(gldimg), volatile=True)

        # if it is conditional VAE, we need to compute the condition
        if args.model == "CVAE_LB":
            cond = Variable(labelembed.index_select(0, Variable(label)), volatile=True)
        elif args.model == "CVAE_INC":
            cond = Variable(occludeimg(gldimg), volatile=True)

        # make it cuda variable if allowed
        gldimg = Variable(gldimg, volatile=True)
        if args.cuda:
            inp = inp.cuda()
            gldimg = gldimg.cuda()
            cond = cond.cuda()

        # forward
        if isVAE:
            # VAE
            recon_batch, mu, logvar = model(inp)
        else:
            # CVAE
            recon_batch, mu, logvar = model(inp, cond)

        test_loss += loss_function(recon_batch, gldimg, mu, logvar).data[0]

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
    # TODO make the training cleaner and faster
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
    torch.save({'state_dict': model.state_dict()}, savefile)
elif args.mode == "inpainting":
    loadfile = getfile()
    print ("loading file from ", loadfile)
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

# visualize result by comparing occluded img and original img
print ("Start testing the autocompletion ...")
test_imgs = []
test_labels = []
test_originals = []
test_centers = []

# extract the 1st 10 images of 1st batch as test, randomly
# occlude part of them
ntest = 3
for batch_idx, (gldimg, label) in enumerate(test_loader):
    iteration = min(args.batch_size, ntest)
    for i in xrange(iteration):
        test_originals.append(Variable(gldimg[i]))
        occluded, center = occludeimg_and_returncenter(gldimg[i])
        test_imgs.append(Variable(occluded))
        test_centers.append(center)
        lb = Variable( torch.LongTensor( [label[i]] ) )
        test_labels.append(labelembed.index_select(0, lb))
    break

if args.model == "VAE":
    for i in xrange(iteration):
        img = test_imgs[i]
        mu, var = model.encode(img.view(1, 784))
        center = test_centers[i]

        maxidx = 1000
        minloss = 10000000
        scale = 0.2
        for j in range(maxidx):
            z_mu = model.reparametrize(mu, var)
            z_std = Variable(torch.FloatTensor(1,20).normal_())
            z = z_mu + scale * z_std
            recon = model.decode(z)
            recon = occludeimg_with_center(recon.view(1, 28, -1), center)
            loss = reconstruction_function(recon.view(-1), img.view(-1))
            if loss < minloss:
                minloss = loss
                min_recon = recon

        compare(img.data, min_recon.data)

elif args.model == "VAE_INC":
    for i in xrange(iteration):
        maxidx = 1000
        minloss = 10000000
        scale = 0.2
        img = test_imgs[i]
        mu, var = model.encode(img.view(1, 784))
        center = test_centers[i]
        for j in range(maxidx):
            z_mu = model.reparametrize(mu, var)
            recon = model.decode(z_mu)
            recon = occludeimg_with_center(recon.view(1, 28, -1), center)
            loss = reconstruction_function(recon.view(-1), img.view(-1))
            if loss < minloss:
                minloss = loss
                min_recon = recon

        compare(img.data, min_recon.data)    





