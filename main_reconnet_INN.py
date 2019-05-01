from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from numpy.random import randn
from torch.autograd import Variable
from torch.nn import init
from torchvision import datasets, transforms
import math
from MInfoLossDotProduct import MutualInfoLoss
from iRevNet import iRevNet

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | adaptiveCS | adaptiveCS_resnet',
                    default='reconnet_INN')
parser.add_argument('--dataset', help='lsun | imagenet | mnist | bsd500 | bsd500_patch', default='cifar10')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--image-size', type=int, default=32, metavar='N',
                    help='The height / width of the input image to the network')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--layers-gan', type=int, default=3, metavar='N',
                    help='number of hierarchies in the GAN (default: 64)')
parser.add_argument('--gpu', type=int, default=1, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--w-loss', type=float, default=0.01, metavar='N.',
                    help='penalty for the mse and bce loss')
parser.add_argument('--cr', type=int, default=10, help='compression ratio')
parser.add_argument('--ch-enc', type=int, default=4, help='output channels of decoder')
parser.add_argument('--ksize-enc', type=int, default=8, help='kernel size of decoder')
parser.add_argument('--in-nc', type=int, default=3, help='input channels')
parser.add_argument('--gpu-ids', type=list, default=[0, 1], help='GPUs will be used')
parser.add_argument('--localFeat', type=int, default=512, help='intermediate features')
parser.add_argument('--measure', default='JSD', help='f-divergence type')
parser.add_argument('--MIMode', default='fd', help='mutual information calculation')

opt = parser.parse_args()
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: please run with GPU")
print(opt)

# torch.cuda.set_device(opt.gpu)
# print('Current gpu device: gpu %d' % (torch.cuda.current_device()))

if opt.seed is None:
    opt.seed = np.random.randint(1, 10000)
print('Random seed: ', opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

if not os.path.exists('%s/%s/cr%s/%s/model' % (opt.outf, opt.dataset, opt.cr, opt.model)):
    os.makedirs('%s/%s/cr%s/%s/model' % (opt.outf, opt.dataset, opt.cr, opt.model))
if not os.path.exists('%s/%s/cr%s/%s/image' % (opt.outf, opt.dataset, opt.cr, opt.model)):
    os.makedirs('%s/%s/cr%s/%s/image' % (opt.outf, opt.dataset, opt.cr, opt.model))


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def data_loader():
    kwopt = {'num_workers': 2, 'pin_memory': True} if opt.cuda else {}

    if opt.dataset == 'lsun':
        train_dataset = datasets.LSUN(db_path=opt.datapath + 'train/', classes=['bedroom_train'],
                                      transform=transforms.Compose([
                                          transforms.Resize(opt.image_size),
                                          transforms.CenterCrop(opt.image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ]))
    elif opt.dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           #transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        val_dataset = datasets.MNIST('./data', train=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(opt.image_size),
                                         transforms.CenterCrop(opt.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         #transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
    elif opt.dataset == 'bsd500':
        train_dataset = datasets.ImageFolder(root='/home/user/kaixu/myGitHub/datasets/BSDS500/train-aug/',
                                             transform=transforms.Compose([
                                                 transforms.Resize(opt.image_size),
                                                 transforms.CenterCrop(opt.image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ]))

        val_dataset = datasets.ImageFolder(root='/home/user/kaixu/myGitHub/datasets/SISR/val/',
                                           transform=transforms.Compose([
                                               transforms.Resize(opt.image_size),
                                               transforms.CenterCrop(opt.image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    elif opt.dataset == 'bsd500_patch':
        train_dataset = datasets.ImageFolder(root=opt.datapath + 'train_64x64',
                                             transform=transforms.Compose([
                                                 transforms.Resize(opt.image_size),
                                                 transforms.CenterCrop(opt.image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ]))

        val_dataset = datasets.ImageFolder(root=opt.datapath + 'val_64x64',
                                           transform=transforms.Compose([
                                               transforms.Resize(opt.image_size),
                                               transforms.CenterCrop(opt.image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    elif opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(opt.image_size),
                                             transforms.CenterCrop(opt.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))

        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    return train_loader, val_loader



def train(epochs, trainloader, valloader):
    # Initialize variables
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    channels = sz_input[1]
    img_size = sz_input[2]
    n = img_size ** 2
    m = n // opt.cr
    opt.measurements = m * channels

    if os.path.exists('sensing_matrix_cr{}_w{}_h{}.npy'.format(opt.cr, opt.image_size, opt.image_size)):
        sensing_matrix = np.load('sensing_matrix_cr{}_w{}_h{}.npy'.format(opt.cr, opt.image_size, opt.image_size))
    else:
        sensing_matrix = randn(channels, m, n)

    input = torch.FloatTensor(opt.batch_size, channels, m)
    target = torch.FloatTensor(opt.batch_size, channels, img_size, img_size)

    # Instantiate models
    decoder = Encoder(img_size**2, img_size**2//opt.cr)
    irevnet = iRevNet(nBlocks=(4, 4, 4), nStrides=(1, 1, 1), nClasses=10, nChannels=(16, 16, 16), in_shape=(opt.in_nc, opt.image_size, opt.image_size))

    # Weight initialization
    weights_init(decoder, init_type='normal')
    weights_init(irevnet, init_type='normal')
    optimizer_dec = optim.Adam(decoder.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_irevnet = optim.Adam(irevnet.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    cudnn.benchmark = True

    opt.device = torch.device('cuda' if opt.gpu_ids is not None else 'cpu')

    irevnet = torch.nn.DataParallel(irevnet, device_ids=opt.gpu_ids).to(opt.device)
    decoder = torch.nn.DataParallel(decoder, device_ids=opt.gpu_ids).to(opt.device)
    criterion_mse.to(opt.device), criterion_bce.to(opt.device)

    for epoch in range(epochs):
        # training level 0
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            irevnet.train()
            decoder.train()

            data_array = data.numpy()
            for i in range(opt.batch_size):
                for j in range(channels):
                    input[i, j, :] = torch.from_numpy(
                            sensing_matrix[j, :, :].dot(data_array[i, j].flatten())).to(opt.device)

            target = torch.from_numpy(data_array).to(opt.device)

            # Train network
            decoder.zero_grad()
            irevnet.zero_grad()

            output = decoder(input)
            output = irevnet(output)               # Forward Propagation

            err_mse = criterion_mse(output, target)
            err_mse.backward()

            optimizer_irevnet.step()
            optimizer_dec.step()

            if idx % opt.log_interval == 0:
                print('[%d/%d][%d/%d] errG_mse: %.4f' % (
                    epoch, epochs, idx, len(trainloader), err_mse.data))

        torch.save(irevnet.state_dict(),
                   '%s/%s/cr%s/%s/model/lapnet0_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.cr, opt.model, epoch))

        vutils.save_image(target.data,
                          '%s/%s/cr%s/%s/image/epoch_%03d_real.png'
                          % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)
        vutils.save_image(output.data,
                          '%s/%s/cr%s/%s/image/epoch_%03d_fake.png'
                          % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)
        irevnet.eval()
        decoder.eval()
        val(opt, epoch, valloader, input, sensing_matrix, irevnet, decoder, criterion_mse)

class Encoder(nn.Module):
    def __init__(self, in_nc, nf, kernel_size=1, stride=1, proj_method='linear', bias=False):
        super(Encoder, self).__init__()

        self.proj_method = proj_method
        self.relu = nn.ReLU(inplace=True)

        if proj_method == 'conv':
            self.decoder = nn.Conv2d(in_nc, nf, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
            self.bn1 = nn.BatchNorm2d(nf)

        elif proj_method == 'linear':
            self.decoder = nn.Linear(nf, in_nc, bias=bias)
            self.img_size = int(math.sqrt(in_nc))

        # for k, v in self.decoder.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        if self.proj_method == 'conv':
            out = self.decoder(x)
        elif self.proj_method == 'linear':
            out_dec = self.decoder(x)
            out_dec = out_dec.view(out_dec.shape[0], out_dec.shape[1], self.img_size, self.img_size)

        return out_dec

def val(opt, epoch, valloader, input, sensing_matrix, net, D, criterion_mse):
    errD_fake_mse_total = 0

    with torch.no_grad():
        for idx, (data, _) in enumerate(valloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            data_array = data.numpy()
            for i in range(opt.batch_size):
                for j in range(opt.in_nc):
                    input[i, j, :] = torch.from_numpy(
                        sensing_matrix[j, :, :].dot(data_array[i, j].flatten())).to(opt.device)

            target = torch.from_numpy(data_array).to(opt.device)

            outE = D(input)
            output = net(outE)

            errD_fake_mse = criterion_mse(output, target)
            errD_fake_mse_total += errD_fake_mse
            if idx % 20 == 0:
                print('Test: [%d][%d/%d] errG_mse: %.4f \n,' % (epoch, idx, len(valloader), errD_fake_mse.data))

    print('Test: [%d] average errG_mse: %.4f,' % (epoch, errD_fake_mse_total.data / len(valloader)))
    vutils.save_image(target.data,
                      '%s/%s/cr%s/%s/image/epoch_%03d_real.png'
                      % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)
    vutils.save_image(output.data,
                      '%s/%s/cr%s/%s/image/epoch_%03d_fake.png'
                      % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)


def main():
    train_loader, val_loader = data_loader()
    train(opt.epochs, train_loader, val_loader)


if __name__ == '__main__':
    main()
