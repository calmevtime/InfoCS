import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import *
from MInfoLossDotProduct import MutualInfoLoss, MI1x1ConvNet, MIFCNet
from init_net import init_weights

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT',
                    help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--in-nc', type=int, default=3, help='input channels')
parser.add_argument('--img-size', type=int, default=32, metavar='N',
                    help='The height / width of the input image to the network')
parser.add_argument('--cr', type=int, default=10, help='compression ratio')
parser.add_argument('--localFeat', type=int, default=512, help='intermediate features')
parser.add_argument('--gpu-ids', type=list, default=[0], help='GPUs will be used')
parser.add_argument('--measure', default='JSD', help='f-divergence type')
parser.add_argument('--MIMode', default='fd', help='mutual information calculation')
parser.add_argument('--model', help='basic | adaptiveCS | adaptiveCS_resnet', default='resnet20_recognition_MI_multitask')
parser.add_argument('--resume_encoder', default='./result/enc_MI/checkpoint_encoder_43.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_localnet', default='./result/enc_MI/checkpoint_local_net_43.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_globalnet', default='./result/enc_MI/checkpoint_global_net_43.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

best_prec = 0


def main():
    global args, best_prec
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    # set gpu environment
    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # Model building
    print('=> Building model...')
    if use_gpu:
        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !
        args.device = torch.device('cuda' if args.gpu_ids is not None else 'cpu')
        m = args.img_size ** 2 // args.cr
        measurements = m * args.in_nc
        encoder = Encoder(args.img_size ** 2, args.img_size ** 2 // args.cr)
        local_net = MI1x1ConvNet(args.in_nc, args.localFeat)
        global_net = MIFCNet(measurements, args.localFeat)
        model = resnet20_cifar(args)

        # model = resnet20_cifar(args)
        # model = resnet32_cifar()
        # model = resnet44_cifar()
        # model = resnet110_cifar()
        # model = preact_resnet110_cifar()
        # model = resnet164_cifar(num_classes=100)
        # model = resnet1001_cifar(num_classes=100)
        # model = preact_resnet164_cifar(num_classes=100)
        # model = preact_resnet1001_cifar(num_classes=100)

        # model = wide_resnet_cifar(depth=26, width=10, num_classes=100)

        # model = resneXt_cifar(depth=29, cardinality=16, baseWidth=64, num_classes=100)

        # model = densenet_BC_cifar(depth=190, k=40, num_classes=100)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/{}'.format(args.model)
        if not os.path.exists(fdir):
            os.makedirs(fdir)


        init_weights(encoder, init_type='normal', scale=1)
        init_weights(local_net, init_type='normal', scale=1)
        init_weights(global_net, init_type='normal', scale=1)
        encoder = nn.DataParallel(encoder).to(args.device)
        local_net = nn.DataParallel(local_net).to(args.device)
        global_net = nn.DataParallel(global_net).to(args.device)
        model = nn.DataParallel(model).to(args.device)
        MICriterion = MutualInfoLoss(args).to(args.device)
        criterion = nn.CrossEntropyLoss().to(args.device)

        optimizer = optim.Adam(list(encoder.parameters()) + list(local_net.parameters()) + list(global_net.parameters()) \
                               + list(model.parameters()), args.lr, betas=(0.5, 0.999))

        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if os.path.isfile(args.resume_encoder):
        checkpoint = torch.load(args.resume_encoder)
        # args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        if isinstance(encoder, nn.DataParallel):
            encoder = encoder.module
        encoder.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        checkpoint = torch.load(args.resume_localnet)
        if isinstance(local_net, nn.DataParallel):
            local_net = local_net.module
        local_net.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(args.resume_globalnet)
        if isinstance(global_net, nn.DataParallel):
            global_net = global_net.module
        global_net.load_state_dict(checkpoint['state_dict'])

        print("=> loaded encoder '{}' epoch {} \t localnet '{}' \t globalnet '{}')". \
              format(args.resume_encoder, checkpoint['epoch'], args.resume_localnet, args.resume_globalnet))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, 1)

        # train for one epoch
        train(args, trainloader, encoder, local_net, global_net, model, MICriterion, criterion, optimizer, epoch)

        # evaluate on test set
        prec = validate(args, testloader, encoder, local_net, global_net, model, MICriterion, criterion)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': encoder.module.state_dict() if isinstance(encoder, nn.DataParallel) else encoder.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir, 'encoder')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': local_net.module.state_dict() if isinstance(local_net, nn.DataParallel) else local_net.state_dict(),
        }, is_best, fdir, 'local_net')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': global_net.module.state_dict() if isinstance(global_net, nn.DataParallel) else global_net.state_dict(),
        }, is_best, fdir, 'global_net')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        }, is_best, fdir, 'model')


def train(args, trainloader, enc, local_net, global_net, model, mi_criterion, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mi_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    enc.train()
    local_net.train()
    global_net.train()
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(args.device), target.to(args.device)

        # compute output
        enc_out = enc(input)
        l_enc = local_net(input)
        g_enc = global_net(enc_out)
        output = model(enc_out)

        err_mi = mi_criterion(l_enc, g_enc)
        err = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        mi_losses.update(err_mi.item(), input.size(0))
        losses.update(err.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        err_mi.backward(retain_graph=True)
        err.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MILoss {miloss.val:.4f} ({miloss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, miloss=mi_losses, top1=top1))

def validate(args, val_loader, enc, local_net, global_net, model, mi_criterion, criterion):
    batch_time = AverageMeter()
    mi_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    enc.eval()
    local_net.eval()
    global_net.eval()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(args.device), target.to(args.device)

            # compute output
            enc_out = enc(input)
            l_enc = local_net(input)
            g_enc = global_net(enc_out)
            output = model(enc_out)
            err_mi = mi_criterion(l_enc, g_enc)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            mi_losses.update(err_mi.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MILoss {miloss.val:.4f} ({miloss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, miloss=mi_losses, top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fdir, name):
    filepath = os.path.join(fdir, 'checkpoint_{}_{}.pth'.format(name, state['epoch']))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif model_type == 2:
        if epoch < 60:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.2
        elif epoch < 160:
            lr = args.lr * 0.04
        else:
            lr = args.lr * 0.008
    elif model_type == 3:
        if epoch < 150:
            lr = args.lr
        elif epoch < 225:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
