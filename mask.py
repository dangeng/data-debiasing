import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from utils import make_weights_for_balanced_classes, UnNormalize

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=bool, default=False,
                    help='Resume training from checkpoint')
parser.add_argument('--ckpt-path', type=str, default=None,
                    help='Path to resume checkpoint')
args = parser.parse_args()

writer = SummaryWriter()

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

class Debiaser(nn.Module):
    def __init__(self):
        super(Debiaser, self).__init__()
        self.gen_mask = nn.Sequential(# b, 3, 224, 224
            # ENCODE
            nn.Conv2d(3, 128, kernel_size=10, stride=4, padding=3),
            # b, 128, 56, 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # b, 128, 28, 28

            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            # b, 192, 28, 28
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # b, 192, 14, 14 = 32,448

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # b, 384, 14, 14
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # b, 256, 14, 14

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # b, 256, 14, 14 = 43,264

            # DECODE
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            # b, 256, 14, 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 384, kernel_size=4, stride=2, padding=1),
            # b, 384, 28, 28
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            # b, 192, 56, 56
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(192, 128, kernel_size=5, padding=2),
            # b, 128, 56, 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 3, kernel_size=12, stride=4, padding=4),
            # b, 3, 224, 224
        )

    def forward(self, x):
        mask = self.gen_mask(x)
        return x + mask

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def train(debiaser, discriminator, device, train_loader, debiaser_optimizer, discriminator_optimizer, epoch):
    debiaser.train()
    discriminator.train()
    discriminator_loss = nn.BCELoss().to(device)
    adversarial_loss = nn.BCELoss().to(device)
    recon_loss = nn.MSELoss().to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        #data, target = data.to(device), target.to(device)
        debiased = debiaser(data)
        pred = discriminator(debiased.detach())
        pred_attached = discriminator(debiased)

        r_loss = recon_loss(debiased, data)
        a_loss = adversarial_loss(pred_attached, 1 - target)
        g_loss = r_loss + a_loss

        d_loss = discriminator_loss(pred, target)

        debiaser_optimizer.zero_grad()
        g_loss.backward()
        debiaser_optimizer.step()

        discriminator_optimizer.zero_grad()
        d_loss.backward()
        discriminator_optimizer.step()

        if batch_idx % 2 == 0:
            writer.add_scalar('data/train_r_loss', r_loss.item(), epoch * len(trainloader) * len(data) + batch_idx * len(data))
            writer.add_scalar('data/train_d_loss', d_loss.item(), epoch * len(trainloader) * len(data) + batch_idx * len(data))
            writer.add_scalar('data/train_a_loss', a_loss.item(), epoch * len(trainloader) * len(data) + batch_idx * len(data))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tR-Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), r_loss.item()))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD-Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), d_loss.item()))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tA-Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), a_loss.item()))

def test(debiaser, discriminator, device, test_loader, epoch):
    debiaser.eval()
    discriminator.eval()
    discriminator_loss = nn.BCELoss().to(device)
    recon_loss = nn.MSELoss().to(device)

    test_r_loss = 0
    test_d_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            #data, target = data.to(device), target.to(device)
            debiased = debiaser(data)
            pred = discriminator(debiased.detach())

            r_loss = recon_loss(debiased, data)
            d_loss = discriminator_loss(pred, target)

            test_r_loss += recon_loss(debiased, data).item()
            test_d_loss += discriminator_loss(pred, target).item()

            correct += (pred > .5).int().eq(target.int()).sum().item()

    test_r_loss /= len(test_loader.dataset)
    test_d_loss /= len(test_loader.dataset)

    writer.add_scalar('data/test_r_loss', test_r_loss, epoch)
    writer.add_scalar('data/test_d_loss', test_d_loss, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_r_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_d_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

trainset = ImageFolder('data/gender_images/train',
        transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

testset = ImageFolder('data/gender_images/test',
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))

weights = make_weights_for_balanced_classes(trainset.imgs, len(trainset.classes))
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

trainloader = DataLoader(trainset, batch_size=16, sampler=sampler)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

e = enumerate(testloader)

def plotIm():
    _, (inputs, targets) = next(e)
    print(targets[0])
    inp = inputs[0].numpy().transpose((1,2,0))
    plt.imshow(inp)
    plt.show()

debiaser = Debiaser().to(device)
discriminator = Discriminator().to(device)

if args.resume:
    print('Loading model')
    checkpoint = torch.load(args.ckpt_path)
    debiaser.load_state_dict(checkpoint['state_dict'])
    print('Loaded model')

debiaser_optimizer = optim.Adam(debiaser.parameters(), lr=1e-3, weight_decay=1e-5)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-5)


# Training autoencoder first
for epoch in range(10):
    train(debiaser, discriminator, device, trainloader, debiaser_optimizer, discriminator_optimizer, epoch)
    test(debiaser, discriminator, device, testloader, epoch)
    #fname = 'checkpoints/ae_' + str(epoch) + '.pth.tar'
    #torch.save({
            #'epoch': epoch,
            #'state_dict': debiaser.state_dict(),
            #'optimizer' : optimizer.state_dict(),
        #}, fname)

