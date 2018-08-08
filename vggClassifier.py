import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=bool, default=False,
                    help='Resume training from checkpoint')
parser.add_argument('--ckpt-path', type=str, default=None,
                    help='Path to resume checkpoint')
args = parser.parse_args()

writer = SummaryWriter()

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
cpu = torch.device("cpu")

# Featurizer

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 1)

model_conv = model_conv.to(device)

def train_discriminator(discriminator, criterion, train_loader, optimizer, epoch):
    discriminator.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(cpu).float().unsqueeze(1)
        pred = discriminator(data)

        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 1 == 0:
            writer.add_scalar('data/discriminator_train_loss', loss.item(), epoch * len(trainloader) * len(data) + batch_idx * len(data))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(int(((pred > .5).float() == target).sum()))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count)) 
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

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
testloader = DataLoader(testset, batch_size=4, shuffle=True)

e = enumerate(testloader)

discriminator_optimizer = optim.Adam(model_conv.fc.parameters(), lr=1e-3, weight_decay=1e-5)

criterion = nn.MSELoss().to(device)

for epoch in range(10):
    train_discriminator(model_conv, criterion, trainloader, discriminator_optimizer, epoch)
