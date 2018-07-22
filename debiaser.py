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

writer = SummaryWriter()

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.encoderm = nn.Sequential(                                   # b, 3, 224, 224
            nn.Conv2d(3, 128, kernel_size=10, stride=4, padding=3),      # b, 128, 56, 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # b, 128, 28, 28
            nn.Conv2d(128, 192, kernel_size=5, padding=2),               # b, 192, 28, 28
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # b, 192, 14, 14 = 32,448
            nn.Conv2d(192, 384, kernel_size=3, padding=1),              # b, 384, 14, 14
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),              # b, 256, 14, 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),              # b, 256, 14, 14 = 43,264
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),                      # b, 256, 6, 6
        )

        self.encoder = nn.Sequential(                                   # b, 3, 224, 224
            nn.Conv2d(3, 128, kernel_size=12, stride=4, padding=2),     # b, 128, 55, 55
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),              # b, 192, 55, 55
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),    # b, 384, 28, 28
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=4, stride=2, padding=1),    # b, 256, 14, 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),              # b, 256, 14, 14 = 43,264
        )

        self.decoder = nn.Sequential(                                   # b, 256, 14, 14
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),              # b, 256, 14, 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 384, kernel_size=4, stride=2, padding=1),    # b, 192, 28, 28
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=1),    # b, 384, 13, 13
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 128, kernel_size=5, padding=2),              # b, 256, 13, 13
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=12, stride=4, padding=2),     # b, 256, 13, 13
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoderm(x)
        enc = x.view(x.size(0), 256 * 14 * 14)
        x = self.decoder(x)
        return x, enc

class Discriminator(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(256 * 14 * 14, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.feedforward(x)
        return x

def train(model, device, criterion, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        #data, target = data.to(device), target.to(device)
        recon, enc = model(data)
        loss = criterion(recon, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            writer.add_scalar('data/train_loss', loss.item(), epoch * len(trainloader) * len(data) + batch_idx * len(data))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, criterion, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            recon, env= model(data)
            test_loss += criterion(recon, data).item() # sum up batch loss
            writer.add_scalar('data/test_loss', test_loss, epoch)

            #print(target)
            #print(output)
            #print((output > .5).int().eq(target.int()).sum().item())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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
testloader = DataLoader(testset, batch_size=1)

e = enumerate(testloader)

def plotIm():
    _, (inputs, targets) = next(e)
    print(targets[0])
    inp = inputs[0].numpy().transpose((1,2,0))
    plt.imshow(inp)
    plt.show()

def eval():
    model.eval()
    _, (inputs, targets) = next(e)
    recon, enc = model(inputs.to(device))

    inputs = unnormalize(inputs)
    inputs = inputs[0].numpy().transpose((1,2,0))
    plt.imshow(inputs)
    plt.show()

    recon = unnormalize(recon)
    recon = recon[0].detach().cpu().numpy().transpose((1,2,0))

    plt.imshow(recon)
    plt.show()

model = AlexNet(1).to(device)
optimizer = optim.SGD(model.parameters(), lr=.001)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

#criterion = nn.CrossEntropyLoss().to(device)
criterion = nn.MSELoss().to(device)
for epoch in range(50):
    train(model, device, criterion, trainloader, optimizer, epoch)
    test(model, device, criterion, testloader, epoch)
    fname = 'checkpoints/ae_' + str(epoch) + '.pth.tar'
    torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, fname)
