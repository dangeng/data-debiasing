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

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
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
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def train(model, device, criterion, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            writer.add_scalar('data/loss', loss.item(), batch_idx * len(data))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            correct += (output > .5).int().eq(target.int()).sum().item()

            #print(target)
            #print(output)
            #print((output > .5).int().eq(target.int()).sum().item())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
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

trainloader = DataLoader(trainset, batch_size=64, sampler=sampler)
testloader = DataLoader(testset, batch_size=1)

e = enumerate(trainloader)

def plotIm():
    _, (inputs, targets) = next(e)
    print(targets[0])
    inp = inputs[0].numpy().transpose((1,2,0))
    plt.imshow(inp)
    plt.show()

def eval():
    model.eval()
    _, (inputs, targets) = next(e)
    outputs = model(inputs.to(device))
    print(outputs)
    print(targets)

model = AlexNet(1).to(device)
model.train()
optimizer = optim.SGD(model.parameters(), lr=.01)

#criterion = nn.CrossEntropyLoss().to(device)
criterion = nn.MSELoss().to(device)
for epoch in range(10):
    train(model, device, criterion, trainloader, optimizer, epoch)
    test(model, device, criterion, testloader)
