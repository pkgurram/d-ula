import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from models.LeNet5 import LeNet5
from optimizers.SGLD import sgld
import matplotlib.pyplot as plt
import numpy as np
import copy


def adjust_lr(optimizer, iterno, a, b, gamma):
    lr = a / ((b + iterno) ** gamma)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_evaluation(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)

        outputs = net(images)

        predicted = outputs.argmax(dim=1, keepdim=True)

        correct += predicted.cpu().eq(labels.view_as(predicted.cpu())).sum().item()

        total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


def get_scores(net, data_loader, device):
    net.eval()

    pred_prob = np.zeros((len(data_loader.dataset), 10))
    for i, (images, _) in enumerate(data_loader):

        images = images.to(device)

        outputs = net(images)

        pred_prob[i*data_loader.batch_size: i*data_loader.batch_size + images.shape[0], :] = F.softmax(outputs, dim=1).detach().cpu().numpy()

    return pred_prob


train_batch_size = 1024
test_batch_size = 4096
num_epochs = 10
torch.manual_seed(1)
a = 2.5 * 1e-4 # 0.2
b = 230
gamma = 0.55

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Train and test data sets and loaders
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = dsets.MNIST(root='./data', train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=trans, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)


svhn_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32, 32)), transforms.ToTensor()])
test_svhn_loader = torch.utils.data.DataLoader(dsets.SVHN('./data', split='test', download=True, transform=svhn_trans), batch_size=test_batch_size, shuffle=False)

# Model and loss
net = LeNet5().to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')

# Optimizer
optimizer = sgld(net.parameters(), lr=0.01, norm_sigma=0.0, num_batches=len(train_loader), addnoise=False) # Gradient of Kaiming uniform prior is 0
iterno = 1
adjust_lr(optimizer, iterno=iterno, a=a, b=b, gamma=gamma)

# Test accuracy before training
test_accuracy = [accuracy_evaluation(net, test_loader, device)]
iterlist = [iterno]
for epoch in range(num_epochs):

    net.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        net.zero_grad()
        optimizer.zero_grad()

        outputs = net(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        # tot_loss += loss.item()
        iterno += 1
        adjust_lr(optimizer, iterno=iterno, a=a, b=b, gamma=gamma)

        if iterno % 10 == 0:
            # print('Iteration: {}'.format(iterno))
            acc = accuracy_evaluation(net, test_loader, device)
            print('Iteration: {}. Accuracy: {}'.format(iterno, acc))
            test_accuracy.append(acc)
            iterlist.append(iterno)


test_svhn_acc = accuracy_evaluation(net, test_svhn_loader, device)
test_mnist_acc = accuracy_evaluation(net, test_loader, device)

class_pred_prob_svhn = get_scores(net, test_svhn_loader, device)
class_pred_prob_mnist = get_scores(net, test_loader, device)

max_pred_prob_svhn = np.max(class_pred_prob_svhn, axis=1)
weights_svhn = np.ones_like(max_pred_prob_svhn) / len(max_pred_prob_svhn)

max_pred_prob_mnist = np.max(class_pred_prob_mnist, axis=1)
weights_mnist = np.ones_like(max_pred_prob_mnist) / len(max_pred_prob_mnist)


fontsize = 10
f1 = plt.figure()
ax = f1.gca()
_ = plt.hist(max_pred_prob_mnist, bins=100, range=(0, 1), weights=weights_mnist, alpha=0.75, label='MNIST')
_ = plt.hist(max_pred_prob_svhn, bins=100, range=(0, 1), weights=weights_svhn, alpha=0.5, label='SVHN')
plt.xlabel('Prob. of Predicted Label')
plt.ylabel('Density')
plt.legend()
plt.grid()
fontsize = 10
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
plt.show()

