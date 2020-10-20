import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from models.LeNet5 import LeNet5
from optimizers.dist_SGLD import dsgld
from utils.getlaplacian import getlaplacian
import matplotlib.pyplot as plt
import numpy as np
import math
import copy


def adjust_lr(optimizer, modelparams, iterno, alpha0, beta0, gamma):
    alpha = alpha0 / ((1 + gamma * iterno) ** 0.55)
    beta = beta0 / ((1 + gamma * iterno) ** 0.05)
    for param_group in optimizer.param_groups:
        param_group['alpha'] = alpha
        param_group['beta'] = beta
        param_group['allmodels'] = modelparams


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


train_batch_size = 256
test_batch_size = 4096
num_epochs = 10
num_agents = 5
torch.manual_seed(1)
gamma = 1 / 230.0
alpha0 = 16.0 * 1e-4 / ((230 ** 0.55) * num_agents)
adj, lap = getlaplacian(num_agents, type=0)
_, sv, _ = np.linalg.svd(lap)
beta0 = 1.01 / np.max(sv)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Train and test data sets
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = dsets.MNIST(root='./data', train=True, transform=trans, download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=trans)

num_datasamples = int(len(train_dataset) / num_agents)
lengths = [num_datasamples] * num_agents
num_mini_batches = math.ceil(num_datasamples / train_batch_size)

# Train and test data loaders
train_data_split = torch.utils.data.random_split(dataset=train_dataset, lengths=lengths)
train_loader_list = []
for k in range(num_agents):
    train_loader_list.append(torch.utils.data.DataLoader(dataset=train_data_split[k], batch_size=train_batch_size,
                                                         shuffle=True))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

svhn_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32, 32)), transforms.ToTensor()])
test_svhn_loader = torch.utils.data.DataLoader(dsets.SVHN('./data', split='test', download=True, transform=svhn_trans), batch_size=test_batch_size, shuffle=False)

# Model and loss
model = []
modelparams = []
for k in range(num_agents):
    net = LeNet5().to(device)
    model.append(net)
    modelparams.append(copy.deepcopy(net))

criterion = nn.CrossEntropyLoss(reduction='sum')

# Optimizer
optimizerlist = []
iterno = 1
for k in range(num_agents):
    optimizer = dsgld(model[k].parameters(), allmodels=modelparams, adj_vec=adj[k, :], alpha=alpha0, beta=beta0,
                      norm_sigma=0.0, num_batches=len(train_loader_list[k]), addnoise=True)
    adjust_lr(optimizer, modelparams, iterno=iterno, alpha0=alpha0, beta0=beta0, gamma=gamma)
    optimizerlist.append(optimizer)

# Test accuracy before training
net_test_accuracy = []
for k in range(num_agents):
    net_test_accuracy.append(accuracy_evaluation(model[k], test_loader, device))
test_accuracy = [net_test_accuracy]
iterlist = [iterno]

class_scores_svhn = []
class_scores_mnist = []
for epoch in range(num_epochs):

    train_dataiter_list = []
    for k in range(num_agents):
        train_dataiter_list.append(iter(train_loader_list[k]))

    for i in range(num_mini_batches):

        iterno += 1
        modelparams = []
        for j in range(num_agents):
            cur_model = model[j]
            cur_model.train()
            data, labels = train_dataiter_list[j].next()
            data = data.to(device)
            labels = labels.to(device)
            optimizer = optimizerlist[j]
            optimizer.zero_grad()
            outputs = cur_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            modelparams.append(copy.deepcopy(model[j]))

        for k in range(num_agents):
            optimizer = optimizerlist[k]
            adjust_lr(optimizer, modelparams, iterno=iterno, alpha0=alpha0, beta0=beta0, gamma=gamma)
            optimizerlist[k] = optimizer

        if iterno % 10 == 0:
            net_test_accuracy = []
            for k in range(num_agents):
                net_test_accuracy.append(accuracy_evaluation(model[k], test_loader, device))
            test_accuracy.append(net_test_accuracy)
            print('Iteration: {}. Accuracy: {}'.format(iterno, np.asarray(net_test_accuracy)))
            iterlist.append(iterno)

        if epoch>4:
            agent_class_scores_svhn = []
            agent_class_scores_mnist = []
            for k in range(num_agents):
                agent_class_scores_svhn.append(get_scores(model[k], test_svhn_loader, device))
                agent_class_scores_mnist.append(get_scores(model[k], test_loader, device))

            class_scores_svhn.append(agent_class_scores_svhn)
            class_scores_mnist.append(agent_class_scores_mnist)

test_svhn_acc = []
test_mnist_acc = []
for k in range(num_agents):
    test_svhn_acc.append(accuracy_evaluation(model[k], test_svhn_loader, device))
    test_mnist_acc.append(accuracy_evaluation(model[k], test_loader, device))

class_pred_prob_svhn = np.asarray(class_scores_svhn)
class_pred_prob_mnist = np.asarray(class_scores_mnist)

for k in range(num_agents):

    exp_class_pred_prob_svhn = np.mean(class_pred_prob_svhn[:, k, :, :].squeeze(), axis=0)
    exp_max_pred_prob_svhn = np.max(exp_class_pred_prob_svhn, axis=1)
    weights_svhn = np.ones_like(exp_max_pred_prob_svhn) / len(exp_max_pred_prob_svhn)
    print(np.mean(exp_max_pred_prob_svhn), np.std(exp_max_pred_prob_svhn))


    exp_class_pred_prob_mnist = np.mean(class_pred_prob_mnist[:, k, :, :].squeeze(), axis=0)
    exp_max_pred_prob_mnist = np.max(exp_class_pred_prob_mnist, axis=1)
    weights_mnist = np.ones_like(exp_max_pred_prob_mnist) / len(exp_max_pred_prob_mnist)
    print(np.mean(exp_max_pred_prob_mnist), np.std(exp_max_pred_prob_mnist))
    fontsize = 10
    f1 = plt.figure()
    ax = f1.gca()
    _ = plt.hist(exp_max_pred_prob_mnist, bins=100, range=(0, 1), weights=weights_mnist, alpha=0.75, label='MNIST')
    _ = plt.hist(exp_max_pred_prob_svhn, bins=100, range=(0, 1), weights=weights_svhn, alpha=0.9, label='SVHN')
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