import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mnist_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
           ])

def fashion_mnist(batch_size=50, valid=0, shuffle=True, transform=mnist_transform, path='./FashionMNIST_data'):
    test_data = datasets.FashionMNIST(path, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    train_data = datasets.FashionMNIST(path, train=True, download=True, transform=transform)
    if valid > 0:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = num_train-valid
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    
        return train_loader, valid_loader, test_loader
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        return train_loader, test_loader


def fashion_mnist_plotdata(transform=mnist_transform, path='./FashionMNIST_data'):
    train_data = datasets.FashionMNIST(path, train=True, download=True, transform=transform)
    images = [train_data[i][0] for i in range(50)]
    return images


def plot_fashion_mnist(images, shape):
    fig = plt.figure(figsize=shape[::-1], dpi=80)
    for j in range(1, len(images) + 1):
        ax = fig.add_subplot(shape[0], shape[1], j)
        ax.matshow(images[j - 1][0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()


def plot_results(m, epoch_train_loss_np, epoch_test_loss_np, epoch_test_acc_np):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title("Model {} - Loss".format(m))
    plt.plot(epoch_train_loss_np.T[m], color='green', label='Train loss')
    plt.plot(epoch_test_loss_np.T[m], color='red', label='Test loss')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title("Model {} - Accuracy".format(m))
    plt.plot(epoch_test_acc_np.T[m], color='blue', label='Accuracy')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.legend()
