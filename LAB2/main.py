import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet, EEGNet_2
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

# def plot_train_acc(train_acc_list, epochs):
#     # TODO plot training accuracy
#     pass

# def plot_train_loss(train_loss_list, epochs):
#     # TODO plot training loss
#     pass

# def plot_test_acc(test_acc_list, epochs):
#     # TODO plot testing loss
#     pass

def plot_train_acc(train_acc_list, epochs):
    # TODO plot training and testing accuracy curve
    plt.figure()
    epochs = range(1, epochs +1)
    plt.plot(epochs, train_acc_list, 'b', label = 'Train acc')
    plt.title('Train Accuracy')
    plt.legend()
    # plt.figure()
    plt.savefig('./Train_drop{}_{}_{}_Accuracy.jpeg'.format(args.dropout_rate, args.activation_function, args.elu_alpha))
    plt.close()


def plot_test_acc(test_acc_list, epochs):
    # TODO plot training and testing accuracy curve
    plt.figure()
    epochs = range(1, epochs +1)
    plt.plot(epochs, test_acc_list, 'b', label = 'Test acc')
    plt.title('Test Accuracy')
    plt.legend()
    # plt.figure()
    plt.savefig('./Test_drop{}_{}_{}_Accuracy.jpeg'.format(args.dropout_rate, args.activation_function, args.elu_alpha))
    plt.close()


def plot_train_loss(train_loss_list, epochs):
    # TODO plot training and testing accuracy curve
    plt.figure()
    epochs = range(1, epochs +1)
    plt.plot(epochs, train_loss_list, 'b', label = 'Train loss')
    plt.title('Train Loss')
    plt.legend()
    # plt.figure()
    plt.savefig('./Train_drop{}_{}_{}_Loss.jpeg'.format(args.dropout_rate, args.activation_function, args.elu_alpha))
    plt.close()


def train(model, loader, criterion, optimizer, args, scheduler):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    for epoch in tqdm(range(1, args.num_epochs+1)):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            
            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')


        test_acc = test(model, test_loader)
        scheduler.step(test_acc)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
            torch.save(best_wts, './weights/EEGNet_drop{}_{}_{}_{:.2f}.pt'.format(args.dropout_rate, args.activation_function, args.elu_alpha, test_acc))
        print(f'Test Acc. (%): {test_acc:3.2f}%')


    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-dropout_rate", type=float, default=0.25)
    parser.add_argument("-activation_function", type=str, default='elu')
    parser.add_argument("-elu_alpha", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = EEGNet(args=args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=1e-3, 
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    model.to(device)
    criterion.to(device)

    train_acc_list, train_loss_list, test_acc_list = train(model, train_loader, criterion, optimizer, args, scheduler)

    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_test_acc(test_acc_list, args.num_epochs)
