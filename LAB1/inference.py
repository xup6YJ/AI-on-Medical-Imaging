import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sn

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_confusion_matrix(confusion_matrix, model_name):
    # TODO plot confusion matrix
    # c_matrix = [[int(tp), int(fn)], [int(fp), int(tn)]]  best_c_matrix
    # result = df.iloc[:,:4].astype(int)
    array = np.array(confusion_matrix)
    # array1 = array.reshape((2,2))
    ind1 = ("Pneumonia", 'N Pneumonia')
    ind2 = ("Pneumonia", 'N Pneumonia')
    df_cm = pd.DataFrame(array, index = [i for i in ind2],
                  columns = [i for i in ind1])
    
    fig, ax1 = plt.subplots(1,1, figsize = (10, 9), dpi = 80)
    sn.heatmap(df_cm, annot=True,cmap="Blues", fmt='g')
    title = 'Pneumonia Classification'
    plt.title(title, fontsize=18)
    plt.ylabel('Predictions', fontsize=18)
    plt.xlabel('Actuals', fontsize=18)
    fig.savefig('{}_{}_Confusion Matrix.jpeg'.format(model_name, args.optimizer))

def test(test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in tqdm(test_loader):
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Test Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model_name', type=str, required=False, default='ResNet101')
    parser.add_argument('--optimizer', type=str, required=False, default='Adam')
    parser.add_argument('--model_path', type=str, required=True, default='/home/yingchihlin/Documents/Lesson/MedAI/EXP1/91.9872_ResNet18_weights.pt')

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=20)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # define model
    if args.model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)
    elif args.model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
    elif args.model_name == 'ResNet101':
        model = models.resnet101(pretrained=True)

    num_neurons = model.fc.in_features
    model.fc = nn.Linear(num_neurons, args.num_classes)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    Evaluate = True
    if Evaluate:
        model_path = args.model_path
        # checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(torch.load(model_path))
        # model = torch.load(model_path)
        print('Model Load Successfully')
        
        val_acc, f1_score, c_matrix = test(test_loader, model)

    plot_confusion_matrix(c_matrix, model_name = args.model_name)

