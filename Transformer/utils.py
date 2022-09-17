import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np


FEATURE_ARCHIVE = "../feature_archive/"



class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)


def prepare_data(test_ratio=0.2, seed=20220712):
    """
    Prepare two dataloader for train and test. The dataset contains 951 samples.
    :param test_ratio: Proportion of test data
    :param seed: random seed for train test split
    :return: trainloader, testloader
    """
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp951.npz")
    X_all = all_data["X"]
    Y_all = all_data["Y"]

    # reshape data to (n_samples, n_features, length)
    X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)

    # train test split
    np.random.seed(seed)
    n_samples = X_all.shape[0]
    test_size = int(n_samples * test_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]
    X_test = X_all[test_idx]
    Y_test = Y_all[test_idx]

    # create dataset and dataloader
    train_dataset = mydataset(X_train, Y_train)
    test_dataset = mydataset(X_test, Y_test)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    return trainloader, testloader


def get_loss_acc(model, dataloader, criterion=nn.CrossEntropyLoss()):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            if len(batch) == 3:
                index_batch = batch[2]   # in case we need the index of samples in this batch
            total += len(Y_batch)
            num_batches += 1
            logits = model(X_batch)
            y_pred = torch.argmax(logits, dim=1)
            correct += torch.sum(y_pred == Y_batch).cpu().numpy()
            loss = criterion(logits, Y_batch)
            total_loss += loss.item()
    acc = correct / total
    total_loss = total_loss / num_batches

    return total_loss, acc


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(count))
    print("Trainable parameters: {}".format(trainable))
    return count, trainable


