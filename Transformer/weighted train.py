import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from utils import FEATURE_ARCHIVE


class mydataset_w_weight(Dataset):
    def __init__(self, X, Y, Weights):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)
        self.weights = torch.Tensor(Weights)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item], self.weights[item]

    def __len__(self):
        return len(self.Label)


def prepare_data_w_weight(test_ratio=0.2, weights=None, seed=20220712):
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp951,npz", allow_pickle=True)
    X_all = all_data["X"]
    Y_all = all_data["Y"]
    X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)

    # split data
    np.random.seed(seed)
    n_samples = X_all.shape[0]
    test_size = int(n_samples * test_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    X_train, Y_train, weights_train = X_all[train_idx], Y_all[train_idx], weights[train_idx]
    X_test, Y_test, weights_test = X_all[test_idx], Y_all[test_idx], weights[test_idx]

    # create dataset and dataloader
    train_dataset = mydataset_w_weight(X_train, Y_train, weights_train)
    test_dataset = mydataset_w_weight(X_test, Y_test, weights_test)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    return trainloader, testloader
