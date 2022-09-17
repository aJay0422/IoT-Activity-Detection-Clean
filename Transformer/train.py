import torch
import torch.nn as nn
import os

from utils import get_loss_acc, prepare_data
from model import transformer_base, transformer_large, transformer_huge


def train(model, epochs, trainloader, testloader, optimizer, criterion, save_path):
    """
    Train a neural network and save the best model on accoding to its performance on testloader
    :param model: the model to be trained
    :param epochs: number of epochs for training
    :param trainloader: train dataloader
    :param testloader: test dataloader
    :param optimizer: the optimizer for gradient descent
    :param criterion: the loss function
    :param save_path: the path for saving model parameters
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Trained on {}".format(device))
    model.to(device)

    # train model
    best_test_acc = 0
    model.train()
    for epoch in range(epochs):
        for batch in trainloader:
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            if len(batch) == 3:
                index_batch = batch[2]   # in case we need the index of samples in this batch

            # forward
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            model.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, criterion)
            # evaluate test
            test_loss, test_acc = get_loss_acc(model, trainloader, criterion)

        print("Epoch {}/{} train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(epoch+1, epochs,
                                                                                              train_loss, test_loss,
                                                                                              train_acc, test_acc))

        # save model weights if it's the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Saved")


def train_transfomer(size="base"):
    """
    Train a transformer model 5 times with different train test split
    :param size: Size of the transformer model. "base" or "large" or "huge"
    """
    seed = 20220728
    model_save_dir = "./model_weights/"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    for i in range(5):
        this_seed = seed + i
        save_path = model_save_dir + "Transformer_{}_{}.pth".format(size, i+1)
        trainloader, testloader = prepare_data(test_ratio=0.2, seed=this_seed)

        # train model
        if size == "base":
            model = transformer_base()
        elif size == "large":
            model = transformer_large()
        elif size == "huge":
            model = transformer_huge()

        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        train(model, epochs, trainloader, testloader, optimizer, criterion, save_path)


if __name__ == "__main__":
    train_transfomer()

