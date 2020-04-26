from pathlib import Path
import requests
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from torch import nn

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
URL = "http://deeplearning.net/data/mnist/"


def load_data():
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
            return pickle.load(f, encoding="latin-1")

def download():
    PATH.mkdir(parents=True, exist_ok=True)

    if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(input, target):
    x = -input[range(target.shape[0]), target]
    return x.mean()

loss_func = nll

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


if __name__ == '__main__':
    # download()
    ((x_train, y_train), (x_valid, y_valid), _) = load_data()
    plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    # plt.show()
    print(x_train.shape)
    # turn the data to torch.tensor
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape
    x_train, x_train.shape, y_train.min(), y_train.max()
    print(x_train, y_train)
    print(x_train.shape)
    print(y_train.min(), y_train.max())

    weights = torch.randn(784, 10) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)

    bs = 64  # batch size

    xb = x_train[0:bs]  # a mini-batch from x
    preds = model(xb)  # predictions
    print(preds[0], preds.shape)

    yb = y_train[0:bs]
    print(loss_func(preds, yb))




