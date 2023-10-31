import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
from torch import nn

learning_rate = 0.01
batch_size = 32

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, 300),
            nn.Sigmoid(),
            nn.Linear(300, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        #Y = nn.Softmax(dim=1)(logits)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer, data_size):
    size = len(dataloader.dataset)
    #print(size)
    i = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        #print(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            #print(batch)
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        i += batch_size
        if i >= data_size-1:
            break


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print(size)
    num_batches = len(dataloader)
    print(num_batches)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

mnist_data_train = torchvision.datasets.MNIST('.', train=True,download=True, transform=ToTensor())
train_data_loader = torch.utils.data.DataLoader(mnist_data_train, batch_size=batch_size, shuffle=False)
mnist_data_test = torchvision.datasets.MNIST('.', train=False,download=True, transform=ToTensor())
test_data_loader = torch.utils.data.DataLoader(mnist_data_test, batch_size=1, shuffle=False)


#train_features, train_labels = next(iter(train_data_loader))
model = NeuralNetwork()
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#logits = model(train_features)
#pred_probab = nn.Softmax(dim=1)(logits)
#y_pred = pred_probab.argmax(1)
#print(f"Predicted class: {y_pred}")

mis_pred_list = []
dataset_l = [10, 100, 500, 1000, 10000, 30000, 60000]
epochs = 20
for dataset in dataset_l:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_data_loader, model, loss_fn, optimizer, dataset)
    correct = test_loop(test_data_loader, model, loss_fn)
    mis_pred_list.append(correct)
print("Done!")

plt.plot(dataset_l, mis_pred_list)
plt.xscale("log")
plt.grid()
plt.savefig('pytorch_nn_learning_curve.pdf', format='pdf')