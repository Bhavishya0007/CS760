import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt
mnist_data_train = torchvision.datasets.MNIST('.', train=True,download=True, transform=ToTensor())
#train_data_loader = torch.utils.data.DataLoader(mnist_data_train, batch_size=1, shuffle=False)
mnist_data_test = torchvision.datasets.MNIST('.', train=False,download=True)
#train_features, train_labels = next(iter(train_data_loader))
#print(f"Label: {d}")
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")

d = 28*28; d1 = 300; d2 = 200;  k = 10; 
W1 = np.random.uniform(-1,1,[d1 , d]);
W2 = np.random.uniform(-1,1,[d2, d1]);
W3 = np.random.uniform(-1,1,[k, d2]);
W1_grad = np.zeros([d1 , d]);
W2_grad = np.zeros([d2, d1]);
W3_grad = np.zeros([k, d2]);
z1 = np.zeros([d1, 1]);
a1 = np.zeros([d1, 1]);
z2 = np.zeros([d2, 1]);
a2 = np.zeros([d2, 1]);
z3 = np.zeros([k, 1]);
soft_max = np.zeros([k, 1]);
alpha = 0.1;
def backprop(train_label,W1,W2,W3,train_data_np,a1,a2):
    y = np.zeros([k,1]);
    y_cap_bar = np.zeros([k,1]);
    y[train_label] = 1;
    y_cap_bar = y-soft_max;
    y_cap = np.matmul(y.T,soft_max)
    #print("y_cap =",y_cap)
    #W3 grads
    W3_grad = -np.matmul(y_cap_bar,a2.T);
    #print(W3_grad);
    #print("W3 shape :", np.shape(W3_grad));
    #W2 grad
    #print(np.shape(np.matmul(W3.T,np.matmul(y_cap_bar,a1.T))))
    W2_grad = -np.matmul(np.diag(a2.squeeze()*(1-a2.squeeze())),np.matmul(W3.T,np.matmul(y_cap_bar,a1.T)));
    #W1 grad
    #print("W1_grad:",np.matmul(y_cap_bar,train_data_np.T),"y_cap_bar :", y_cap_bar,train_data_np.T)
    W1_grad = -np.matmul(np.diag(a1.squeeze()*(1-a1.squeeze())),np.matmul(W2.T,np.matmul(W3.T,np.matmul(y_cap_bar,train_data_np.T))));
    return [W1_grad, W2_grad, W3_grad];
    

epochs = 100;
train_data_len = 100;
Loss = np.zeros(epochs*train_data_len);
for i in range(epochs):
    #for data in mnist_data_train[:100]:
    for j in range(train_data_len):
        train_data, train_label =  mnist_data_train[j];
        #print(train_label)
        train_data_np = np.ndarray.flatten(train_data.numpy()).reshape(-1,1);
        #print(np.shape(train_data_np))
        #first layer
        z1 = np.matmul(W1,train_data_np);
        a1 = 1/(1+np.exp(-z1));
        #print("a1:",a1);
        #print(np.shape(z1),np.shape(a1));
         #second layer
        z2 = np.matmul(W2,a1);
        a2 = 1/(1+np.exp(-z2));
        #print(np.shape(z2),np.shape(a2));
         #third layer
        z3 = np.matmul(W3,a2);
        soft_max = np.exp(z3)/np.sum(np.exp(z3));
        y = np.zeros([k,1]);
        y[train_label] = 1;
        y_cap = np.matmul(y.T,soft_max)
        #print("y_cap:",y_cap,"y =",y,"softmax =", soft_max)
        Loss[i*train_data_len+j] = -np.log(y_cap)
        #print("Loss :", Loss);
        #a3 = 1/(1+np.exp(-z3));
        #print(np.shape(z3),np.shape(soft_max));
        #print("diag :",np.shape(np.diag(a2.squeeze()*(1-a2.squeeze()))))
        #print("Before Backprop:", W1)
        [W1_grad, W2_grad, W3_grad] = backprop(train_label,W1,W2,W3,train_data_np,a1,a2);
        #print("After Backprop:",W1);
        #Weight Update
        W1 = W1 - alpha*W1_grad;
        W2 = W2 - alpha*W2_grad;
        W3 = W3 - alpha*W3_grad;
        #print("After Backprop:",W1)