from  Models import LeNet5
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import MNIST
import torch.nn as nn
import  numpy as np
from torchvision.transforms import transforms
from torchvision import  datasets


def return_model(mode, **kwargs):
    if mode=='Lenet_5':
        model = LeNet5(10)
        return model


class mnist_modi(Dataset):
    def __init__(self,root,download,train,transform):
        self.mnist = datasets.MNIST(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)
    def __getitem__(self, index):
        data, target = self.mnist[index]
        return data, target, index
    def __len__(self):
        return len(self.mnist)

def crete_data(name,train_size,num_test,batch_size):
    if name=="mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32)),
                                        transforms.Normalize((0.5,), (0.5,))])
        dataset = mnist_modi(root='./data', train=True, transform=transform, download=True)
        val_set = mnist_modi(root='./data', train=False, transform=transform, download=True)
        indices = np.random.randint(0,59999,(1,train_size))
        total_range= [i for i in range(60000)]
        train_index = [i for i in indices[0,:]]
        heldout_index = list(set(total_range).difference(set(train_index)))

        train_set = torch.utils.data.Subset(dataset, train_index)
        heldout_set = torch.utils.data.Subset(dataset, heldout_index)
        return  train_set,val_set,heldout_set,dataset


def error(mem):
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)

def training(model,train_loader, criterion, optimizer, device,learning_rate):
    '''
    Function for the training step of the training loop
    '''
    if optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        model.train()
        running_loss = 0

        for X, y_true,index in train_loader:
            optimizer.zero_grad()

            X = X.to(device)
            model = model.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_hat= model(X)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        return model, optimizer, epoch_loss
def prediction_cost():
    pass
def score(model,test_set,device):
    test_loader = DataLoader(dataset=test_set,
                                  batch_size=32,
                                  shuffle=False)
    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true,index in test_loader:
            X = X.to(device)
            model = model.to(device)
            y_true = y_true.to(device)

            y_hat = model(X)
            y_prob = F.softmax(y_hat,dim=0)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def predict_proba():
        pass
def train_gshap(model,train_loader,test_set,criterion,optimizer,learning_rate,device):
    vals=[]
    indexes=[]
    if optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    model.train()
    running_loss = 0

    for X, y_true,index in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        model = model.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat= model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        vals.append(score(model,test_set,device))
        indexes.append(index)
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, np.array(vals),np.array(indexes)
def fit_gshape(model,train_set,test_set,max_epochs,criterion,optimizer,learning_rate,device):
    train_loader = DataLoader(dataset=train_set,
                                  batch_size=1,
                                  shuffle=True)

    history = {'metrics': [], 'idxs': []}
    for epoch in range(0, max_epochs):
        model, vals_metrics, idxs = train_gshap(model, train_loader,test_set, criterion, optimizer,learning_rate, device)
        history['idxs'].append(idxs)
        history['metrics'].append(vals_metrics)
    return history

def fit(train_set,model,max_epochs,criterion,optimizer,device,learning_rate):
    train_loader = DataLoader(dataset=train_set,
                              batch_size=32,
                              shuffle=True)
    train_losses = []
    # Train model
    for epoch in range(0, max_epochs):
            # training
        model, optimizer1, train_loss = training(model,train_loader ,criterion, optimizer, device,learning_rate)
        train_losses.append(train_loss)
    return model, optimizer1, train_losses

