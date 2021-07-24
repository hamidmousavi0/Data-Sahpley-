import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import copy
import torch.nn.functional as F
from shap_utils_p import *
from torch.utils.data import DataLoader,Dataset
import _pickle as pkl
from torchvision.transforms import transforms
from torchvision import  datasets
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

def crete_data(name,train_index):
    if name=="mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32)),
                                        transforms.Normalize((0.5,), (0.5,))])
        dataset = mnist_modi(root='./data', train=True, transform=transform, download=True)
        val_set = mnist_modi(root='./data', train=False, transform=transform, download=True)
        # indices = np.random.randint(0,59999,(1,train_size))
        total_range= [i for i in range(60000)]
        # train_index = [i for i in indices[0,:]]
        heldout_index = list(set(total_range).difference(set(train_index)))

        train_set = torch.utils.data.Subset(dataset, train_index)
        heldout_set = torch.utils.data.Subset(dataset, heldout_index)
        return  train_set,val_set,heldout_set,dataset
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
def value(model,test_set, metric=None):
    if metric is None:
        metric = 'accuracy'
    if metric == 'accuracy':
        return  score(model,test_set,'cpu')
def performance_plots(vals,train_set,directory,name=None,
                      num_plot_markers=20, sources=None):
    """Plots the effect of removing valuable points.

    Args:
        vals: A list of different valuations of data points each
             in the format of an array in the same length of the data.
        name: Name of the saved plot if not None.
        num_plot_markers: number of points in each plot.
        sources: If values are for sources of data points rather than
               individual points. In the format of an assignment array
               or dict.

    Returns:
        Plots showing the change in performance as points are removed
        from most valuable to least.
    """
    plt.rcParams['figure.figsize'] = 8, 8
    plt.rcParams['font.size'] = 25
    plt.xlabel('Fraction of train data removed (%)')
    plt.ylabel('Prediction accuracy (%)', fontsize=20)
    if not isinstance(vals, list) and not isinstance(vals, tuple):
        vals = [vals]
    if sources is None:
        sources = {i: np.array([i]) for i in range(len(train_set))}
    elif not isinstance(sources, dict):
        sources = {i: np.where(sources == i)[0] for i in set(sources)}
    vals_sources = [np.array([np.sum(val[sources[i]])
                              for i in range(len(sources.keys()))])
                    for val in vals]
    if len(sources.keys()) < num_plot_markers:
        num_plot_markers = len(sources.keys()) - 1
    plot_points = np.arange(
        0,
        max(len(sources.keys()) - 10, num_plot_markers),
        max(len(sources.keys()) // num_plot_markers, 1)
    )
    perfs = [_portion_performance(
        np.argsort(vals_source)[::1], plot_points, sources=sources)
        for vals_source in vals_sources]
    rnd = np.mean([_portion_performance(
        np.random.permutation(np.argsort(vals_sources[0])[::1]),
        plot_points, sources=sources) for _ in range(10)], 0)
    plt.plot(plot_points / len(train_set) * 100, perfs[-1] * 100,
             '-', lw=5, ms=10, color='b')
    if len(vals) == 3:
        plt.plot(plot_points / len(train_set) * 100, perfs[1] * 100,
                 '--', lw=5, ms=10, color='orange')
        legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
    elif len(vals) == 2:
        legends = ['TMC-Shapley ', 'LOO', 'Random']
    else:
        legends = ['TMC-Shapley ', 'Random']
    plt.plot(plot_points / len(train_set) * 100, perfs[0] * 100,
             '-.', lw=5, ms=10, color='g')
    plt.plot(plot_points / len(train_set) * 100, rnd * 100,
             ':', lw=5, ms=10, color='r')
    plt.legend(legends)
    if directory is not None and name is not None:
        plt.savefig(os.path.join(
            directory,'{}.png'.format(name)),
            bbox_inches='tight')
        plt.close()

def return_model(mode, **kwargs):
    if mode=='Lenet_5':
        model = LeNet5(10)
        return model
def _portion_performance(idxs, plot_points, sources=None):
    """Given a set of indexes, starts removing points from
    the first elemnt and evaluates the new model after
    removing each point."""
    criterion  = torch.nn.CrossEntropyLoss()
    device = 'cpu'
    learning_rate = 0.001
    model = return_model('Lenet_5')
    if sources is None:
        sources = {i: np.array([i]) for i in range(len(train_set))}
    elif not isinstance(sources, dict):
        sources = {i: np.where(sources == i)[0] for i in set(sources)}
    scores = []
    init_score = 0.11
    for i in range(len(plot_points), 0, -1):
        keep_idxs = np.concatenate([sources[idx] for idx
                                    in idxs[plot_points[i - 1]:]], -1)
        data = torch.utils.data.Subset(train_set, keep_idxs)
        model, optimizer1, train_losses = fit(data, copy.deepcopy(model), 10,
                                              criterion, 'adam', device, learning_rate)
        scores.append(value(model, test_set))  # self.heldout_set

    return np.array(scores)[::-1]
if __name__ == '__main__':
    with open('mem_tmc_0000.pkl', 'rb') as f:
        data = pkl.load(f)
        values_tmc = data['mem_tmc']
    with open('mem_g_0000.pkl', 'rb') as f:
        data = pkl.load(f)
        values_g = data['mem_g']
    with open('loo.pkl', 'rb') as f:
        data = pkl.load(f)
        values_loo = data['loo']
    index_list=[]
    with open('mem_tmc_0000.pkl', 'rb') as f:
        data = pkl.load(f)
        for i in range(len(data['idxs_tmc'])):
            index_list.append(data['idxs_tmc'][0][i])
    # index_list=sorted(index_list)
    # print(index_list)
    train_set, test_set, heldout_set, dataset = crete_data('mnist', np.array(index_list))
    vals=[values_tmc,values_g,values_loo]
    directory='./'
    performance_plots(vals, train_set, directory, name='result',
                      num_plot_markers=20, sources=None)

