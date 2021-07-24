from DShap_p import DShap
from shap_utils_p import *
import matplotlib.pyplot as plt
import torch
MEM_DIR = './'
train_size = 100
num_test = 1000
max_epochs=20
criterion=torch.nn.CrossEntropyLoss()
learning_rate=0.001
model = LeNet5(10)
optimizer= 'adam'
device='cpu'
mode = 'Lenet-5'
for _ in range(100):
    model = LeNet5(10)
    train_set, test_set, heldout_set, dataset = crete_data('mnist', train_size, num_test, 32)
    model, optimizer1, train_losses=fit(train_set,model,max_epochs,criterion,optimizer,device,learning_rate)
    accuracy = score(model, test_set, device)
    print(accuracy)
    if accuracy>0.6:
        break
directory = './temp'
dshap = DShap(dataset,train_set,test_set,heldout_set, num_test,
              sources=None,
              sample_weight=None,
              model_family=mode,
              metric='accuracy',
              overwrite=True,
              directory=directory, seed=0)
dshap.run(100, 0.05, g_run=True)

# # X, y = X_raw[:100], y_raw[:100]
# # X_test, y_test = X_raw[100:], y_raw[100:]
# # model = 'NN'
# # problem = 'classification'
# # num_test = 1000
# # directory = './temp'
# # dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
# #               directory=directory, seed=1)
# # dshap.run(100, 0.1)
# #
# #
# # X, y = X_raw[:100], y_raw[:100]
# # X_test, y_test = X_raw[100:], y_raw[100:]
# # model = 'NN'
# # problem = 'classification'
# # num_test = 1000
# # directory = './temp'
# # dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
# #               directory=directory, seed=2)
# # dshap.run(100, 0.1)
# #
# #
dshap.merge_results()
# #
# # convergence_plots(dshap.marginals_tmc)
# #
# # convergence_plots(dshap.marginals_g)
# #
# #
dshap.performance_plots([dshap.vals_g,dshap.vals_tmc, dshap.vals_loo], num_plot_markers=20,
                       sources=dshap.sources,name="result")
plt.show()