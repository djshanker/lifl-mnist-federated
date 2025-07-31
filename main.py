
import torch
import torchvision
import torchvision.transforms as transforms
from models.mnist_cnn import MNIST_CNN
from clients.client_base import Client
from utils.metrics import record_metrics, plot_results
import random

NUM_CLIENTS = 10
EPOCHS = 1
ROUNDS = 5

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

data_size = len(trainset) // NUM_CLIENTS
clients = []
for i in range(NUM_CLIENTS):
    indices = list(range(i * data_size, (i + 1) * data_size))
    subset = torch.utils.data.Subset(trainset, indices)
    clients.append(Client(i, subset, MNIST_CNN, device='cpu'))

metrics_log = {}

for rnd in range(ROUNDS):
    print(f'--- Round {rnd+1} ---')
    state_dicts = []
    losses, accs = [], []
    for client in clients:
        weights, loss, acc = client.train(epochs=EPOCHS)
        state_dicts.append(weights)
        losses.append(loss)
        accs.append(acc)

    # Average model
    global_weights = state_dicts[0]
    for k in global_weights.keys():
        for i in range(1, len(state_dicts)):
            global_weights[k] += state_dicts[i][k]
        global_weights[k] = torch.div(global_weights[k], len(state_dicts))

    for client in clients:
        client.model.load_state_dict(global_weights)

    record_metrics(rnd+1, losses, accs)
    metrics_log[rnd+1] = accs

plot_results(metrics_log)
print("✅ Training complete. Metrics logged and results.png saved.")
