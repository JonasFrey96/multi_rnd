# Import Libraries


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchmetrics import AUROC
import torchmetrics
import matplotlib.pyplot as plt
import os
from scipy import stats
import copy
import numpy as np
from tqdm import tqdm
args = {}
kwargs = {}
args["batch_size"] = 1000
args["test_batch_size"] = 1000

# The number of Epochs is the number of times you go through the full dataset.
args["epochs"] = 10
args["lr"] = 0.01  # Learning rate is how fast it will decend.

# SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args["momentum"] = 0.5
args["nr_seeds"] = 3  # random seed
args["log_interval"] = 10

# if number = 1 only digit 0 is tested, 2 -> [0,1] is tested
args["nr_digits_to_test"] = 10

device = "cuda"

root_folder = "/media/jfrey/git/multi_rnd/results"
run_name = "all_datasets"


class TwinNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layer: list, bn: bool):
        super(TwinNet, self).__init__()

        ls = []
        input_dim_running = input_dim
        for j, h in enumerate(hidden_layer):
            ls.append(nn.Linear(input_dim_running, h))

            if j != len(hidden_layer) - 1:
                # Do not add activation for last layer
                if bn:
                    ls.append(nn.BatchNorm1d(h))
                ls.append(nn.ReLU())

            input_dim_running = h

        self.net = nn.Sequential(*ls)

        new_ls = []
        input_dim_running = input_dim
        for j, h in enumerate(hidden_layer):
            new_ls.append(nn.Linear(input_dim_running, h))

            if j != len(hidden_layer) - 1:
                # Do not add activation for last layer
                if bn:
                    new_ls.append(nn.BatchNorm1d(h))
                new_ls.append(nn.ReLU())

            input_dim_running = h

        self.random_net = nn.Sequential(*new_ls)

        for param in self.random_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def random_forward(self, x):
        return self.random_net(x)


class SimpleModule:
    def __init__(
        self,
        input_dim: int,
        hidden_layer: list,
        bn: bool,
        lr: float,
        in_distribution: int,
    ):
        self.model = TwinNet(input_dim=input_dim, hidden_layer=hidden_layer, bn=bn)
        self.optimizer = optim.SGD(self.model.net.parameters(), lr=lr)
        self.depth = len(hidden_layer)
        self.metric = AUROC(task="binary", thresholds=10000)
        self.mean_metric = torchmetrics.aggregation.MeanMetric()
        self.in_distribution = in_distribution

    def training_step(self, batch):
        data, target = batch

        BS = data.shape[0]
        data = data.reshape(BS, -1)
        target = target.reshape(BS, -1)

        self.optimizer.zero_grad()

        m = (target == self.in_distribution)[:, 0]

        target_values = self.model.random_forward(data[m].clone())
        pred_values = self.model.forward(data[m].clone())

        loss = F.mse_loss(pred_values, target_values, reduction="mean")

        loss.backward()

        self.optimizer.step()

        return loss

    @torch.no_grad()
    def test_step(self, batch):
        data, target = batch

        BS = data.shape[0]
        data = data.reshape(BS, -1)
        target = target.reshape(BS, -1)

        target_values = self.model.random_forward(data.clone())
        pred_values = self.model.forward(data.clone())

        loss = F.mse_loss(pred_values, target_values, reduction="none")
        loss = loss.mean(dim=1)

        return loss


def train(epoch):
    for m in modules:
        m.model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Variables in Pytorch are differenciable.
        data, target = Variable(data), Variable(target)
        # This will zero out the gradients for this batch.
        batch = (data, target)
        for m in modules:
            loss = m.training_step(batch)

            # if batch_idx % 10 == 0:
            #     print( f"Module Depth {m.depth}: {loss.item()}")


def test(data_loader):
    for m in modules:
        m.model.eval()

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)

        batch = (data, target)

        for m in modules:
            loss = m.test_step(batch)

            target = target == 0

            m.mean_metric(loss[target])
            # This is sketchy given that the normalization is now done per batch
            # In later iteration define normalization over the full dataset instead
            loss = loss / loss.max()

            m.metric(loss, target)


def init_ls(nr, ele):
    return [copy.deepcopy(ele) for _ in range(nr)]


train_loaders, test_loaders = [], []

# load the data
for d in [datasets.MNIST, datasets.KMNIST, datasets.FashionMNIST]:
    train_loaders.append(
        torch.utils.data.DataLoader(
            d(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args["batch_size"],
            shuffle=True,
            **kwargs,
        )
    )
    test_loaders.append(
        torch.utils.data.DataLoader(
            d(
                "../data",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args["test_batch_size"],
            shuffle=True,
            **kwargs,
        )
    )
pretty_loader_names = ["MNIST", "KMNIST", "FashionMNIST"]

red_shades = ["salmon", "darkred", "firebrick", "indianred", "lightcoral"]
blue_shades = ["cyan", "darkblue", "mediumblue", "royalblue", "deepskyblue"]

# 1. Dataloaders
for train_loader, test_loader, pretty_loader_name in zip(
    train_loaders, test_loaders, pretty_loader_names
):
    p = os.path.join(root_folder, run_name, pretty_loader_name)
    os.makedirs(p, exist_ok=True)

    # 2. Task Setting (which digit to use for training)
    for nr in range(args["nr_digits_to_test"]):
        nr_modules = 3
        train_auroc_data = init_ls(nr_modules, [[] for _ in range(args["nr_seeds"])])
        test_auroc_data = init_ls(nr_modules, [[] for _ in range(args["nr_seeds"])])
        train_mse_data = init_ls(nr_modules, [[] for _ in range(args["nr_seeds"])])
        test_mse_data = init_ls(nr_modules, [[] for _ in range(args["nr_seeds"])])

        # 3. Run multiple random seeds
        for seed in tqdm(range(args["nr_seeds"]), colour = 'GREEN', desc="Seeds"):
            torch.manual_seed(seed)

            modules = []
            for i in list(range(1, nr_modules + 1))[::-1]:
                modules.append(
                    SimpleModule(
                        input_dim=784,
                        hidden_layer=[64] * i,
                        bn=False,
                        lr=0.01,
                        in_distribution=nr,
                    )
                )

            # modules.append( SimpleModule(input_dim = 784, hidden_layer = [64, 64 ,64, 64], bn=False, lr= 0.01, in_distribution=nr) )
            # modules.append( SimpleModule(input_dim = 784, hidden_layer = [64, 64 ,64], bn=False, lr= 0.01, in_distribution=nr) )
            # modules.append( SimpleModule(input_dim = 784, hidden_layer = [64 ,64], bn=False, lr= 0.01, in_distribution=nr) )
            # modules.append( SimpleModule(input_dim = 784, hidden_layer = [64], bn=False, lr= 0.01, in_distribution=nr) )

            for m in modules:
                m.model.to(device)
                m.metric.to(device)
                m.mean_metric.to(device)

            for epoch in tqdm(range(args["epochs"]), colour = 'BLUE', desc="Epochs"):
                # 4. Run multiple epochs

                test(data_loader=train_loader)
                tqdm.write(f"Evaluation Result Training-Dataset Epoch: {epoch:>4}  Seed {seed:>4}  Task: {nr:>4}")
                for idx, m in enumerate(modules):
                    tqdm.write(
                        f"Module Depth {m.depth}   AUROC: {m.metric.compute().item():.3f}   MSE on Target Examples: {m.mean_metric.compute().item():.3f}"
                    )
                    train_auroc_data[idx][seed].append(m.metric.compute().item())
                    train_mse_data[idx][seed].append(m.mean_metric.compute().item())

                    m.mean_metric.reset()
                    m.metric.reset()

                test(data_loader=test_loader)
                tqdm.write(f"Evaluation Result Test-Dataset Epoch: {epoch:>4}  Seed {seed:>4}  Task: {nr:>4}")
                for idx, m in enumerate(modules):
                    tqdm.write(
                        f"Module Depth {m.depth}   AUROC: {m.metric.compute().item():.3f}   MSE on Target Examples: {m.mean_metric.compute().item():.3f}"
                    )
                    test_auroc_data[idx][seed].append(m.metric.compute().item())
                    test_mse_data[idx][seed].append(m.mean_metric.compute().item())

                    m.mean_metric.reset()
                    m.metric.reset()

                tqdm.write(" ")
                tqdm.write(" ")
                train(epoch)

        # Create plots for each run per digit
        plt.figure(figsize=(12, 8))

        def plot_with_confi(data, color, label):
            data_array = np.array(data)
            # Calculate mean and confidence intervals
            mean_values = np.mean(data_array, axis=0)
            confidence_intervals = stats.t.interval(
                0.95,
                len(data_array[0]),
                loc=np.mean(data_array, axis=0),
                scale=stats.sem(data_array, axis=0),
            )
            # Plotting
            epochs = range(0, args["epochs"])

            mean = round(mean_values[-1], 4)
            plt.plot(epochs, mean_values, color=color, label=label + f" {mean}")
            plt.fill_between(
                epochs,
                confidence_intervals[0],
                confidence_intervals[1],
                color=color,
                alpha=0.2,
            )

        # AUROC plots
        plt.subplot(2, 1, 1)
        for idx, m in enumerate(modules):
            plot_with_confi(
                train_auroc_data[idx],
                color=red_shades[idx],
                label=f"Module Depth {m.depth} - Train AUROC",
            )
            plot_with_confi(
                test_auroc_data[idx],
                color=blue_shades[idx],
                label=f"Module Depth {m.depth} - Test AUROC",
            )

        seeds = args["nr_seeds"]
        plt.title(f"AUROC Target Digit {nr} - Random Seeds {seeds}")
        plt.xlabel("Epochs")
        plt.ylabel("AUROC")
        plt.legend(loc='center right')

        # MSE plots
        plt.subplot(2, 1, 2)
        for idx, m in enumerate(modules):
            plot_with_confi(
                train_mse_data[idx],
                color=red_shades[idx],
                label=f"Module Depth {m.depth} - Train MSE",
            )
            plot_with_confi(
                test_mse_data[idx],
                color=blue_shades[idx],
                label=f"Module Depth {m.depth} - Test MSE",
            )

        plt.title(f"MSE Target Digit {nr} - Random Seeds {seeds}")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend(loc='center right')
        plt.tight_layout()
        plt.savefig(os.path.join(p, f"performance_plot_{nr}.png"))
        plt.close()
        # plt.show()
