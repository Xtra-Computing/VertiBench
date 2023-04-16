from typing import Callable
import argparse
import warnings

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import MLP

from tqdm import tqdm

from dataset import LocalDataset, VFLRawDataset, VFLAlignedDataset
from utils import get_device_from_gpu_id, get_metric_from_str


class SplitMLP(nn.Module):
    def __init__(self, local_input_channels, local_hidden_channels, agg_hidden_channels,
                 out_activation=nn.Sigmoid, **kwargs):
        """
        SplitNN that all layers are fully connected layers (MLP).
        Usage: (f is the number of features)
        -------------------- Examples ----------------------------
        - SplitMLP(local_layers=[[10], [10,20]], agg_layers=[50], output_dim=1)
        | 2 parties, party 0 has 1 hidden layer with 10 neurons, party 1 has 1 hidden layer with 20 neurons,
        | the aggregation layer has 50 neurons, and the output layer has 1 neuron.
        |            50 x 1
        |               |
        |            30 x 50
        |               |
        |            concat
        |            /     \
        |        f x 10   10 x 20
        |           /       \
        |        data1    f x 10
        |                    \
        |                   data2
        |
        - SplitMLP(local_layers=[[10]] * 3, agg_layers=[], output_dim=1)
        | 3 parties, each party has 1 hidden layers with 10 neurons, and the output layer has 1 neuron.
        |            30 x 1
        |               |
        |            concat
        |         /     |      \
        |     f x 10  f x 10  f x 10
        |        /      |       \
        |     data1   data2    data3
        |
        ---------------------------------------------------------

        Parameters
        ----------
        local_input_channels : list[int]
            list of local input channels. Same as the number of features of each party.
        local_hidden_channels : list[list[int]]
            list of local hidden channels. Note that each sublist must be non-empty, because each party must have at
            least one hidden layer to avoid directly transferring the data to the primary party.
        agg_hidden_channels : list[int]
            list of aggregation channels. Note that this list must be non-empty, because the aggregation layer must
            output a prediction result. The last element of this list is the number of output channels.
        output_channel : int
            output channel of the model
        out_activation : Callable
            activation function of the output layer
        kwargs : dict
            other parameters for torchvision.ops.MLP
            For example:
            - hidden_activation: nn.ReLU
            - dropout: 0.0
            ...
        """
        super().__init__()
        self.local_input_channels = local_input_channels
        self.local_hidden_channels = local_hidden_channels
        self.agg_hidden_channels = agg_hidden_channels
        self.out_activation = out_activation
        self.n_parties = len(local_input_channels)
        assert len(local_input_channels) == len(local_hidden_channels), \
            f"The number of parties must be the same. Got {len(local_input_channels)} and {len(local_hidden_channels)}."

        self.local_mlps = nn.ModuleList()
        for i in range(self.n_parties):
            local_mlp = MLP(local_input_channels[i], local_hidden_channels[i], **kwargs)
            self.local_mlps.append(local_mlp)

        self.cut_dim = sum([channel[-1] for channel in local_hidden_channels])  # the dimension of the cut layer
        self.agg_mlp = MLP(self.cut_dim, agg_hidden_channels, **kwargs)

    def forward(self, Xs: list[torch.Tensor]):
        """
        Forward propagation of the model.

        Parameters
        ----------
        Xs : list[torch.Tensor]
            list of local data. Each element is a tensor of shape (batch_size, local_input_channels[i])

        Returns
        -------
        torch.Tensor
            output of the model. Shape: (batch_size, output_channel)
        """
        local_outputs = [mlp(Xi) for mlp, Xi in zip(self.local_mlps, Xs)]
        agg_input = torch.cat(local_outputs, dim=1)
        agg_output = self.agg_mlp(agg_input)
        return self.out_activation(agg_output)


# train the model
def fit(model, optimizer, loss_fn, metric_fn, train_loader, test_loader=None, epochs=10, gpu_id=0, n_classes=1,
        task='bin-cls'):
    device = get_device_from_gpu_id(gpu_id)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_pred_y = torch.zeros([0, 1]).to(device)
        train_y = torch.zeros([0, 1]).to(device)
        total_loss = 0
        for Xs, y in tqdm(train_loader):
            # to device
            Xs = [Xi.to(device) for Xi in Xs]
            y = y.to(device)
            if task in ['multi-cls']:
                y = y.long()

            optimizer.zero_grad()
            y_pred = model(Xs)
            if task in ['reg', 'bin-cls']:
                y_pred = y_pred.flatten()
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            if n_classes == 1:
                y_pred_res = y_pred.reshape(-1, 1)
                y = y.reshape(-1, 1)
            elif n_classes == 2:
                y_pred_res = torch.round(y_pred).reshape(-1, 1)
                y = y.reshape(-1, 1)
            else:
                y_pred_res = torch.argmax(y_pred, dim=1).reshape(-1, 1)
                y = y.reshape(-1, 1)
            train_pred_y = torch.cat([train_pred_y, y_pred_res], dim=0)
            train_y = torch.cat([train_y, y], dim=0)
            loss.backward()
            optimizer.step()

        train_score = metric_fn(train_y.data.cpu().numpy(), train_pred_y.data.cpu().numpy())
        print(f"Epoch: {epoch}, Train Loss: {total_loss / len(train_loader)}, Train Score: {train_score}")

        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                test_y_pred = torch.zeros([0, 1]).to(device)
                test_y = torch.zeros([0, 1]).to(device)
                for Xs, y in test_loader:
                    # to device
                    Xs = [Xi.to(device) for Xi in Xs]
                    y = y.to(device)
                    if task in ['multi-cls']:
                        y = y.long()
                    y_pred = model(Xs)
                    if task in ['reg', 'bin-cls']:
                        y_pred = y_pred.flatten()
                    test_loss += loss_fn(y_pred, y)
                    if n_classes == 1:
                        y_pred_res = y_pred.reshape(-1, 1)
                        y = y.reshape(-1, 1)
                    elif n_classes == 2:
                        y_pred_res = torch.round(y_pred).reshape(-1, 1)
                        y = y.reshape(-1, 1)
                    else:
                        y_pred_res = torch.argmax(y_pred, dim=1).reshape(-1, 1)
                        y = y.reshape(-1, 1)
                    test_y_pred = torch.cat([test_y_pred, y_pred_res], dim=0)
                    test_y = torch.cat([test_y, y], dim=0)
                test_score = metric_fn(test_y.data.cpu().numpy(), test_y_pred.data.cpu().numpy())
                print(f"Epoch: {epoch}, Test Loss: {test_loss / len(test_loader)}, Test Score: {test_score}")


# evaluate the model on the test set
def evaluate(model, test_loader, metric_fn: Callable, gpu_id=0, n_classes=1):
    device = get_device_from_gpu_id(gpu_id)
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_all = torch.zeros([0, 1]).to(device)
        y_pred_all = torch.zeros([0, 1]).to(device)
        for Xs, y in test_loader:
            # to device
            Xs = [Xi.to(device) for Xi in Xs]
            y = y.to(device).reshape(-1, 1)
            y_pred = model(Xs)
            if n_classes == 1:
                y_pred = y_pred.reshape(-1, 1)
            else:
                y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)
            y_pred_all = torch.cat((y_pred_all, y_pred), dim=0)
            y_all = torch.cat((y_all, y), dim=0)
        y_pred_all = y_pred_all.cpu().numpy()
        y_all = y_all.cpu().numpy()
    return metric_fn(y_pred_all, y_all)


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help="GPU ID. Set to None if you want to use CPU")
    parser.add_argument('--lr', '-lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--n_classes', '-c', type=int, default=7,
                        help="number of classes. 1 for regression, 2 for binary classification,"
                              ">=3 for multi-class classification")
    parser.add_argument('--metric', '-m', type=str, default='acc',
                        help="metric to evaluate the model. Supported metrics: [accuracy, rmse]")
    parser.add_argument('--dataset', '-d', type=str, default='covtype',
                        help="dataset to use. Supported datasets: [covtype, msd, higgs]")
    parser.add_argument('--n_parties', '-p', type=int, default=4,
                        help="number of parties. Should be >=2")
    args = parser.parse_args()

    # Note: torch.compile() in torch 2.0 significantly harms the accuracy with a few improvements in speed.
    train_dataset = VFLAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', 4, type='train')
    test_dataset = VFLAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', 4, type='test')

    # create the model
    if args.n_classes == 1:         # regression
        task = 'reg'
        loss_fn = nn.MSELoss()
        out_dim = 1
        if args.metric == 'acc':    # if metric is accuracy, change it to rmse
            args.metric = 'rmse'
            warnings.warn("Metric is changed to rmse for regression task")
    elif args.n_classes == 2:       # binary classification
        task = 'bin-cls'
        loss_fn = nn.BCELoss()
        out_dim = 1
    else:                           # multi-class classification
        task = 'multi-cls'
        loss_fn = nn.CrossEntropyLoss()
        out_dim = args.n_classes
    model = SplitMLP(train_dataset.local_input_channels, [[50]] * 4, [100, out_dim], out_activation=nn.Sigmoid())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    metric_fn = get_metric_from_str(args.metric)
    fit(model, optimizer, loss_fn, metric_fn, train_loader, epochs=args.epochs, gpu_id=args.gpu, n_classes=args.n_classes, test_loader=test_loader, task=task)
    # evaluate(model, train_loader, accuracy_score, gpu_id=0, n_classes=7)




