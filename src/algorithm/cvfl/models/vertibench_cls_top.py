import torch.nn as nn


__all__ = ['VertiBench_Cls_top', 'vertibench_cls_top']


class VertiBench_Cls_top(nn.Module):
    def __init__(self, num_classes, num_clients, activation = None):
        super().__init__()
        self.activation = activation
        print("🔵 VertiBench_Cls_top num_classes: ", num_classes)
        self.linear = nn.Sequential(
            # nn.Linear(n_features, 100),
            # nn.ReLU(),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            nn.Linear(100 * num_clients, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def vertibench_cls_top(pretrained=False, **kwargs):
    model = VertiBench_Cls_top(**kwargs)
    if pretrained:
        print("No pretrained model available for VertiBench_Cls_top")
    return model
