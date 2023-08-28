import torch.nn as nn

__all__ = ['VertiBench_Cls_bottom', 'vertibench_cls_bottom']


class VertiBench_Cls_bottom(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        print("🔵 VertiBench_Cls_bottom all_features: ", n_features)
        self.linear = nn.Sequential(
            nn.Linear(n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            # nn.Linear(100 * num_clients, 200),
            # nn.ReLU(),
            # nn.Linear(200, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.linear(x)
        return x


def vertibench_cls_bottom(pretrained=False, **kwargs):
    model = VertiBench_Cls_bottom(**kwargs)
    if pretrained:
        print("No pretrained model available for VertiBench_Cls_bottom")
    return model
