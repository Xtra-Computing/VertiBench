import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['RESNET_top', 'resnet_top']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class RESNET_top(nn.Module):

    def __init__(self, num_classes = 10, num_clients=4):
        super(RESNET_top, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * num_clients, num_classes), # 跟 SplitNN 里面的对齐
        )
        print("RESNET_top", self.classifier)
        #self.classifier = nn.Sequential(
        #    nn.Linear(num_classes * num_clients, 100),
        #    nn.Linear(100, 100),
        #    nn.Linear(100, num_classes),
        #)

    def forward(self, x):
        # print("RESNET_top forward", x.shape)
        pooled_view = self.classifier(x)
        return pooled_view


def resnet_top(pretrained=False, **kwargs):
    r"""MVCNN model architecture from the
    `"Multi-view Convolutional..." <hhttp://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RESNET_top(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model
