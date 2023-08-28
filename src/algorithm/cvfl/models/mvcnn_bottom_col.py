import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['MVCNN_bottom', 'mvcnn_bottom']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class MVCNN_bottom(nn.Module):

    def __init__(self, num_classes=1000):
        super(MVCNN_bottom, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=3),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        
        view_pool = []
        
        for v in x:
            v = self.features(v)
            v = v.view(v.size(0), 256 * 6 * 6)
            
            view_pool.append(v)
        
        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])
        
        pooled_view = self.classifier(pooled_view)
        return pooled_view


def mvcnn_bottom(pretrained=False, **kwargs):
    r"""MVCNN model architecture from the
    `"Multi-view Convolutional..." <hhttp://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MVCNN_bottom(**kwargs)
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
