import torch
from torch import nn
from einops import rearrange

class DiNOBackbone(nn.Module):
    def __init__(self, ver='dino_vitb16'):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dino:main", ver, pretrained=True)
    
    def forward(self, images, return_cls_token=False):
        features = self.backbone.get_intermediate_layers(images)[0]  # [B, L, C]
        hw = int(features.shape[1] ** 0.5)
        cls_token = features[:, 0]
        features = rearrange(features[:, 1:], 'b (h w) c -> b c h w', h=hw)
        if return_cls_token:
            return features, cls_token
        else:
            return features