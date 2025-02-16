# hubconf.py
import torch
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.upsamplers import get_upsampler
from torch.nn import Module

dependencies = ['torch', 'torchvision', 'PIL', 'segment_anything', 'featup']  # List any dependencies here


class UpsampledBackbone(Module):

    def __init__(self, model_name, use_norm):
        super().__init__()
        model, patch_size, self.dim = get_featurizer(model_name, "token", num_classes=1000)
        if use_norm:
            self.model = torch.nn.Sequential(model, ChannelNorm(self.dim))
        else:
            self.model = model
        self.upsampler = get_upsampler("jbu_stack", self.dim)

    def forward(self, image):
        return self.upsampler(self.model(image), image)


def _download_to_torch_cache(url):
    import os
    import requests
    from tqdm import tqdm

    torch_cache_home = torch.hub.get_dir()
    os.makedirs(torch_cache_home, exist_ok=True)
    filename = os.path.join(torch_cache_home, url.split('/')[-1])
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        with open(filename, "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)
    return filename


def _load_backbone(pretrained, use_norm, model_name):
    """
    The function that will be called by Torch Hub users to instantiate your model.
    Args:
        pretrained (bool): If True, returns a model pre-loaded with weights.
    Returns:
        An instance of your model.
    """
    model = UpsampledBackbone(model_name, use_norm)
    if pretrained:
        # Define how you load your pretrained weights here
        # For example:
        if use_norm:
            exp_dir = ""
        else:
            exp_dir = "no_norm/"

        my_model_list = ["sam"]
        if model_name in my_model_list:
            norm = "yes" if use_norm else "no"
            checkpoint_url = f"https://raw.githubusercontent.com/huzeyann/FeatUp/refs/heads/main/ckpts/{model_name}_{norm}_norm.ckpt"
            filename = _download_to_torch_cache(checkpoint_url)
            state_dict = torch.load(filename)
        else:
            checkpoint_url = f"https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/{exp_dir}{model_name}_jbu_stack_cocostuff.ckpt"
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if "scale_net" not in k and "downsampler" not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


def vit(pretrained=True, use_norm=False):
    return _load_backbone(pretrained, use_norm, "vit")


def dino16(pretrained=True, use_norm=False):
    return _load_backbone(pretrained, use_norm, "dino16")

def dino(pretrained=True, use_norm=False):
    return _load_backbone(pretrained, use_norm, "dino16")

def sam(pretrained=True, use_norm=False):
    return _load_backbone(pretrained, use_norm, "sam")

def clip(pretrained=True, use_norm=False):
    return _load_backbone(pretrained, use_norm, "clip")


def dinov2(pretrained=True, use_norm=False):
    return _load_backbone(pretrained, use_norm, "dinov2")


def resnet50(pretrained=True, use_norm=False):
    return _load_backbone(pretrained, use_norm, "resnet50")

def maskclip(pretrained=True, use_norm=False):
    assert not use_norm, "MaskCLIP only supports unnormed model"
    return _load_backbone(pretrained, use_norm, "maskclip")
