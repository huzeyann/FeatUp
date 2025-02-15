from einops import rearrange
import torch
from torch import nn

import os
class SAM(torch.nn.Module):
    def __init__(self, model_size='vit_b', **kwargs):
        super().__init__(**kwargs)
        try:
            from segment_anything import sam_model_registry
            from segment_anything.modeling.sam import Sam
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install segment_anything package by running: 
            `pip install segment_anything`
            """
            raise ImportError(s)

        ckpt_dict = {
            'vit_h': "sam_vit_h_4b8939.pth",
            'vit_l': "sam_vit_l_0b3195.pth",
            'vit_b': "sam_vit_b_01ec64.pth",
        }
        
        state_dict = torch.hub.load_state_dict_from_url(f"https://dl.fbaipublicfiles.com/segment_anything/{ckpt_dict[model_size]}")
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        checkpoint_path = os.path.join(model_dir, ckpt_dict[model_size])

        sam: Sam = sam_model_registry[model_size](checkpoint=checkpoint_path)

        self.image_encoder = sam.image_encoder
        
        # save the original position embedding for resampling later
        self.original_pos_embed = nn.Parameter(self.image_encoder.pos_embed.clone())
    
    def resample_pos_embed(self, x):
        patch_size = 16
        assert x.shape[2] % patch_size == 0 and x.shape[3] % patch_size == 0, \
                "Input size must be divisible by the patch size (16)"
        
        pos_embed = self.original_pos_embed.clone()
        H, W = x.shape[2] // patch_size, x.shape[3] // patch_size
        pos_embed = rearrange(pos_embed, 'b h w c -> b c h w')
        pos_embed = torch.nn.functional.interpolate(pos_embed, size=(H, W), mode="bilinear")
        pos_embed = rearrange(pos_embed, 'b c h w -> b h w c')
        self.image_encoder.pos_embed = nn.Parameter(pos_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)        
        # resample the position embedding, to match the variable input H and W
        self.resample_pos_embed(x)
        
        out = self.image_encoder(x) # (B, C, H, W)
        return out
    