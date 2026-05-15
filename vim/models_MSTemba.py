# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple_getC import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def compute_c_state_diversity_loss_simple(c_states_list):
    """
    Compute diversity loss between C states using cosine similarity.
    This version is more compatible with Mamba's backward pass.
    
    Args:
        c_states_list: List of C states from different dilation groups
    
    Returns:
        diversity_loss: Cosine similarity diversity loss between C states
    """
    if len(c_states_list) < 2:
        return torch.tensor(0.0, device=c_states_list[0].device)
    
    total_loss = 0.0
    num_pairs = 0
    
    # Compute pairwise diversity losses
    for i in range(len(c_states_list)):
        for j in range(i + 1, len(c_states_list)):
            c_i = c_states_list[i]
            c_j = c_states_list[j]
            
            # Ensure both C states have the same shape for comparison
            if c_i.shape != c_j.shape:
                # Take the minimum length and truncate
                min_len = min(c_i.shape[-1], c_j.shape[-1])
                c_i = c_i[..., :min_len]
                c_j = c_j[..., :min_len]
            
            # Flatten C states to compute cosine similarity
            c_i_flat = c_i.reshape(c_i.shape[0], -1)  # (B, d_state * seq_len)
            c_j_flat = c_j.reshape(c_j.shape[0], -1)  # (B, d_state * seq_len)
            
            # Normalize for cosine similarity
            c_i_norm = F.normalize(c_i_flat, p=2, dim=1, eps=1e-8)
            c_j_norm = F.normalize(c_j_flat, p=2, dim=1, eps=1e-8)
            
            # Compute cosine similarity (1 - cosine_similarity for diversity)
            cosine_sim = torch.sum(c_i_norm * c_j_norm, dim=1)
            # diversity_loss = cosine_sim.mean()  # Penalize high similarity (closer to 1)
            diversity_loss = (cosine_sim ** 2).mean()  # Penalize any correlation (push towards orthogonal)
            
            total_loss += diversity_loss
            num_pairs += 1
    
    return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=c_states_list[0].device)
    

__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states, C = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual, C

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 temp_dim=256, 
                 patch_size=16, 
                 stride=16,
                 depth=24, 
                 in_feat_dim=1024,
                 embed_dim=192, 
                 d_state=16,
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = True, 
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        # self.proj = nn.Conv1d(in_channels=in_feat_dim, out_channels=embed_dim, kernel_size=1)

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, temp_dim + self.num_tokens, self.embed_dim))
            # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        # self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        # x = self.patch_embed(x)
        # x = self.proj(x)
        # x = x.permute(0, 2, 1)
        B, M, _ = x.shape
        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]
        if self.if_abs_pos_embed:
            # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
            #     x = x + self.pos_embed
            # else:
            #     pos_embed = interpolate_pos_embed_online(
            #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
            #             )

            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:

            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)




        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        C = None  # Initialize C state
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])
                hidden_states, residual, C = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f, C_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b, C_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
                
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2, C
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :], C
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :], C
                else:
                    return hidden_states[:, token_position, :], C

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :], C
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1), C
        elif self.final_pool_type == 'max':
            return hidden_states, C
        elif self.final_pool_type == 'all':
            return hidden_states, C
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
        if return_features:
            return x

        # x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x

class LinearProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (B, T, C)
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class MultiScaleAttentionFuser(nn.Module):
    '''Multi-Scale Attention Fuser with Dynamic Routing'''
    def __init__(self, embed_dim, num_scales=3, nhead=8, dropout=0.25):
        super().__init__()
        # Keep heads valid for any embed_dim.
        while nhead > 1 and embed_dim % nhead != 0:
            nhead -= 1

        self.pre_proj = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.GELU(),
        )
        
        # PRE-NORM Components
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Token-wise dynamic routing over multi-scale features.
        self.router = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_scales)
        )
        self.post_norm = nn.LayerNorm(embed_dim)
        self.router_tau = nn.Parameter(torch.ones(1))
        nn.init.constant_(self.router[1].weight, 0)
        nn.init.constant_(self.router[1].bias, 0)

    def forward(self, multi_scale_features):
        # multi_scale_features: list of tensors 
        raw_concat = torch.cat(multi_scale_features, dim=-1)
        x = self.pre_proj(raw_concat)

        # Attention with PRE-NORM
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)

        # FFN with PRE-NORM
        x = x + self.ffn(self.norm2(x))

        # Dynamic Routing
        stacked_scales = torch.stack(multi_scale_features, dim=2)  # (B, T, S, C)
        tau = torch.clamp(self.router_tau, min=0.1)                 # floor at 0.1 prevents near-zero temp collapse
        routing_weights = torch.softmax(self.router(x) / tau, dim=-1)
        routed = (routing_weights.unsqueeze(-1) * stacked_scales).sum(dim=2)

        return self.post_norm(x + routed), routing_weights

class MultiScaleAttentionNoFFNNoRouter(nn.Module):
    '''Ablation: Self-Attention Only, No FFN, No Routing'''
    def __init__(self, embed_dim, num_scales=3, nhead=8, dropout=0.25):
        super().__init__()

        while nhead > 1 and embed_dim % nhead != 0:
            nhead -= 1

        self.pre_proj = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.GELU())

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, multi_scale_features):

        raw_concat = torch.cat(multi_scale_features, dim=-1)   # (B, T, 3C)
        x = self.pre_proj(raw_concat)                          # (B, T, C)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)

        return self.post_norm(x), None

class MultiScaleAttentionNoRouter(nn.Module):
    '''Ablation: Self-Attention + FFN, No Routing'''
    def __init__(self, embed_dim, num_scales=3, nhead=8, dropout=0.25):
        super().__init__()

        while nhead > 1 and embed_dim % nhead != 0:
            nhead -= 1

        self.pre_proj = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.GELU())

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, multi_scale_features):
        raw_concat = torch.cat(multi_scale_features, dim=-1)   # (B, T, 3C)
        x = self.pre_proj(raw_concat)                          # (B, T, C)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))

        return self.post_norm(x), None
    
class MultiScaleAttentionRoutingNoFFN(nn.Module):
    '''Ablation: Self-Attention + Routing, No FFN'''
    def __init__(self, embed_dim, num_scales=3, nhead=8, dropout=0.25):
        super().__init__()
        while nhead > 1 and embed_dim % nhead != 0:
            nhead -= 1

        self.pre_proj = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.GELU())

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.router = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_scales))

        self.post_norm = nn.LayerNorm(embed_dim)
        self.router_tau = nn.Parameter(torch.ones(1))

        nn.init.constant_(self.router[1].weight, 0)
        nn.init.constant_(self.router[1].bias, 0)

    def forward(self, multi_scale_features):
        raw_concat = torch.cat(multi_scale_features, dim=-1)   # (B, T, 3C)
        x = self.pre_proj(raw_concat)                          # (B, T, C)

        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)

        stacked_scales = torch.stack(multi_scale_features, dim=2)   # (B, T, 3, C)
        tau = torch.clamp(self.router_tau, min=0.1)
        routing_weights = torch.softmax(self.router(x) / tau, dim=-1)
        routed = (routing_weights.unsqueeze(-1) * stacked_scales).sum(dim=2)

        return self.post_norm(x + routed), routing_weights

class MultiScaleAttentionX3Fuser(nn.Module):
    """AttentionX3: Per-Scale Self-Attention + FFN + Projection, then sum & post-norm."""
    def __init__(self, dims, embed_dim, dropout=0.25):
        super().__init__()
        self.dims = dims
        self.embed_dim = embed_dim

        self.attns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.projs = nn.ModuleList()

        nheads = [max(1, d // 64) for d in dims]
        nheads = [nh - (d % nh != 0) * (nh % (d // nh)) for nh, d in zip(nheads, dims)]

        safe_nheads = []
        for d, nh in zip(dims, nheads):
            while nh > 1 and d % nh != 0:
                nh -= 1
            safe_nheads.append(nh)

        for d, nh in zip(dims, safe_nheads):
            self.attns.append(
                nn.MultiheadAttention(d, nh, dropout=dropout, batch_first=True)
            )
            self.norms1.append(nn.LayerNorm(d))
            self.norms2.append(nn.LayerNorm(d))
            self.ffns.append(nn.Sequential(
                nn.Linear(d, d * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d * 4, d),
                nn.Dropout(dropout),
            ))
            self.projs.append(nn.Linear(d, embed_dim))

        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, scale_features):
        projected = []
        for i, x in enumerate(scale_features):
            # Pre-norm self-attention
            x_norm = self.norms1[i](x)
            attn_out, _ = self.attns[i](x_norm, x_norm, x_norm, need_weights=False)
            x = x + attn_out
            # Pre-norm FFN
            x = x + self.ffns[i](self.norms2[i](x))
            projected.append(self.projs[i](x))

        fused = sum(projected)  # (B, T, E)
        return self.post_norm(fused), None


class SequenceBatchNorm1d(nn.Module):
    """BatchNorm1d wrapper that handles (B, T, C) inputs."""
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x: (B, T, C) → transpose to (B, C, T) for BN → back
        return self.bn(x.transpose(1, 2)).transpose(1, 2)



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class MSTemba(nn.Module):
    def __init__(self,
                 in_feat_dim=768, #CLIP; 1024 for I3D
                 num_classes=157,
                 embed_dims=[256, 384, 576],
                 depths=[1, 1, 1],
                 d_state=16,
                 fuser='sum',
                 head_drop=0.0,
                 flip_img_sequences_ratio=0.0,
                 **kwargs):
        super().__init__()
        self.fuser = fuser
        # Add linear layers for each block
        self.block_heads = nn.ModuleList([
            nn.Linear(embed_dims[i], num_classes) for i in range(3)
        ])
        
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        self.proj = nn.Linear(in_feat_dim, embed_dims[0])

        self.scale_proj1 = nn.Linear(embed_dims[0], embed_dims[2])
        self.scale_proj2 = nn.Linear(embed_dims[1], embed_dims[2])
        self.scale_proj3 = nn.Linear(embed_dims[2], embed_dims[2])

        if self.fuser == 'weighted':
            self.fuser_weights = nn.Parameter(torch.ones(3))

        elif self.fuser == 'attention':
            self.fuser_attention_module = MultiScaleAttentionFuser(embed_dims[-1], num_scales=3, nhead=8, dropout=0.25)
        elif self.fuser == 'attention_x3':
            self.fuser_attx3_module = MultiScaleAttentionX3Fuser(
                dims=embed_dims,          # [256, 384, 576]
                embed_dim=embed_dims[-1], # 576
                dropout=0.25,
            )

        elif self.fuser == 'token-attention':
            d = embed_dims[2]
            self.fuser_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)  # learned query
            self.fuser_k = nn.Linear(d, d, bias=False)
            self.fuser_v = nn.Linear(d, d, bias=False)
            self.fuser_scale = d ** -0.5

        elif self.fuser == 'cross-token-attention':
            self.fuser_attention = nn.MultiheadAttention(
                embed_dim=embed_dims[2], num_heads=4, batch_first=True)
            self.fuser_q = nn.Parameter(torch.randn(1, 1, embed_dims[2]) * 0.02)
        
        elif self.fuser == 'concat-proj':
            self.fuser_concat_proj = nn.Sequential(
                nn.Linear(embed_dims[-1] * 3, embed_dims[-1]),
                nn.GELU())
            
        elif self.fuser == 'attn-noffn-norouter':
            self.fuser_attention_noffn_norouter = MultiScaleAttentionNoFFNNoRouter(
                embed_dims[-1], num_scales=3, nhead=8, dropout=0.25)
            
        elif self.fuser == 'attn-ffn-norouter':
            self.fuser_attention_no_router = MultiScaleAttentionNoRouter(
                embed_dims[-1], num_scales=3, nhead=8, dropout=0.25)
            
        elif self.fuser == 'attn-router-noffn':
            self.fuser_attention_router_noffn = MultiScaleAttentionRoutingNoFFN(
                embed_dims[-1], num_scales=3, nhead=8, dropout=0.25)

        # Hierarchical blocks
        self.blocks = nn.ModuleList()

        # First block - single SSM
        self.blocks.append(LinearProjection(embed_dims[0], embed_dims[0]))
        self.blocks.append(self._create_mamba_block(embed_dims[0], d_state, depths[0], flip_img_sequences_ratio=flip_img_sequences_ratio, **kwargs))

        # Second block - two SSMs for odd/even tokens
        self.blocks.append(LinearProjection(embed_dims[0], embed_dims[1]))
        self.blocks.append(nn.ModuleList([
            self._create_mamba_block(embed_dims[1], d_state, depths[1], flip_img_sequences_ratio=flip_img_sequences_ratio, **kwargs),
            self._create_mamba_block(embed_dims[1], d_state, depths[1], flip_img_sequences_ratio=flip_img_sequences_ratio, **kwargs)
        ]))

        # Third block - three SSMs
        self.blocks.append(LinearProjection(embed_dims[1], embed_dims[2]))
        self.blocks.append(nn.ModuleList([
            self._create_mamba_block(embed_dims[2], d_state, depths[2], flip_img_sequences_ratio=flip_img_sequences_ratio, **kwargs),
            self._create_mamba_block(embed_dims[2], d_state, depths[2], flip_img_sequences_ratio=flip_img_sequences_ratio, **kwargs),
            self._create_mamba_block(embed_dims[2], d_state, depths[2], flip_img_sequences_ratio=flip_img_sequences_ratio, **kwargs)
        ]))

        self.interaction_block = self._create_mamba_block(embed_dims[-1], d_state, depths[-1], flip_img_sequences_ratio=flip_img_sequences_ratio, **kwargs)

        # Final norm and classifier
        self.norm = nn.LayerNorm(embed_dims[-1])
        if self.fuser =='attention_x3':
            head_drop = head_drop
        else:
            head_drop = 0.2
        self.head_dropout = nn.Dropout(p=head_drop)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _create_mamba_block(self, embed_dim, d_state, depth, flip_img_sequences_ratio=0.0, **kwargs):
        return VisionMamba(
            embed_dim=embed_dim,
            depth=depth,
            d_state=d_state,
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True, 
            final_pool_type='all', 
            if_abs_pos_embed=False, 
            if_rope=False, 
            if_rope_residual=False, 
            bimamba_type="v2", 
            if_cls_token=False, 
            if_divide_out=True, 
            use_middle_cls_token=True,
            flip_img_sequences_ratio=flip_img_sequences_ratio,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        x = x.permute(0, 2, 1)
        x = self.proj(x)                                         # projection first
        if self.training:
            x = x + torch.randn_like(x) * 0.03                  # noise in projected space
            mask = (torch.rand(x.shape[0], x.shape[1], 1,
                            device=x.device) > 0.10).float()
            x = x * mask
        concat_x = []
        block_outputs = []  # Store raw block outputs
        all_c_states = []  # Store C states from all blocks for diversity loss
        
        for i, block in enumerate(self.blocks):
            if i == 0 or i == 1:  # First block
                if isinstance(block, LinearProjection):
                    x = block(x)
                    block_c_states = []  # No C states for linear projection
                else:  # VisionMamba 
                    B, T, C = x.shape
                    x, C_state = block.forward_features(x)
                    concat_x.append(x)
                    block_outputs.append(x)  # Store raw output
                    block_c_states = [C_state] if C_state is not None else []
            
            elif i == 2 or i == 3:  # Second block - split into odd/even tokens
                if isinstance(block, LinearProjection):
                    x = block(x)    
                    B, T, C = x.shape
                    # Split into odd and even tokens
                    x_even = x[:, ::2, :]  # Even tokens
                    x_odd = x[:, 1::2, :]  # Odd tokens
                    block_c_states = []  # No C states for linear projection
                else:  # VisionMamba with two separate SSMs
                    # Process even and odd tokens through separate SSMs
                    x_even_out, C_even = block[0].forward_features(x_even)
                    x_odd_out, C_odd = block[1].forward_features(x_odd)
                    
                    # Interleave the outputs back together
                    x = torch.zeros(B, T, C, device=x_even_out.device)
                    x[:, ::2, :] = x_even_out
                    x[:, 1::2, :] = x_odd_out
                    
                    concat_x.append(x)
                    block_outputs.append(x)  # Store raw output
                    block_c_states = [C_even, C_odd] if C_even is not None and C_odd is not None else []

            elif i == 4 or i == 5:  # Third block - split into three groups
                if isinstance(block, LinearProjection):
                    x = block(x)    
                    B, T, C = x.shape
                    # Ensure T is divisible by 3
                    pad_size = (3 - (T % 3)) % 3
                    if pad_size > 0:
                        x = F.pad(x, (0, 0, 0, pad_size))
                        T = T + pad_size
                    
                    # Split into three groups
                    x_group1 = x[:, ::3, :]  # First group (0, 3, 6, ...)
                    x_group2 = x[:, 1::3, :]  # Second group (1, 4, 7, ...)
                    x_group3 = x[:, 2::3, :]  # Third group (2, 5, 8, ...)
                    block_c_states = []  # No C states for linear projection
                else:  # VisionMamba with three separate SSMs
                    # Process each group through its own SSM
                    x_out1, C_group1 = block[0].forward_features(x_group1)
                    x_out2, C_group2 = block[1].forward_features(x_group2)
                    x_out3, C_group3 = block[2].forward_features(x_group3)
                    
                    # Interleave the outputs back together
                    x = torch.zeros(B, T, C, device=x_out1.device)
                    x[:, ::3, :] = x_out1
                    x[:, 1::3, :] = x_out2
                    x[:, 2::3, :] = x_out3
                    
                    # Remove padding if it was added
                    if pad_size > 0:
                        x = x[:, :-pad_size, :]
                    
                    concat_x.append(x)
                    block_outputs.append(x)  # Store raw output
                    block_c_states = [C_group1, C_group2, C_group3] if all(c is not None for c in [C_group1, C_group2, C_group3]) else []
            
            all_c_states.append(block_c_states)

        return concat_x, block_outputs, all_c_states

    def forward(self, x):
        concat_x, block_outputs, all_c_states = self.forward_features(x)

        # Fusion and Mamba interaction
        x1, x2, x3 = concat_x

        x1 = self.norm(self.scale_proj1(x1))
        x2 = self.norm(self.scale_proj2(x2))
        x3 = self.norm(self.scale_proj3(x3))

        # Process each block output
        block_predictions = []
        for i, block_out in enumerate(block_outputs):
            # Apply block-specific head
            if self.fuser == 'attention_x3':
                block_pred = self.block_heads[i](self.head_dropout(block_out))
            else:
                block_pred = self.block_heads[i](block_out)
            block_predictions.append(block_pred)

        # standard fuser sums the three block outputs togheter (original MS-Temba paper)
        if self.fuser == 'sum':
            x = x1 + x2 + x3
            self._last_fusion_weights = torch.tensor([1/3, 1/3, 1/3], device=x.device)  # For logging equal weights

        # weighted fuser learns to weight the three block outputs, which can be useful to understand the importance of each block and for potentially improving performance by allowing the model to focus more on certain blocks. The weights are normalized with softmax to ensure they sum to 1.
        elif self.fuser == 'weighted':
            fusion_weights = torch.softmax(self.fuser_weights, dim=0)
            self._last_fusion_weights = fusion_weights.detach()
            x = fusion_weights[0] * x1 + fusion_weights[1] * x2 + fusion_weights[2] * x3
        
        elif self.fuser == 'token-attention':
            stacked = torch.stack([x1, x2, x3], dim=2)        # (B, T, 3, C)
            K = self.fuser_k(stacked)                         # (B, T, 3, C)
            V = self.fuser_v(stacked)                         # (B, T, 3, C)
            Q = self.fuser_q.expand(stacked.size(0), stacked.size(1), -1)   # (B, T, C)
            # Score each branch by dot product with the query
            scores = torch.einsum('btc,btkc->btk', Q, K) * self.fuser_scale  # (B, T, 3)
            weights = torch.softmax(scores, dim=-1)
            self._last_fusion_weights = weights.detach()
            x = (V * weights.unsqueeze(-1)).sum(dim=2)        # (B, T, C)
        
        elif self.fuser == 'cross-token-attention':
            B, T, C = x1.shape
            stacked = torch.stack([x1, x2, x3], dim=2).reshape(B * T, 3, C)  # (B*T, 3, C)
            q = self.fuser_q.expand(B * T, 1, C)
            fused, attn = self.fuser_attention(q, stacked, stacked,           # (B*T, 1, C)
                                            need_weights=True, average_attn_weights=True)
            # attn: (B*T, 1, 3) → reshape to (B, T, 3)
            attn = attn.reshape(B, T, 3)
            self._last_fusion_weights = attn.detach()

            x = fused.reshape(B, T, C)

        elif self.fuser == 'attention':
            x, routing_weights = self.fuser_attention_module([x1, x2, x3])
            self._last_fusion_weights = routing_weights      # no .detach() — gradients must flow

        elif self.fuser == 'concat-proj':

            x = torch.cat([x1, x2, x3], dim=-1)   # (B, T, 3C)

            x = self.fuser_concat_proj(x)         # (B, T, C)

            self._last_fusion_weights = None
        elif self.fuser == 'attn-noffn-norouter':
            x, routing_weights = self.fuser_attention_noffn_norouter([x1, x2, x3])
            self._last_fusion_weights = None

        elif self.fuser == 'attn-ffn-norouter':
            x, routing_weights = self.fuser_attention_no_router([x1, x2, x3])
            self._last_fusion_weights = None

        elif self.fuser == 'routing-only':
            x, routing_weights = self.fuser_routing_only([x1, x2, x3])
            self._last_fusion_weights = routing_weights

        elif self.fuser == 'attn-router-noffn':
            x, routing_weights = self.fuser_attention_router_noffn([x1, x2, x3])
            self._last_fusion_weights = routing_weights

        # attention_x3: per-scale attention in native dims, project to E inside module
        elif self.fuser == 'attention_x3':
            # x1_raw, x2_raw, x3_raw are in native dims [256, 384, 576]
            # (before scale_proj which would project everything to 576)
            x1_raw, x2_raw, x3_raw = concat_x  # [256], [384], [576]
            x, routing_weights = self.fuser_attx3_module([x1_raw, x2_raw, x3_raw])
            self._last_fusion_weights = None

        else:
            raise ValueError(f"Unknown fuser mode: {self.fuser}")


        #if self.fuser != 'attention':
        x, _ = self.interaction_block(x)
        
        x = self.head_dropout(x)
        x = self.head(x)
        
        # Compute C state diversity loss with gradients flowing to Mamba components
        # Initialize diversity loss as zero scalar; gradients flow from block_diversity_loss
        diversity_loss = torch.tensor(0.0, device=x.device)
        for block_idx, block_c_states in enumerate(all_c_states):
            if len(block_c_states) >= 2:  # Only compute if we have multiple C states
                # Filter out None values but keep gradients
                valid_c_states = [c_state for c_state in block_c_states if c_state is not None]
                
                if len(valid_c_states) >= 2:
                    # Compute diversity loss for all valid C states in this block
                    # Allow gradients to flow back to Mamba components
                    block_diversity_loss = compute_c_state_diversity_loss_simple(valid_c_states)
                    diversity_loss = diversity_loss + block_diversity_loss
        
        # Return fusion weights if available (for logging)
        fusion_weights = getattr(self, '_last_fusion_weights', None)
        return x, block_predictions, diversity_loss, fusion_weights


@register_model
def mstemba(pretrained=False, **kwargs):
    # Define the model
    model = MSTemba(
        embed_dims=[256, 384, 576],
        depths=[1, 1, 1],
        d_state=16,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=False,
        **kwargs
    )

    model.default_cfg = _cfg()
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    
    return model