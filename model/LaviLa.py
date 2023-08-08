

from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn
from model.openai_clip import load as load_openai_clip
from model.openai_model import QuickGELU, Transformer


# util functions to convert CLIP-style model keys to TimeSformer-style
def remap_keys(clip_state_dict, transformer_layers=12):
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "class_embedding": "cls_token",
        "positional_embedding": "pos_embed",
        "conv1.weight": "patch_embed.proj.weight",
        "ln_pre.weight": "ln_pre.weight",
        "ln_pre.bias": "ln_pre.bias",
        "ln_post.weight": "norm.weight",
        "ln_post.bias": "norm.bias",
    }
    for layer in range(transformer_layers):
        key_mapping[f"transformer.resblocks.{layer}.attn.in_proj_weight"] = f"blocks.{layer}.attn.qkv.weight"
        key_mapping[f"transformer.resblocks.{layer}.attn.in_proj_bias"] = f"blocks.{layer}.attn.qkv.bias"
        key_mapping[f"transformer.resblocks.{layer}.attn.out_proj.weight"] = f"blocks.{layer}.attn.proj.weight"
        key_mapping[f"transformer.resblocks.{layer}.attn.out_proj.bias"] = f"blocks.{layer}.attn.proj.bias"
        key_mapping[f"transformer.resblocks.{layer}.ln_1.weight"] = f"blocks.{layer}.norm1.weight"
        key_mapping[f"transformer.resblocks.{layer}.ln_1.bias"] = f"blocks.{layer}.norm1.bias"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_fc.weight"] = f"blocks.{layer}.mlp.fc1.weight"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_fc.bias"] = f"blocks.{layer}.mlp.fc1.bias"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_proj.weight"] = f"blocks.{layer}.mlp.fc2.weight"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_proj.bias"] = f"blocks.{layer}.mlp.fc2.bias"
        key_mapping[f"transformer.resblocks.{layer}.ln_2.weight"] = f"blocks.{layer}.norm2.weight"
        key_mapping[f"transformer.resblocks.{layer}.ln_2.bias"] = f"blocks.{layer}.norm2.bias"

    for key in clip_state_dict:
        if key == 'proj':
            continue  # due to possible dim mismatch, we load this later
        if key == "class_embedding":
            clip_state_dict[key] = clip_state_dict[key].unsqueeze(0).unsqueeze(0)
        if key == "positional_embedding":
            clip_state_dict[key] = clip_state_dict[key].unsqueeze(0)
        remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict

def CLIP_OPENAI_TIMESFORMER_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, use_adapter=False,**kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
        use_adapter=use_adapter,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


def CLIP_OPENAI_TIMESFORMER_LARGE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256,use_adapter=False, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
        use_adapter = use_adapter,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class VideoPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8, ln_pre=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        # ln_pre is inserted to be compatible with CLIP-style model
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

    def forward(self, x):
        B, F, C, H, W = x.shape
        # assert F <= self.num_frames
        x = x.reshape(-1, C, H, W)
        x = self.proj(x)
        return x


class VarAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, einops_from, einops_to, einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale

        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class Adapter(nn.Module):
    def __init__(self, input_dim, down_dim, act='relu') -> None:
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(input_dim)
        # self.post_layer_norm = nn.LayerNorm(input_dim)

        self.linear = nn.Sequential(nn.Linear(input_dim,down_dim),
                                    nn.ReLU(),
                                    nn.Linear(down_dim, input_dim))
        self.scaling = nn.Parameter(torch.zeros(1))


    def forward(self, x, attn_output):
        residual = attn_output
        hidden = self.pre_layer_norm(x+attn_output)
        hidden = self.linear(hidden)*self.scaling
        output = hidden + residual
        return output


class SpaceTimeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, n_layer=0, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_init='zeros',
                 attention_style='frozen-in-time', is_tanh_gating=False, use_adapter=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.timeattn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            initialize=time_init)

        if is_tanh_gating:
            self.alpha_timeattn = nn.Parameter(torch.zeros([]))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.attention_style = attention_style
        if n_layer ==12 and use_adapter:
                print(n_layer)
                self.use_adapter = True
                self.spatial_adapter = Adapter(1024,64)
                self.temporal_adapter = Adapter(1024,64)
        else:
            self.use_adapter = False
        

        # self._init_adapter_modules()

    # def _init_adapter_modules(self):
    #     self.output_adapters = AdapterLayer("output_adapter", self.config)
    #     self.adapters = nn.ModuleDict(dict())
    #     self.adapter_fusion_layer = nn.ModuleDict(dict())
        
    def forward(self, x, einops_from_space, einops_to_space, einops_from_time, einops_to_time,
                time_n, space_f, use_checkpoint=False):
        if use_checkpoint:
            time_output = checkpoint.checkpoint(
                self.timeattn, self.norm3(x), einops_from_time, einops_to_time, {"n": time_n},
                use_reentrant=False
            )
        else:
            time_output = self.timeattn(self.norm3(x), einops_from_time, einops_to_time, {"n": time_n})
        if hasattr(self, "alpha_timeattn"):
            time_output = torch.tanh(self.alpha_timeattn) * time_output
        if self.use_adapter:
            if use_checkpoint:
                time_output = checkpoint.checkpoint(
                    self.temporal_adapter, x, time_output,
                    use_reentrant=False
                )
            else:
                time_output = self.temporal_adapter(x, time_output)
        time_residual = x + time_output

        if use_checkpoint:
            space_output = checkpoint.checkpoint(
                self.attn, self.norm1(time_residual), einops_from_space, einops_to_space, {"f": space_f},
                use_reentrant=False
            )
        else:
            space_output = self.attn(self.norm1(time_residual), einops_from_space,
                                     einops_to_space, {"f": space_f})
        if self.use_adapter:
            if use_checkpoint:
                space_output = checkpoint.checkpoint(
                    self.spatial_adapter, x, space_output,
                    use_reentrant=False
                )
            else:
                space_output = self.spatial_adapter(x, space_output)

        if self.attention_style == 'frozen-in-time':
            space_residual = x + self.drop_path(space_output)
        else:
            raise NotImplementedError

        x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))

        return x


class SpaceTimeTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650
    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].
    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 num_frames=8, time_init='rand', attention_style='frozen-in-time', ln_pre=False,
                 act_layer=nn.GELU, is_tanh_gating=False,use_adapter=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input
            time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
                        as ViT.
            attention_style: (str) how to attend to space and time.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        print("######USING ATTENTION STYLE: ", attention_style)
        if hybrid_backbone is not None:
            raise NotImplementedError('hybrid backbone not implemented')
        else:
            self.patch_embed = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames, ln_pre=ln_pre)
        num_patches = self.patch_embed.num_patches
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        if ln_pre:
            self.ln_pre = nn.LayerNorm(embed_dim)
        else:
            self.ln_pre = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, n_layer=i,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
                attention_style=attention_style, act_layer=act_layer, is_tanh_gating=is_tanh_gating, use_adapter=use_adapter)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

        # einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze_spatial_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                pass
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def freeze_temporal_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                p.requires_grad = False
                freeze_list.append(n)
            else:
                pass
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def forward_features(self, x, use_checkpoint=False, cls_at_last=True):
        # print(x.shape)
        b, curr_frames, channels, _, _ = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)

        BF = x.shape[0]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, curr_frames, 1)
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)

        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames

        for blk in self.blocks:
            x = blk(x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
                    self.einops_to_time,
                    time_n=n, space_f=f, use_checkpoint=use_checkpoint)

        if cls_at_last:
            x_cls = self.norm(x)[:, 0]
            x_cls = self.pre_logits(x_cls)

        return x_cls,self.norm(x)

    def forward(self, x, use_checkpoint=False):
        # Note:  B C T H W => B T C H W
        # The default input order is different from the one in Frozen-in-Time
        # x = x.permute(0, 2, 1, 3, 4).contiguous()
        x_cls,x = self.forward_features(x, use_checkpoint=use_checkpoint)
        x_cls = self.head(x_cls)
        return x_cls, x




class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        x_cls, x = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x_cls, x
        x_cls = x_cls @ self.image_projection
        return x_cls, x

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_cls = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x_cls, x

    def forward(self, image, text, use_checkpoint=False, norm_embed=True, return_feature_map=False):
        image_embed, image_fmap = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed, text_fmap = self.encode_text(text, use_checkpoint=use_checkpoint)
        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        if not return_feature_map:
            return {'image_embed': image_embed,
                    'text_embed': text_embed,
                    'logit_scale': self.logit_scale.exp()}
        else:
            return {'image_embed': image_embed,
                    'text_embed': text_embed,
                    'image_feature_map':image_fmap,
                    'text_feature_map':text_fmap,
                    'logit_scale': self.logit_scale.exp()}