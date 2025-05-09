# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file



import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np

from . import utils, layers, layerspp, normalization

# Define aliases for custom layers
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

################
################
################
################
################
#PYTORCH VERSION
################
################
################
################
################

# class NCSNpp(nn.Module):
#     """NCSN++ model in PyTorch"""

#     def __init__(self, image_size, attn_resolutions, num_in_channels=1, num_out_channels=1,
#                  label_dim=10, use_cfg=True, ch_mult=(1, 2, 2), nf=32, num_res_blocks=2,
#                  dropout=0.0, resamp_with_conv=True, fir=False, fir_kernel=(1, 3, 3, 1),
#                  skip_rescale=True, resblock_type='biggan', progressive='none',
#                  progressive_input='none', embedding_type='positional', init_scale=0.0,
#                  combine_method='sum', fourier_scale=16, nonlinearity='swish'):
#         super().__init__()

#         self.image_size = image_size
#         self.attn_resolutions = attn_resolutions
#         self.num_in_channels = num_in_channels
#         self.num_out_channels = num_out_channels
#         self.label_dim = label_dim
#         self.use_cfg = use_cfg
#         self.ch_mult = ch_mult
#         self.nf = nf
#         self.num_res_blocks = num_res_blocks
#         self.dropout = dropout
#         self.resamp_with_conv = resamp_with_conv
#         self.fir = fir
#         self.fir_kernel = fir_kernel
#         self.skip_rescale = skip_rescale
#         self.resblock_type = resblock_type
#         self.progressive = progressive
#         self.progressive_input = progressive_input
#         self.embedding_type = embedding_type
#         self.init_scale = init_scale
#         self.combine_method = combine_method
#         self.fourier_scale = fourier_scale
#         self.act = get_act(nonlinearity)

#         self.num_resolutions = len(ch_mult)
#         self.all_resolutions = [image_size // (2 ** i) for i in range(self.num_resolutions)]

#         self.conditional = True  # noise-conditional
#         combiner = functools.partial(Combine, method=combine_method)

#         # Timestep embedding
#         if self.embedding_type == 'fourier':
#             self.fourier_proj = layerspp.GaussianFourierProjection(
#                 embedding_size=nf, scale=fourier_scale
#             )
#             embed_dim = 2 * nf
#         elif self.embedding_type == 'positional':
#             embed_dim = nf
#         else:
#             raise ValueError(f'Unknown embedding type {self.embedding_type}')

#         # Label embedding (for conditional models)
#         if self.conditional and self.label_dim > 0:
#             self.label_embed = nn.Embedding(self.label_dim + (1 if self.use_cfg else 0), embed_dim)

#         self.temb_dense_1 = nn.Linear(nf, nf * 4)
#         self.temb_dense_2 = nn.Linear(nf * 4, nf * 4)

#         # Define residual blocks
#         if self.resblock_type == 'ddpm':
#             ResnetBlock = functools.partial(ResnetBlockDDPM, act=self.act, dropout=self.dropout,
#                                             init_scale=self.init_scale, skip_rescale=self.skip_rescale,
#                                             temb_dim=nf * 4)
#         elif self.resblock_type == 'biggan':
#             ResnetBlock = functools.partial(ResnetBlockBigGAN, act=self.act, dropout=self.dropout,
#                                             fir=self.fir, fir_kernel=self.fir_kernel, init_scale=self.init_scale,
#                                             skip_rescale=self.skip_rescale, temb_dim=nf * 4)
#         else:
#             raise ValueError(f'Unknown resblock type {self.resblock_type}')

#         self.res_blocks = nn.ModuleList()
#         in_ch = nf
#         for i_level in range(self.num_resolutions):
#             for _ in range(self.num_res_blocks):
#                 out_ch = nf * self.ch_mult[i_level]
#                 self.res_blocks.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
#                 in_ch = out_ch

#     def forward(self, x, time_cond, y=None):
#         """Forward pass"""
#         temb = None
#         if self.embedding_type == 'fourier':
#             temb = self.fourier_proj(torch.log(time_cond))
#         elif self.embedding_type == 'positional':
#             temb = layers.get_timestep_embedding(time_cond, self.nf)

#         if self.conditional and self.label_dim is not None:
#             if y is not None:
#                 temb = temb + self.label_embed(y)
#             else:
#                 raise ValueError("Class label required for conditional model.")

#         temb = self.act(self.temb_dense_1(temb))
#         temb = self.temb_dense_2(temb)

#         # Pass through residual blocks
#         for block in self.res_blocks:
#             x = block(x, temb)

#         return x


############################################################
############ #Fixed NCSNpp class for Flax/JAX ############
############################################################

from . import utils, layers, layerspp, normalization
import flax.linen as nn
import functools
import jax.numpy as jnp
import numpy as np
import ml_collections


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class NCSNpp(nn.Module):
    """NCSN++ model in Flax"""

    image_size: int
    attn_resolutions: list
    num_in_channels: int = 1
    num_out_channels: int = 1
    label_dim: int = 10
    use_cfg: bool = True
    ch_mult: tuple = (1, 2, 2)
    nf: int = 32
    num_res_blocks: int = 2
    dropout: float = 0.0
    resamp_with_conv: bool = True
    fir: bool = False
    fir_kernel: list = (1, 3, 3, 1)
    skip_rescale: bool = True
    resblock_type: str = 'biggan'
    progressive: str = 'none'
    progressive_input: str = 'none'
    embedding_type: str = 'positional'
    init_scale: float = 0.0
    combine_method: str = 'sum'
    fourier_scale: int = 16
    nonlinearity: str = 'swish'

    def setup(self):
        self.act = get_act(self.nonlinearity)
        self.num_resolutions = len(self.ch_mult)
        self.all_resolutions = [self.image_size // (2 ** i) for i in range(self.num_resolutions)]

        self.conditional = True  # noise-conditional
        combiner = functools.partial(Combine, method=self.combine_method)

        # Timestep embedding
        if self.embedding_type == 'fourier':
            self.fourier_proj = layerspp.GaussianFourierProjection(
                embedding_size=self.nf, scale=self.fourier_scale
            )
            embed_dim = 2 * self.nf
        elif self.embedding_type == 'positional':
            embed_dim = self.nf
        else:
            raise ValueError(f'Unknown embedding type {self.embedding_type}')

        # Label embedding (conditional model)
        if self.conditional and self.label_dim > 0:
            self.label_embed = nn.Embed(self.label_dim + (1 if self.use_cfg else 0), embed_dim)

        self.temb_dense_1 = nn.Dense(self.nf * 4, kernel_init=default_initializer())
        self.temb_dense_2 = nn.Dense(self.nf * 4, kernel_init=default_initializer())

        # Define residual blocks
        if self.resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=self.act, dropout=self.dropout,
                                            init_scale=self.init_scale, skip_rescale=self.skip_rescale,
                                            temb_dim=self.nf * 4)
        elif self.resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=self.act, dropout=self.dropout,
                                            fir=self.fir, fir_kernel=self.fir_kernel, init_scale=self.init_scale,
                                            skip_rescale=self.skip_rescale, temb_dim=self.nf * 4)
        else:
            raise ValueError(f'Unknown resblock type {self.resblock_type}')

        self.res_blocks = []
        in_ch = self.nf
        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                out_ch = self.nf * self.ch_mult[i_level]
                self.res_blocks.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

    def __call__(self, x, time_cond, y):
        """Forward pass"""
        temb = None
        if self.embedding_type == 'fourier':
            temb = self.fourier_proj(jnp.log(time_cond))
        elif self.embedding_type == 'positional':
            temb = layers.get_timestep_embedding(time_cond, self.nf)

        if self.conditional and self.label_dim is not None:
            if y is not None:
                temb = temb + self.label_embed(y)
            else:
                raise ValueError("Class label required.")

        temb = self.temb_dense_1(temb)
        temb = self.temb_dense_2(self.act(temb))

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, temb)

        return x



#old code 
# from . import utils, layers, layerspp, normalization
# import flax.linen as nn
# import functools
# import jax.numpy as jnp
# import numpy as np
# import ml_collections



# ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
# ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
# Combine = layerspp.Combine
# conv3x3 = layerspp.conv3x3
# conv1x1 = layerspp.conv1x1
# get_act = layers.get_act
# get_normalization = normalization.get_normalization
# default_initializer = layers.default_init


# class NCSNpp(nn.Module):
#     """NCSN++ model"""

#     def __init__(self,
#                  image_size,
#                  attn_resolutions,
#                  num_in_channels=1,
#                  num_out_channels=1,
#                  label_dim=10,
#                  use_cfg=True,
#                  ch_mult=(1,2,2),
#                  nf=32,
#                  num_res_blocks=2,
#                  dropout=0.,
#                  resamp_with_conv=True,
#                  fir=False,
#                  fir_kernel=[1, 3, 3, 1],
#                  skip_rescale=True,
#                  resblock_type='biggan',
#                  progressive='none',
#                  progressive_input='none',
#                  embedding_type='positional',
#                  init_scale=0.,
#                  combine_method='sum',
#                  fourier_scale=16,
#                  nonlinearity='swish'):

#         super().__init__()

#         self.act = act = get_act(nonlinearity)

#         self.nf = nf
#         self.num_res_blocks = num_res_blocks
#         self.attn_resolutions = attn_resolutions
#         self.num_resolutions = num_resolutions = len(ch_mult)
#         self.all_resolutions = all_resolutions = [
#             image_size // (2 ** i) for i in range(num_resolutions)]

#         self.conditional = conditional = True  # noise-conditional
#         self.skip_rescale = skip_rescale
#         self.resblock_type = resblock_type
#         self.progressive = progressive
#         self.progressive_input = progressive_input
#         self.embedding_type = embedding_type
#         assert progressive in ['none', 'output_skip', 'residual']
#         assert progressive_input in ['none', 'input_skip', 'residual']
#         assert embedding_type in ['fourier', 'positional']
#         combiner = functools.partial(Combine, method=combine_method)

#         modules = []
#         # timestep/noise_level embedding; only for continuous training
#         if embedding_type == 'fourier':
#             modules.append(layerspp.GaussianFourierProjection(
#                 embedding_size=nf, scale=fourier_scale
#             ))
#             embed_dim = 2 * nf

#         elif embedding_type == 'positional':
#             embed_dim = nf

#         else:
#             raise ValueError(f'embedding type {embedding_type} unknown.')

#         self.label_dim = label_dim if label_dim > 0 else None

#         if conditional:
#             if label_dim > 0:
#                 modules.append(nn.Embed(label_dim if not use_cfg else label_dim + 1, embed_dim))
#             modules.append(nn.Dense(features=nf * 4))
#             modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
#             nn.init.zeros_(modules[-1].bias)
#             modules.append(nn.Linear(nf * 4, nf * 4))
#             modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
#             nn.init.zeros_(modules[-1].bias)

#         AttnBlock = functools.partial(layerspp.AttnBlockpp,
#                                       init_scale=init_scale,
#                                       skip_rescale=skip_rescale)

#         Upsample = functools.partial(layerspp.Upsample,
#                                      with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

#         if progressive == 'output_skip':
#             self.pyramid_upsample = Upsample(
#                 fir=fir, fir_kernel=fir_kernel, with_conv=False)
#         elif progressive == 'residual':
#             pyramid_upsample = functools.partial(layerspp.Upsample,
#                                                  fir=fir, fir_kernel=fir_kernel, with_conv=True)

#         Downsample = functools.partial(layerspp.Downsample,
#                                        with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

#         if progressive_input == 'input_skip':
#             self.pyramid_downsample = Downsample(
#                 fir=fir, fir_kernel=fir_kernel, with_conv=False)
#         elif progressive_input == 'residual':
#             pyramid_downsample = functools.partial(layerspp.Downsample,
#                                                    fir=fir, fir_kernel=fir_kernel, with_conv=True)

#         if resblock_type == 'ddpm':
#             ResnetBlock = functools.partial(ResnetBlockDDPM,
#                                             act=act,
#                                             dropout=dropout,
#                                             init_scale=init_scale,
#                                             skip_rescale=skip_rescale,
#                                             temb_dim=nf * 4)

#         elif resblock_type == 'biggan':
#             ResnetBlock = functools.partial(ResnetBlockBigGAN,
#                                             act=act,
#                                             dropout=dropout,
#                                             fir=fir,
#                                             fir_kernel=fir_kernel,
#                                             init_scale=init_scale,
#                                             skip_rescale=skip_rescale,
#                                             temb_dim=nf * 4)

#         else:
#             raise ValueError(f'resblock type {resblock_type} unrecognized.')

#         # Downsampling block

#         channels = num_in_channels
#         if progressive_input != 'none':
#             input_pyramid_ch = channels

#         modules.append(conv3x3(channels, nf))
#         hs_c = [nf]

#         in_ch = nf
#         for i_level in range(num_resolutions):
#             # Residual blocks for this resolution
#             for i_block in range(num_res_blocks):
#                 out_ch = nf * ch_mult[i_level]
#                 modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
#                 in_ch = out_ch

#                 if all_resolutions[i_level] in attn_resolutions:
#                     modules.append(AttnBlock(channels=in_ch))
#                 hs_c.append(in_ch)

#             if i_level != num_resolutions - 1:
#                 if resblock_type == 'ddpm':
#                     modules.append(Downsample(in_ch=in_ch))
#                 else:
#                     modules.append(ResnetBlock(down=True, in_ch=in_ch))

#                 if progressive_input == 'input_skip':
#                     modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
#                     if combine_method == 'cat':
#                         in_ch *= 2

#                 elif progressive_input == 'residual':
#                     modules.append(pyramid_downsample(
#                         in_ch=input_pyramid_ch, out_ch=in_ch))
#                     input_pyramid_ch = in_ch

#                 hs_c.append(in_ch)

#         in_ch = hs_c[-1]
#         modules.append(ResnetBlock(in_ch=in_ch))
#         modules.append(AttnBlock(channels=in_ch))
#         modules.append(ResnetBlock(in_ch=in_ch))

#         pyramid_ch = 0
#         # Upsampling block
#         for i_level in reversed(range(num_resolutions)):
#             for i_block in range(num_res_blocks + 1):
#                 out_ch = nf * ch_mult[i_level]
#                 modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
#                                            out_ch=out_ch))
#                 in_ch = out_ch

#             if all_resolutions[i_level] in attn_resolutions:
#                 modules.append(AttnBlock(channels=in_ch))

#             if progressive != 'none':
#                 if i_level == num_resolutions - 1:
#                     if progressive == 'output_skip':
#                         modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
#                                                     num_channels=in_ch, eps=1e-6))
#                         modules.append(
#                             conv3x3(in_ch, channels, init_scale=init_scale))
#                         pyramid_ch = channels
#                     elif progressive == 'residual':
#                         modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
#                                                     num_channels=in_ch, eps=1e-6))
#                         modules.append(conv3x3(in_ch, in_ch, bias=True))
#                         pyramid_ch = in_ch
#                     else:
#                         raise ValueError(f'{progressive} is not a valid name.')
#                 else:
#                     if progressive == 'output_skip':
#                         modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
#                                                     num_channels=in_ch, eps=1e-6))
#                         modules.append(
#                             conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
#                         pyramid_ch = channels
#                     elif progressive == 'residual':
#                         modules.append(pyramid_upsample(
#                             in_ch=pyramid_ch, out_ch=in_ch))
#                         pyramid_ch = in_ch
#                     else:
#                         raise ValueError(f'{progressive} is not a valid name')

#             if i_level != 0:
#                 if resblock_type == 'ddpm':
#                     modules.append(Upsample(in_ch=in_ch))
#                 else:
#                     modules.append(ResnetBlock(in_ch=in_ch, up=True))

#         assert not hs_c

#         if progressive != 'output_skip':
#             modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
#                                         num_channels=in_ch, eps=1e-6))
#             modules.append(
#                 conv3x3(in_ch, num_out_channels, init_scale=init_scale))

#         self.all_modules = nn.ModuleList(modules)

#     def forward(self, x, time_cond, y):
#         # timestep/noise_level embedding; only for continuous training
#         modules = self.all_modules
#         m_idx = 0
#         if self.embedding_type == 'fourier':
#             # Gaussian Fourier features embeddings.
#             temb = modules[m_idx](torch.log(time_cond))
#             m_idx += 1

#         elif self.embedding_type == 'positional':
#             # Sinusoidal positional embeddings.
#             temb = layers.get_timestep_embedding(time_cond, self.nf)

#         else:
#             raise ValueError(f'embedding type {self.embedding_type} unknown.')

#         if self.conditional:
#             if self.label_dim is not None:
#                 if y is not None:
#                     temb = temb + modules[m_idx](y)
#                 else:
#                     raise ValueError('Need to give a class label.')
#                 m_idx += 1

#             temb = modules[m_idx](temb)
#             m_idx += 1
#             temb = modules[m_idx](self.act(temb))
#             m_idx += 1
#         else:
#             temb = None

#         # Downsampling block
#         input_pyramid = None
#         if self.progressive_input != 'none':
#             input_pyramid = x

#         hs = [modules[m_idx](x)]
#         m_idx += 1
#         for i_level in range(self.num_resolutions):
#             # Residual blocks for this resolution
#             for i_block in range(self.num_res_blocks):
#                 h = modules[m_idx](hs[-1], temb)
#                 m_idx += 1
#                 if h.shape[-1] in self.attn_resolutions:
#                     h = modules[m_idx](h)
#                     m_idx += 1

#                 hs.append(h)

#             if i_level != self.num_resolutions - 1:
#                 if self.resblock_type == 'ddpm':
#                     h = modules[m_idx](hs[-1])
#                     m_idx += 1
#                 else:
#                     h = modules[m_idx](hs[-1], temb)
#                     m_idx += 1

#                 if self.progressive_input == 'input_skip':
#                     input_pyramid = self.pyramid_downsample(input_pyramid)
#                     h = modules[m_idx](input_pyramid, h)
#                     m_idx += 1

#                 elif self.progressive_input == 'residual':
#                     input_pyramid = modules[m_idx](input_pyramid)
#                     m_idx += 1
#                     if self.skip_rescale:
#                         input_pyramid = (input_pyramid + h) / np.sqrt(2.)
#                     else:
#                         input_pyramid = input_pyramid + h
#                     h = input_pyramid

#                 hs.append(h)

#         h = hs[-1]
#         h = modules[m_idx](h, temb)
#         m_idx += 1
#         h = modules[m_idx](h)
#         m_idx += 1
#         h = modules[m_idx](h, temb)
#         m_idx += 1

#         pyramid = None

#         # Upsampling block
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks + 1):
#                 h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
#                 m_idx += 1

#             if h.shape[-1] in self.attn_resolutions:
#                 h = modules[m_idx](h)
#                 m_idx += 1

#             if self.progressive != 'none':
#                 if i_level == self.num_resolutions - 1:
#                     if self.progressive == 'output_skip':
#                         pyramid = self.act(modules[m_idx](h))
#                         m_idx += 1
#                         pyramid = modules[m_idx](pyramid)
#                         m_idx += 1
#                     elif self.progressive == 'residual':
#                         pyramid = self.act(modules[m_idx](h))
#                         m_idx += 1
#                         pyramid = modules[m_idx](pyramid)
#                         m_idx += 1
#                     else:
#                         raise ValueError(
#                             f'{self.progressive} is not a valid name.')
#                 else:
#                     if self.progressive == 'output_skip':
#                         pyramid = self.pyramid_upsample(pyramid)
#                         pyramid_h = self.act(modules[m_idx](h))
#                         m_idx += 1
#                         pyramid_h = modules[m_idx](pyramid_h)
#                         m_idx += 1
#                         pyramid = pyramid + pyramid_h
#                     elif self.progressive == 'residual':
#                         pyramid = modules[m_idx](pyramid)
#                         m_idx += 1
#                         if self.skip_rescale:
#                             pyramid = (pyramid + h) / np.sqrt(2.)
#                         else:
#                             pyramid = pyramid + h
#                         h = pyramid
#                     else:
#                         raise ValueError(
#                             f'{self.progressive} is not a valid name')

#             if i_level != 0:
#                 if self.resblock_type == 'ddpm':
#                     h = modules[m_idx](h)
#                     m_idx += 1
#                 else:
#                     h = modules[m_idx](h, temb)
#                     m_idx += 1

#         assert not hs

#         if self.progressive == 'output_skip':
#             h = pyramid
#         else:
#             h = self.act(modules[m_idx](h))
#             m_idx += 1
#             h = modules[m_idx](h)
#             m_idx += 1

#         assert m_idx == len(modules)
#         return h


        