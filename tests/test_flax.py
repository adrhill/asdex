"""Integration tests for Flax NNX models."""
#
#
# Verifies that asdex can trace through Flax NNX modules
# and detect Jacobian sparsity without errors.
# These models exercise a wide range of JAX primitives
# (conv, batch norm, layer norm, attention, pooling, etc.)
# and serve as a smoke test for missing primitive handlers.
#
# The ResNet and ViT architectures are adapted from Bonsai,
# a collection of minimal JAX model implementations:
# https://github.com/jax-ml/bonsai
#
# Copyright 2025 The JAX Authors.
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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.linen.pooling import max_pool

from asdex import jacobian_sparsity

# ResNet
# Adapted from https://github.com/jax-ml/bonsai/blob/main/bonsai/models/resnet/modeling.py


# ResNet bottleneck block: 1x1 -> 3x3 -> 1x1 with skip connection.
class _Bottleneck(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        downsample: nnx.Module | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv0 = nnx.Conv(
            in_channels,
            mid_channels,
            kernel_size=(1, 1),
            strides=1,
            padding=0,
            use_bias=False,
            rngs=rngs,
        )
        self.bn0 = nnx.BatchNorm(mid_channels, use_running_average=True, rngs=rngs)

        self.conv1 = nnx.Conv(
            mid_channels,
            mid_channels,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(mid_channels, use_running_average=True, rngs=rngs)

        self.conv2 = nnx.Conv(
            mid_channels,
            mid_channels * 4,
            kernel_size=(1, 1),
            strides=1,
            padding=0,
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(mid_channels * 4, use_running_average=True, rngs=rngs)

        self.downsample = downsample

    def __call__(self, x):
        identity = x
        x = nnx.relu(self.bn0(self.conv0(x)))
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return nnx.relu(x + identity)


# 1x1 conv + batch norm for residual dimension matching.
class _Downsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            strides=stride,
            padding=0,
            use_bias=False,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(x))


# Stack of bottleneck blocks with optional downsampling on the first.
class _BlockGroup(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_blocks: int,
        stride: int,
        *,
        rngs: nnx.Rngs,
    ):
        out_channels = mid_channels * 4
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = _Downsample(in_channels, out_channels, stride, rngs=rngs)

        blocks = [_Bottleneck(in_channels, mid_channels, stride, downsample, rngs=rngs)]
        blocks.extend(
            _Bottleneck(
                out_channels, mid_channels, stride=1, downsample=None, rngs=rngs
            )
            for _ in range(1, num_blocks)
        )
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# Small ResNet: stem -> 4 block groups -> global avg pool -> linear.
class _ResNet(nnx.Module):
    def __init__(
        self,
        block_layers: list[int],
        num_classes: int,
        base: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.stem_conv = nnx.Conv(
            3,
            base,
            kernel_size=(7, 7),
            strides=2,
            padding=3,
            use_bias=False,
            rngs=rngs,
        )
        self.stem_bn = nnx.BatchNorm(base, use_running_average=True, rngs=rngs)
        self.pool = partial(
            max_pool,
            window_shape=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
        )

        self.layer0 = _BlockGroup(base, base, block_layers[0], stride=1, rngs=rngs)
        self.layer1 = _BlockGroup(
            base * 4, base * 2, block_layers[1], stride=2, rngs=rngs
        )
        self.layer2 = _BlockGroup(
            base * 8, base * 4, block_layers[2], stride=2, rngs=rngs
        )
        self.layer3 = _BlockGroup(
            base * 16, base * 8, block_layers[3], stride=2, rngs=rngs
        )

        self.fc = nnx.Linear(base * 32, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.stem_bn(self.stem_conv(x)))
        x = self.pool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(axis=(0, 1))  # global average pool (H, W, C) -> (C,)
        return self.fc(x)


# ViT
# Adapted from https://github.com/jax-ml/bonsai/blob/main/bonsai/models/vit/modeling.py


# Patch embedding + class token + positional embedding.
class _Embeddings(nnx.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        num_patches = (image_size // patch_size) ** 2
        self.projection = nnx.Conv(
            3,
            hidden_dim,
            kernel_size=(patch_size, patch_size),
            strides=patch_size,
            rngs=rngs,
        )
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, hidden_dim)))
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_patches + 1, hidden_dim)),
        )

    def __call__(self, x):
        x = self.projection(x[None])  # (1, H', W', D)
        b, h, w, d = x.shape
        x = x.reshape(b, h * w, d)  # (1, num_patches, D)
        cls = jnp.tile(self.cls_token[...], (b, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + self.pos_embed[...]
        return x[0]  # (num_patches + 1, D)


# Pre-norm transformer block: LN -> MHSA -> residual -> LN -> MLP -> residual.
class _TransformerBlock(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            decode=False,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear1 = nnx.Linear(hidden_dim, mlp_dim, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_dim, hidden_dim, rngs=rngs)

    def __call__(self, x):
        h = self.ln1(x)
        x = self.attention(h) + x
        h = self.ln2(x)
        h = jax.nn.gelu(self.linear1(h))
        return self.linear2(h) + x


# Small Vision Transformer: patch embed -> N transformer blocks -> classify.
class _ViT(nnx.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed = _Embeddings(image_size, patch_size, hidden_dim, rngs=rngs)
        self.layers = nnx.List(
            [
                _TransformerBlock(hidden_dim, num_heads, mlp_dim, rngs=rngs)
                for _ in range(num_layers)
            ]
        )
        self.ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.classifier = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.classifier(x[0])  # classify from CLS token


# Fixtures


def _make_resnet_fn():
    """Create a tiny ResNet and return a pure function.

    Uses [1, 1, 1, 1] blocks with 8 base channels on 8x8 images.
    This exercises the full architecture
    (stem, bottleneck, skip connections, pooling)
    while keeping the element count small enough for fast tracing.
    """
    rngs = nnx.Rngs(0)
    model = _ResNet([1, 1, 1, 1], num_classes=4, base=8, rngs=rngs)
    graphdef, state = nnx.split(model)

    def apply(x):
        m = nnx.merge(graphdef, state)
        return m(x)

    return apply


def _make_vit_fn():
    """Create a tiny ViT and return a pure function.

    Uses 1 layer, 32-dim, 2 heads on 8x8 images with 4x4 patches (4 patches).
    This exercises patch embedding, attention, GELU MLP, layer norm,
    and CLS token classification
    while keeping the sequence length and hidden dim small.
    """
    rngs = nnx.Rngs(0)
    model = _ViT(
        image_size=8,
        patch_size=4,
        hidden_dim=32,
        num_heads=2,
        mlp_dim=64,
        num_layers=1,
        num_classes=4,
        rngs=rngs,
    )
    graphdef, state = nnx.split(model)

    def apply(x):
        m = nnx.merge(graphdef, state)
        return m(x)

    return apply


# Tests


@pytest.mark.slow
@pytest.mark.jacobian
@pytest.mark.bug
def test_resnet_sparsity_detection():
    """Jacobian sparsity detection traces through a ResNet without errors.

    TODO(reduce_window_max): ResNet uses max_pool,
    which lowers to the reduce_window_max primitive.
    """
    resnet_fn = _make_resnet_fn()
    input_shape = (8, 8, 3)

    with pytest.raises(NotImplementedError, match="reduce_window_max"):
        jacobian_sparsity(resnet_fn, np.zeros(input_shape))


@pytest.mark.slow
@pytest.mark.jacobian
def test_vit_sparsity_detection():
    """Jacobian sparsity detection traces through a ViT without errors."""
    vit_fn = _make_vit_fn()
    input_shape = (8, 8, 3)

    sparsity = jacobian_sparsity(vit_fn, np.zeros(input_shape))

    n_in = 8 * 8 * 3
    assert sparsity.n == n_in
    assert sparsity.m == 4
    assert sparsity.nnz > 0
