# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging

from torch import nn
import numpy as np
import spconv
import torch

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args
from second.pytorch.models.sparse_utils import configure_sparse_state


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from shared.parameters import bind_named as _bind_named


class RpnInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    use_norm: object
    num_class: object
    layer_nums: object
    layer_strides: object
    num_filters: object
    upsample_strides: object
    num_upsample_filters: object
    num_input_features: object
    num_anchor_per_loc: object
    encode_background_as_zeros: object
    use_direction_classifier: object
    use_groupnorm: object
    num_groups: object
    use_bev: object
    box_code_size: object
    use_rc_net: object
    name: object


class SparserpnInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    output_shape: object
    num_input_features: object
    num_filters_down1: object
    num_filters_down2: object
    use_norm: object
    num_class: object
    layer_nums: object
    layer_strides: object
    num_filters: object
    upsample_strides: object
    num_upsample_filters: object
    num_anchor_per_loc: object
    encode_background_as_zeros: object
    use_direction_classifier: object
    use_groupnorm: object
    num_groups: object
    use_bev: object
    box_code_size: object
    use_rc_net: object
    name: object


LOGGER = logging.getLogger(__name__)


def _prediction_dict(module, features, include_rc=True):
    """Build channel-last prediction tensors shared by the RPN variants."""
    ret_dict = {
        "box_preds": module.conv_box(features).permute(0, 2, 3, 1).contiguous(),
        "cls_preds": module.conv_cls(features).permute(0, 2, 3, 1).contiguous(),
    }
    if module.use_direction_classifier:
        ret_dict["dir_cls_preds"] = (
            module.conv_dir_cls(features).permute(0, 2, 3, 1).contiguous()
        )
    if include_rc and module.use_rc_net:
        ret_dict["rc_preds"] = (
            module.conv_rc(features).permute(0, 2, 3, 1).contiguous()
        )
    return ret_dict


_RPN_PARAMETER_NAMES = (
    "use_norm",
    "num_class",
    "layer_nums",
    "layer_strides",
    "num_filters",
    "upsample_strides",
    "num_upsample_filters",
    "num_input_features",
    "num_anchor_per_loc",
    "encode_background_as_zeros",
    "use_direction_classifier",
    "use_groupnorm",
    "num_groups",
    "use_bev",
    "box_code_size",
    "use_rc_net",
    "name",
)


_RPN_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    RpnInitParameters,
    _RPN_PARAMETER_NAMES,
    {
        "use_norm": True,
        "num_class": 2,
        "layer_nums": None,
        "layer_strides": None,
        "num_filters": None,
        "upsample_strides": None,
        "num_upsample_filters": None,
        "num_input_features": 128,
        "num_anchor_per_loc": 2,
        "encode_background_as_zeros": True,
        "use_direction_classifier": True,
        "use_groupnorm": False,
        "num_groups": 32,
        "use_bev": False,
        "box_code_size": 7,
        "use_rc_net": False,
        "name": "rpn",
    },
    frozenset(()),
    17,
    None,
    None,
)
_RPNV2_INIT_PARAMETERS_SCHEMA = _RPN_INIT_PARAMETERS_SCHEMA
_SPARSERPN_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    SparserpnInitParameters,
    (
        "output_shape",
        "num_input_features",
        "num_filters_down1",
        "num_filters_down2",
    ) + _RPN_PARAMETER_NAMES[:7] + _RPN_PARAMETER_NAMES[8:],
    {
        "num_input_features": 128,
        "num_filters_down1": None,
        "num_filters_down2": None,
        "use_norm": True,
        "num_class": 2,
        "layer_nums": None,
        "layer_strides": None,
        "num_filters": None,
        "upsample_strides": None,
        "num_upsample_filters": None,
        "num_anchor_per_loc": 2,
        "encode_background_as_zeros": True,
        "use_direction_classifier": True,
        "use_groupnorm": False,
        "num_groups": 32,
        "use_bev": False,
        "box_code_size": 7,
        "use_rc_net": False,
        "name": "sparse_rpn",
    },
    frozenset(()),
    20,
    None,
    None,
)


def _initialized_rpn_state(module, parameters):
    state = parameters._asdict()
    state["self"] = module
    defaults = {
        "layer_nums": [3, 5, 5],
        "layer_strides": [2, 2, 2],
        "num_filters": [128, 128, 256],
        "upsample_strides": [1, 2, 4],
        "num_upsample_filters": [256, 256, 256],
        "num_filters_down1": [64],
        "num_filters_down2": [64, 64],
    }
    for name, default in defaults.items():
        if name in state and state[name] is None:
            state[name] = default
    return state


def _validate_rpn_layers(
    layer_nums, layer_strides, num_filters, upsample_strides, num_upsample_filters
):
    layer_count = len(layer_nums)
    if not (
        layer_count == 3
        and len(layer_strides) == layer_count
        and len(num_filters) == layer_count
        and len(upsample_strides) == layer_count
        and len(num_upsample_filters) == layer_count
    ):
        raise AssertionError
    factors = [
        np.prod(layer_strides[: index + 1]) // upsample_strides[index]
        for index in range(layer_count)
    ]
    if any(
        int(np.prod(layer_strides[: index + 1])) % upsample_strides[index]
        for index in range(layer_count)
    ) or not all(factor == factors[0] for factor in factors):
        raise AssertionError


def _rpn_layer_factories(use_norm, use_groupnorm, num_groups):
    if not use_norm:
        return Empty, change_default_args(bias=True)(nn.Conv2d), change_default_args(
            bias=True
        )(nn.ConvTranspose2d)
    if use_groupnorm:
        batch_norm2d = change_default_args(num_groups=num_groups, eps=1e-3)(
            GroupNorm
        )
    else:
        batch_norm2d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
    return (
        batch_norm2d,
        change_default_args(bias=False)(nn.Conv2d),
        change_default_args(bias=False)(nn.ConvTranspose2d),
    )


def _make_rpn_block(*call_args, **call_kwargs):
    (
        conv2d,
        batch_norm2d,
        in_filters,
        out_filters,
        stride,
        repeats,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('conv2d', 'batch_norm2d', 'in_filters', 'out_filters', 'stride', 'repeats'),
    )
    block = Sequential(
        nn.ZeroPad2d(1),
        conv2d(in_filters, out_filters, 3, stride=stride),
        batch_norm2d(out_filters),
        nn.ReLU(),
    )
    for _ in range(repeats):
        block.add(conv2d(out_filters, out_filters, 3, padding=1))
        block.add(batch_norm2d(out_filters))
        block.add(nn.ReLU())
    return block


def _make_rpn_deconv(conv_transpose2d, batch_norm2d, in_filters, out_filters, stride):
    return Sequential(
        conv_transpose2d(in_filters, out_filters, stride, stride=stride),
        batch_norm2d(out_filters),
        nn.ReLU(),
    )


def _initialize_rpn_blocks(model, factories, layer_config, num_input_features, use_bev):
    batch_norm2d, conv2d, conv_transpose2d = factories
    layer_nums, layer_strides, num_filters = layer_config[:3]
    upsample_strides, num_upsample_filters = layer_config[3:]
    block2_input_filters = num_filters[0]
    if use_bev:
        model.bev_extractor = Sequential(
            conv2d(6, 32, 3, padding=1), batch_norm2d(32), nn.ReLU(),
            conv2d(32, 64, 3, padding=1), batch_norm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        block2_input_filters += 64
    block_inputs = (num_input_features, block2_input_filters, num_filters[1])
    for index in range(3):
        block = _make_rpn_block(
            conv2d, batch_norm2d, block_inputs[index], num_filters[index],
            layer_strides[index], layer_nums[index],
        )
        deconv = _make_rpn_deconv(
            conv_transpose2d, batch_norm2d, num_filters[index],
            num_upsample_filters[index], upsample_strides[index],
        )
        setattr(model, "block{}".format(index + 1), block)
        setattr(model, "deconv{}".format(index + 1), deconv)


class RPN(nn.Module):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(_RPN_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs)
        (use_norm, num_class, layer_nums, layer_strides, num_filters,
         upsample_strides, num_upsample_filters, num_input_features,
         num_anchor_per_loc, encode_background_as_zeros, use_direction_classifier,
         use_groupnorm, num_groups, use_bev, box_code_size, use_rc_net, name) = bound_parameters
        layer_nums = [3, 5, 5] if layer_nums is None else layer_nums
        layer_strides = [2, 2, 2] if layer_strides is None else layer_strides
        num_filters = [128, 128, 256] if num_filters is None else num_filters
        upsample_strides = [1, 2, 4] if upsample_strides is None else upsample_strides
        num_upsample_filters = [256, 256, 256] if num_upsample_filters is None else num_upsample_filters
        super(RPN, self).__init__()
        self.num_anchor_per_location = num_anchor_per_loc
        self.use_direction_classifier = use_direction_classifier
        self.use_bev = use_bev
        self.use_rc_net = use_rc_net
        _validate_rpn_layers(
            layer_nums,
            layer_strides,
            num_filters,
            upsample_strides,
            num_upsample_filters,
        )
        factories = _rpn_layer_factories(
            use_norm, use_groupnorm, num_groups
        )
        layer_config = (
            layer_nums, layer_strides, num_filters,
            upsample_strides, num_upsample_filters,
        )
        _initialize_rpn_blocks(
            self, factories, layer_config, num_input_features, use_bev
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1
        )
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1
            )

        if self.use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1
            )

    def forward(self, x, bev=None):

        x = self.block1(x)
        up1 = self.deconv1(x)
        if self.use_bev:
            bev[:, -1] = torch.clamp(torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        return _prediction_dict(self, x)


def _rpnv2_initialize_part_1(_state):
    _state["self"].num_anchor_per_location = _state["num_anchor_per_loc"]
    _state["self"].use_direction_classifier = _state["use_direction_classifier"]
    _state["self"].use_bev = _state["use_bev"]
    _state["self"].use_rc_net = _state["use_rc_net"]
    if not len(_state["layer_strides"]) == len(_state["layer_nums"]):
        raise AssertionError
    if not len(_state["num_filters"]) == len(_state["layer_nums"]):
        raise AssertionError
    if not len(_state["upsample_strides"]) == len(_state["layer_nums"]):
        raise AssertionError
    if not len(_state["num_upsample_filters"]) == len(_state["layer_nums"]):
        raise AssertionError
    if _state["use_norm"]:
        if _state["use_groupnorm"]:
            _state["batch_norm2d"] = change_default_args(
                num_groups=_state["num_groups"], eps=0.001
            )(GroupNorm)
        else:
            _state["batch_norm2d"] = change_default_args(eps=0.001, momentum=0.01)(
                nn.BatchNorm2d
            )
        _state["conv2d"] = change_default_args(bias=False)(nn.Conv2d)
        _state["conv_transpose2d"] = change_default_args(bias=False)(nn.ConvTranspose2d)
    else:
        _state["batch_norm2d"] = Empty
        _state["conv2d"] = change_default_args(bias=True)(nn.Conv2d)
        _state["conv_transpose2d"] = change_default_args(bias=True)(nn.ConvTranspose2d)
    _state["in_filters"] = [_state["num_input_features"], *_state["num_filters"][:-1]]
    _state["blocks"] = []
    _state["deblocks"] = []


def _rpnv2_initialize_part_2_chunk_1(_state):
    for _state["i"], _state["layer_num"] in enumerate(_state["layer_nums"]):
        _state["block"] = Sequential(
            nn.ZeroPad2d(1),
            _state["conv2d"](
                _state["in_filters"][_state["i"]],
                _state["num_filters"][_state["i"]],
                3,
                stride=_state["layer_strides"][_state["i"]],
            ),
            _state["batch_norm2d"](_state["num_filters"][_state["i"]]),
            nn.ReLU(),
        )
        for _state["_"] in range(_state["layer_num"]):
            _state["block"].add(
                _state["conv2d"](
                    _state["num_filters"][_state["i"]],
                    _state["num_filters"][_state["i"]],
                    3,
                    padding=1,
                )
            )
            _state["block"].add(
                _state["batch_norm2d"](_state["num_filters"][_state["i"]])
            )
            _state["block"].add(nn.ReLU())
        _state["blocks"].append(_state["block"])
        _state["deblock"] = Sequential(
            _state["conv_transpose2d"](
                _state["num_filters"][_state["i"]],
                _state["num_upsample_filters"][_state["i"]],
                _state["upsample_strides"][_state["i"]],
                stride=_state["upsample_strides"][_state["i"]],
            ),
            _state["batch_norm2d"](_state["num_upsample_filters"][_state["i"]]),
            nn.ReLU(),
        )
        _state["deblocks"].append(_state["deblock"])
    _state["self"].blocks = nn.ModuleList(_state["blocks"])
    _state["self"].deblocks = nn.ModuleList(_state["deblocks"])
    if _state["encode_background_as_zeros"]:
        _state["num_cls"] = _state["num_anchor_per_loc"] * _state["num_class"]
    else:
        _state["num_cls"] = _state["num_anchor_per_loc"] * (_state["num_class"] + 1)
    LOGGER.info("%s %s", "num_class is :", _state["num_class"])


def _rpnv2_initialize_part_2_chunk_2(_state):
    _state["self"].conv_cls = nn.Conv2d(
        sum(_state["num_upsample_filters"]), _state["num_cls"], 1
    )
    _state["self"].conv_box = nn.Conv2d(
        sum(_state["num_upsample_filters"]),
        _state["num_anchor_per_loc"] * _state["box_code_size"],
        1,
    )


def _rpnv2_initialize_part_2(_state):
    _rpnv2_initialize_part_2_chunk_1(_state)
    _rpnv2_initialize_part_2_chunk_2(_state)


def _rpnv2_initialize_part_3(_state):
    if _state["use_direction_classifier"]:
        _state["self"].conv_dir_cls = nn.Conv2d(
            sum(_state["num_upsample_filters"]), _state["num_anchor_per_loc"] * 2, 1
        )
    if _state["self"].use_rc_net:
        _state["self"].conv_rc = nn.Conv2d(
            sum(_state["num_upsample_filters"]),
            _state["num_anchor_per_loc"] * _state["box_code_size"],
            1,
        )


class RPNV2(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage."""

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _RPNV2_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(RPNV2, self).__init__()
        _state = _initialized_rpn_state(self, parameters)
        _rpnv2_initialize_part_1(_state)
        _rpnv2_initialize_part_2(_state)
        _rpnv2_initialize_part_3(_state)

    def forward(self, x, bev=None):
        ups = []
        for i, _ in enumerate(self.blocks):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        return _prediction_dict(self, x)


class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(2)


def _sparserpn_initialize_part_1(_state):
    _state["self"].num_anchor_per_location = _state["num_anchor_per_loc"]
    _state["self"].use_direction_classifier = _state["use_direction_classifier"]
    configure_sparse_state(
        _state, change_default_args, nn, spconv, Empty, LOGGER
    )
    _state["self"].block1 = spconv.SparseSequential(
        _state["sp_conv3d"](
            _state["num_input_features"],
            _state["num_filters"][0],
            3,
            stride=[2, _state["layer_strides"][0], _state["layer_strides"][0]],
            padding=[0, 1, 1],
        ),
        _state["batch_norm1d"](_state["num_filters"][0]),
        nn.ReLU(),
    )


def _sparserpn_initialize_part_2_chunk_1(_state):
    for _state["_"] in range(_state["layer_nums"][0]):
        _state["self"].block1.add(
            _state["sub_m_conv3d"](
                _state["num_filters"][0],
                _state["num_filters"][0],
                3,
                padding=1,
                indice_key="subm0",
            )
        )
        _state["self"].block1.add(_state["batch_norm1d"](_state["num_filters"][0]))
        _state["self"].block1.add(nn.ReLU())
    _state["self"].deconv1 = spconv.SparseSequential(
        _state["sp_conv3d"](
            _state["num_filters"][0],
            _state["num_filters"][0],
            (3, 1, 1),
            stride=(2, 1, 1),
        ),
        _state["batch_norm1d"](_state["num_filters"][0]),
        nn.ReLU(),
        _state["sp_conv3d"](
            _state["num_filters"][0],
            _state["num_upsample_filters"][0],
            (2, 1, 1),
            stride=1,
        ),
        _state["batch_norm1d"](_state["num_upsample_filters"][0]),
        nn.ReLU(),
        spconv.ToDense(),
        Squeeze(),
    )
    _state["self"].block2 = spconv.SparseSequential(
        _state["sp_conv3d"](
            _state["num_filters"][0],
            _state["num_filters"][1],
            3,
            stride=[2, _state["layer_strides"][1], _state["layer_strides"][1]],
            padding=[0, 1, 1],
        ),
        _state["batch_norm1d"](_state["num_filters"][1]),
        nn.ReLU(),
    )


def _sparserpn_initialize_part_2_chunk_2(_state):
    for _state["_"] in range(_state["layer_nums"][1]):
        _state["self"].block2.add(
            _state["sub_m_conv3d"](
                _state["num_filters"][1],
                _state["num_filters"][1],
                3,
                padding=1,
                indice_key="subm1",
            )
        )
        _state["self"].block2.add(_state["batch_norm1d"](_state["num_filters"][1]))
        _state["self"].block2.add(nn.ReLU())


def _sparserpn_initialize_part_2(_state):
    _sparserpn_initialize_part_2_chunk_1(_state)
    _sparserpn_initialize_part_2_chunk_2(_state)


def _sparserpn_initialize_part_3(_state):
    _state["self"].deconv2 = spconv.SparseSequential(
        _state["sp_conv3d"](
            _state["num_filters"][1], _state["num_filters"][1], (2, 1, 1), stride=1
        ),
        _state["batch_norm1d"](_state["num_filters"][1]),
        nn.ReLU(),
        spconv.ToDense(),
        Squeeze(),
        _state["conv_transpose2d"](
            _state["num_filters"][1],
            _state["num_upsample_filters"][1],
            _state["upsample_strides"][1],
            stride=_state["upsample_strides"][1],
        ),
        _state["batch_norm2d"](_state["num_upsample_filters"][1]),
        nn.ReLU(),
    )
    _state["self"].block3 = spconv.SparseSequential(
        _state["sp_conv3d"](
            _state["num_filters"][1],
            _state["num_filters"][2],
            [2, 3, 3],
            stride=[1, _state["layer_strides"][2], _state["layer_strides"][2]],
            padding=[0, 1, 1],
        ),
        _state["batch_norm1d"](_state["num_filters"][2]),
        nn.ReLU(),
    )
    for _state["_"] in range(_state["layer_nums"][2]):
        _state["self"].block3.add(
            _state["sub_m_conv3d"](
                _state["num_filters"][2],
                _state["num_filters"][2],
                3,
                padding=1,
                indice_key="subm2",
            )
        )
        _state["self"].block3.add(_state["batch_norm1d"](_state["num_filters"][2]))
        _state["self"].block3.add(nn.ReLU())


def _sparserpn_initialize_part_4(_state):
    _state["self"].deconv3 = Sequential(
        spconv.ToDense(),
        Squeeze(),
        _state["conv_transpose2d"](
            _state["num_filters"][2],
            _state["num_upsample_filters"][2],
            _state["upsample_strides"][2],
            stride=_state["upsample_strides"][2],
        ),
        _state["batch_norm2d"](_state["num_upsample_filters"][2]),
        nn.ReLU(),
    )
    _state["self"].post = Sequential(
        _state["conv2d"](
            sum(_state["num_upsample_filters"]), 128, 3, stride=1, padding=1
        ),
        _state["batch_norm2d"](128),
        nn.ReLU(),
        _state["conv2d"](128, 64, 3, stride=1, padding=1),
        _state["batch_norm2d"](64),
        nn.ReLU(),
    )
    if _state["encode_background_as_zeros"]:
        _state["num_cls"] = _state["num_anchor_per_loc"] * _state["num_class"]
    else:
        _state["num_cls"] = _state["num_anchor_per_loc"] * (_state["num_class"] + 1)
    _state["self"].conv_cls = nn.Conv2d(64, _state["num_cls"], 1)
    _state["self"].conv_box = nn.Conv2d(
        64, _state["num_anchor_per_loc"] * _state["box_code_size"], 1
    )
    if _state["use_direction_classifier"]:
        _state["self"].conv_dir_cls = nn.Conv2d(64, _state["num_anchor_per_loc"] * 2, 1)


class SparseRPN(nn.Module):
    """Don't use this."""

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPARSERPN_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SparseRPN, self).__init__()
        _state = _initialized_rpn_state(self, parameters)
        _sparserpn_initialize_part_1(_state)
        _sparserpn_initialize_part_2(_state)
        _sparserpn_initialize_part_3(_state)
        _sparserpn_initialize_part_4(_state)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        sx = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        b1 = self.block1(sx)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        up1 = self.deconv1(b1)
        up2 = self.deconv2(b2)
        up3 = self.deconv3(b3)
        x = torch.cat([up1, up2, up3], dim=1)
        x = self.post(x)
        return _prediction_dict(self, x, include_rc=False)
