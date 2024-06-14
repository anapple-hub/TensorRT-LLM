# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import math
import os
import struct
import tarfile
import weakref
from functools import partial
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union

import numpy as np
from packaging import version

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

# isort: off
import torch
import tensorrt as trt
# isort: on

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})
np_float8 = np.dtype('V1', metadata={"dtype": "float8"})


def torch_to_numpy(x: torch.Tensor):
    assert isinstance(x, torch.Tensor), \
        f'x must be a torch.Tensor object, but got {type(x)}.'
    return x.detach().cpu().numpy()


def numpy_to_torch(x):
    return torch.tensor(x)


def numpy_to_dtype(x, dtype: str):
    if str_dtype_to_np(dtype) == x.dtype:
        return x
    if x.dtype not in [np_float8] and dtype not in ['fp8']:
        return x.astype(str_dtype_to_np(dtype))
    else:
        return torch_to_numpy(numpy_to_torch(x).to(str_dtype_to_torch(dtype)))


fp32_array = partial(np.array, dtype=np.float32)
fp16_array = partial(np.array, dtype=np.float16)
int32_array = partial(np.array, dtype=np.int32)


def bf16_array(x):
    x = torch_to_numpy(x)
    return x


def copy_torch_to_numpy(x: torch.Tensor, ndarray: np.array):
    torch.from_numpy(ndarray).copy_(x)
    return ndarray


def trt_version():
    return trt.__version__


# TRT supports strongly_type in 9.1
def support_strongly_type():
    return version.parse(trt_version()) >= version.parse("9.1.0")


# Preview change in TRT 10.0
def preview_trt_version():
    return version.parse(trt_version()).major > 9


def torch_version():
    return torch.__version__


_str_to_np_dict = dict(
    float16=np.float16,
    float32=np.float32,
    int32=np.int32,
    int8=np.int8,
    bool=np.bool_,
    fp8=np_float8,
)


def str_dtype_to_np(dtype):
    ret = _str_to_np_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_str_to_torch_dtype_dict = dict(
    float16=torch.float16,
    float32=torch.float32,
    int32=torch.int32,
    int8=torch.int8,
    bool=torch.bool,
)


def str_dtype_to_torch(dtype):
    ret = _str_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_str_to_trt_dtype_dict = dict(float16=trt.float16,
                              float32=trt.float32,
                              int32=trt.int32,
                              int8=trt.int8,
                              bool=trt.bool,
                              fp8=trt.fp8)


def str_dtype_to_trt(dtype):
    ret = _str_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_str_dtype_dict = {v: k for k, v in _str_to_trt_dtype_dict.items()}


def trt_dtype_to_str(dtype: trt.DataType) -> str:
    assert isinstance(dtype, trt.DataType)
    return _trt_to_str_dtype_dict[dtype]


_np_to_trt_dtype_dict = {
    np.int8: trt.int8,
    np.int32: trt.int32,
    np.float16: trt.float16,
    np.float32: trt.float32,
    np.bool_: trt.bool,

    # hash of np.dtype('int32') != np.int32
    np.dtype('int8'): trt.int8,
    np.dtype('int32'): trt.int32,
    np.dtype('float16'): trt.float16,
    np.dtype('float32'): trt.float32,
    np.dtype('bool'): trt.bool,
    np_float8: trt.fp8,
}


def np_dtype_to_trt(dtype):
    ret = _np_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_np_dtype_dict = {
    trt.int8: np.int8,
    trt.int32: np.int32,
    trt.float16: np.float16,
    trt.float32: np.float32,
    trt.bool: np.bool_,
    trt.fp8: np_float8,
}


def trt_dtype_to_np(dtype):
    ret = _trt_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_to_np_dtype_dict = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def torch_dtype_to_np(dtype):
    ret = _torch_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def trt_dtype_to_torch(dtype):
    ret = _trt_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


def is_same_dtype(type_a: Union[str, trt.DataType],
                  type_b: Union[str, trt.DataType]) -> bool:
    if isinstance(type_a, str):
        type_a = str_dtype_to_trt(type_a)

    if isinstance(type_b, str):
        type_b = str_dtype_to_trt(type_b)

    return type_a == type_b


def dim_to_trt_axes(dim):
    """Converts torch dim, or tuple of dims to a tensorrt axes bitmask"""
    if not isinstance(dim, tuple):
        dim = (dim, )

    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << d

    return axes


def trt_axes_to_dim(axes: int) -> List[int]:
    """Converts tensorrt axes bitmask to dims"""
    dim = []
    for i in range(32):
        if axes & (1 << i):
            dim.append(i)

    return dim


def dim_resolve_negative(dim, ndim):
    if not isinstance(dim, tuple):
        dim = (dim, )
    pos = []
    for d in dim:
        if d < 0:
            d = ndim + d
        pos.append(d)
    return tuple(pos)


def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def mpi_rank():
    return 0 if ENABLE_MULTI_DEVICE else mpi_comm().Get_rank()


def mpi_world_size():
    return 1 if ENABLE_MULTI_DEVICE else mpi_comm().Get_size()


def mpi_barrier():
    mpi_comm().Barrier()


def mpi_broadcast(obj, root=0):
    return mpi_comm().bcast(obj, root)


def pad_vocab_size(vocab_size, tp_size):
    return int(math.ceil(vocab_size / tp_size) * tp_size)


def to_dict(obj):
    return copy.deepcopy(obj.__dict__)


def to_json_string(obj):
    if not isinstance(obj, dict):
        obj = to_dict(obj)
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def to_json_file(obj, json_file_path):
    with open(json_file_path, "w", encoding="utf-8") as writer:
        writer.write(to_json_string(obj))


def numpy_fp32_to_bf16(src):
    # Numpy doesn't support bfloat16 type
    # Convert float32 to bfloat16 manually and assign with bf16 abstract type
    original_shape = src.shape
    src = src.flatten()
    src = np.ascontiguousarray(src)

    assert src.dtype == np.float32
    dst = np.empty_like(src, dtype=np.uint16)
    for i in range(len(dst)):
        bytes = struct.pack('<f', src[i])
        dst[i] = struct.unpack('<H', struct.pack('BB', bytes[2], bytes[3]))[0]
    return dst.reshape(original_shape)


def fromfile(dir_path, name, shape=None, dtype=None):
    dtype = np_dtype if dtype is None else dtype
    p = dir_path
    if not isinstance(p, PosixPath):
        p = Path(p)
    p = p / name

    if Path(p).exists():
        t = np.fromfile(p, dtype=dtype)
        if shape is not None:
            t = t.reshape(shape)
        return t
    return None


_extra_attrs_by_object: Dict[int, Dict[str, Any]] = {}


def get_extra_attr(obj, attr_name):
    if id(obj) not in _extra_attrs_by_object:
        return None
    extra_attrs = _extra_attrs_by_object[id(obj)]
    return extra_attrs.get(attr_name)


def _clean_extra_attrs(obj_id):
    if obj_id in _extra_attrs_by_object:
        del _extra_attrs_by_object[obj_id]


def set_extra_attr(obj, attr_name, value):
    if id(obj) not in _extra_attrs_by_object:
        _extra_attrs_by_object[id(obj)] = {}
        weakref.finalize(obj, _clean_extra_attrs, id(obj))
    _extra_attrs_by_object[id(obj)][attr_name] = value


def has_extra_attr(obj, attr_name):
    if id(obj) not in _extra_attrs_by_object:
        return False
    return attr_name in _extra_attrs_by_object[id(obj)]


def unpack_nemo_ckpt(nemo_archive_path: Union[str, Path],
                     out_dir_path: Union[str, Path]):
    nemo_archive_path = Path(nemo_archive_path)
    if not nemo_archive_path.exists():
        raise FileNotFoundError(f"{nemo_archive_path} does not exist")

    for tar_mode in ["r:", "r:gz"]:
        try:
            with tarfile.open(nemo_archive_path, mode=tar_mode) as tar_file:

                def is_within_directory(directory, target):

                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)

                    prefix = os.path.commonprefix([abs_directory, abs_target])

                    return prefix == abs_directory

                def safe_members(tar_file):
                    members = []
                    for member in tar_file.getmembers():
                        member_path = os.path.join(out_dir_path, member.name)
                        if not is_within_directory(out_dir_path, member_path):
                            raise Exception(
                                "Attempted Path Traversal in Tar File")
                        members.append(member)
                    return members

                tar_file.extractall(  # nosec B202
                    out_dir_path,
                    members=safe_members(tar_file),
                    numeric_owner=False)

            return out_dir_path
        except tarfile.ReadError:
            pass

    raise RuntimeError(f"Could not unpack {nemo_archive_path}")
