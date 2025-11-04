"""
IO for tensors inside binary files

notes
in C, python floats are represented as double
"""
import os
import struct
import numpy as np


class TensorIO:
    @staticmethod
    def write_double(fname, tensor):
        return write_tensor_double(fname, tensor)
    
    @staticmethod
    def read_double(fname, shape=None):
        return read_tensor_double(fname, shape=shape)



def write_tensor_double(fname, tensor):
    """
    writes a tensor as a list of values with type double (32 bits)
    and saves on disk
    """
    shape = tensor.shape
    flat = tensor.flatten()
    data = struct.pack('d'*len(flat), *flat)
    # print(data, len(data))
    with open(fname, 'wb') as fd:
        fd.write(data)
    return os.path.exists(fname)


def write_tensor_int4(fname, tensor):
    """
    writes a tensor as a list of values with type int4 (4 bits)
    and saves on disk
    """
    shape = tensor.shape
    flat = tensor.flatten()
    data = struct.pack('d'*len(flat), *flat)
    # print(data, len(data))
    with open(fname, 'wb') as fd:
        fd.write(data)
    return os.path.exists(fname)


def read_tensor_double(fname, shape=None):
    """
    reads the tensor from a binary file (double, 32bits)
    """
    binary_data = open(fname, 'rb').read()
    total_float_size = len(binary_data) // 8

    # reads value of type double with 'd'
    fmt = 'd' * total_float_size
    tensor = np.array(struct.unpack(fmt, binary_data))

    # we convert to floats?
    # tensor = tensor.astype(float)


    tensor = tensor.reshape(shape) if shape is not None else tensor
    return tensor
