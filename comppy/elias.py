from abc import ABC, abstractmethod

import numpy as np

import _comppy.delta as delta
import _comppy.gamma as gamma
import _comppy.omega as omega


class Elias(ABC):
    def __init__(self, offset=0, map_negative_numbers=False):
        super().__init__()
        self.offset = offset
        self.map_negative_numbers = map_negative_numbers

    @abstractmethod
    def compress(self, array):
        ...

    @abstractmethod
    def decompress(self):
        ...


class EliasGamma(Elias):
    def __init__(self, offset=0, map_negative_numbers=False):
        super().__init__(offset, map_negative_numbers)

    def compress(self, array):
        return gamma.compress(array, self.offset, self.map_negative_numbers)

    def decompress(self, array, output_length, output_dtype=np.int64):
        if issubclass(output_dtype, np.int64):
            return gamma.decompress_int64(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.uint64):
            return gamma.decompress_uint64(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.int32):
            return gamma.decompress_int32(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.uint32):
            return gamma.decompress_uint32(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        raise TypeError(
            "The type of the argument 'output_dtype' must be one of numpy.int64, numpy.uint64, numpy.int32, or numpy.uint32"
        )


class EliasDelta(Elias):
    def __init__(self, offset=0, map_negative_numbers=False):
        super().__init__(offset, map_negative_numbers)

    def compress(self, array):
        return delta.compress(array, self.offset, self.map_negative_numbers)

    def decompress(self, array, output_length, output_dtype=np.int64):
        if issubclass(output_dtype, np.int64):
            return delta.decompress_int64(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.uint64):
            return delta.decompress_uint64(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.int32):
            return delta.decompress_int32(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.uint32):
            return delta.decompress_uint32(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        raise TypeError(
            "The type of the argument 'output_dtype' must be one of numpy.int64, numpy.uint64, numpy.int32, or numpy.uint32"
        )


class EliasOmega(Elias):
    def __init__(self, offset=0, map_negative_numbers=False):
        super().__init__(offset, map_negative_numbers)

    def compress(self, array):
        return omega.compress(array, self.offset, self.map_negative_numbers)

    def decompress(self, array, output_length, output_dtype=np.int64):
        if issubclass(output_dtype, np.int64):
            return omega.decompress_int64(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.uint64):
            return omega.decompress_uint64(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.int32):
            return omega.decompress_int32(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        if issubclass(output_dtype, np.uint32):
            return omega.decompress_uint32(array, array.shape[0], output_length, self.offset, self.map_negative_numbers)
        raise TypeError(
            "The type of the argument 'output_dtype' must be one of numpy.int64, numpy.uint64, numpy.int32, or numpy.uint32"
        )
