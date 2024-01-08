from abc import ABC, abstractmethod
from typing import Union

import numpy as np

import _compintpy.delta as delta
import _compintpy.gamma as gamma
import _compintpy.omega as omega


class Elias(ABC):
    """
    Abstract base class of all Elias compression implementations.
    """

    def __init__(self, offset: int = 0, map_negative_numbers: bool = False):
        """
        All Elias compression algorithm support natural numbers (>0).
        However, it can be extended to all numbers by using the offset and mapping the negative numbers.

        Args:
            offset (int, optional): Offset to be added to all compression inputs. Defaults to 0.
            map_negative_numbers (bool, optional): If true, then the negative numbers are mapped to positive numbers. Defaults to False.
        """
        super().__init__()
        self.offset = offset
        self.map_negative_numbers = map_negative_numbers

    @abstractmethod
    def compress(self, array: np.ndarray) -> np.ndarray:
        """
        Abstract method defining the compression method used by all Elias implementations.

        Args:
            array (np.ndarray): Array to be compressed. The array must be of types Union[np.int64, np.uint64, np.int32, np.uint32].
        """
        ...

    @abstractmethod
    def decompress(
        self, array: np.ndarray, output_length: int, output_dtype: Union[np.int64, np.uint64, np.int32, np.uint32]
    ) -> np.ndarray:
        """
        Abstract method defining the decompression method used by all Elias implementations.

        Args:
            array (np.ndarray): Numpy array containing data compressed with an Elias algorithm. The dtype must be np.uint8.
            output_length (int): Length of the decompressed output.
            output_dtype (Union[np.int64, np.uint64, np.int32, np.uint32]): Dtype of the output array.
        """
        ...


class EliasGamma(Elias):
    """
    Implementation of the Elias Gamma compression algorithm.
    """

    def __init__(self, offset: int = 0, map_negative_numbers: bool = False):
        """
        By default it only supports natural numbers (>0).
        However, it can be extended to all numbers by using the offset and mapping the negative numbers.

        Args:
            offset (int, optional): Offset to be added to all compression inputs. Defaults to 0.
            map_negative_numbers (bool, optional): If true, then the negative numbers are mapped to positive numbers. Defaults to False.
        """
        super().__init__(offset, map_negative_numbers)

    def compress(self, array: np.ndarray) -> np.ndarray:
        """
        Compresses an input array of integers with the Elias Gamma algorithm.

        Args:
            array (np.ndarray): Array to be compressed. The array must be of types Union[np.int64, np.uint64, np.int32, np.uint32].

        Returns:
            np.ndarray: Array of compressed integers. The dtype of the array is np.uint8.
        """
        return gamma.compress(array, self.offset, self.map_negative_numbers)

    def decompress(
        self,
        array: np.ndarray,
        output_length: int,
        output_dtype: Union[np.int64, np.uint64, np.int32, np.uint32] = np.int64,
    ) -> np.ndarray:
        """
        Decompress an array compressed by the Elias Gamma algorithm.

        Args:
            array (np.ndarray): Numpy array containing the compressed data. The dtype must be np.uint8.
            output_length (int): Length of the decompressed output.
            output_dtype (Union[np.int64, np.uint64, np.int32, np.uint32], optional): Dtype of the output array. Defaults to np.int64.

        Raises:
            TypeError: If the `output_dtype` is not supported a TypeError  is raised.

        Returns:
            np.ndarray: Array of integers of type `output_dtype`.
        """
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
    """
    Implementation of the Elias Delta compression algorithm.
    """

    def __init__(self, offset: int = 0, map_negative_numbers: bool = False):
        """
        By default it only supports natural numbers (>0).
        However, it can be extended to all numbers by using the offset and mapping the negative numbers.

        Args:
            offset (int, optional): Offset to be added to all compression inputs. Defaults to 0.
            map_negative_numbers (bool, optional): If true, then the negative numbers are mapped to positive numbers. Defaults to False.
        """
        super().__init__(offset, map_negative_numbers)

    def compress(self, array: np.ndarray) -> np.ndarray:
        """
        Compresses an input array of integers with the Elias Delta algorithm.

        Args:
            array (np.ndarray): Array of integers to be compressed. The array must be of types Union[np.int64, np.uint64, np.int32, np.uint32].

        Returns:
            np.ndarray: Array of compressed integers. The dtype of the array is np.uint8.
        """
        return delta.compress(array, self.offset, self.map_negative_numbers)

    def decompress(
        self,
        array: np.ndarray,
        output_length: int,
        output_dtype: Union[np.int64, np.uint64, np.int32, np.uint32] = np.int64,
    ) -> np.ndarray:
        """
        Decompress an array compressed by the Elias Delta algorithm.

        Args:
            array (np.ndarray): Numpy array containing the compressed data. The dtype must be np.uint8.
            output_length (int): Length of the decompressed output.
            output_dtype (Union[np.int64, np.uint64, np.int32, np.uint32], optional): Dtype of the output array. Defaults to np.int64.

        Raises:
            TypeError: If the `output_dtype` is not supported a TypeError  is raised.

        Returns:
            np.ndarray: Array of integers of type `output_dtype`.
        """
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
    """
    Implementation of the Elias Omega compression algorithm.
    """

    def __init__(self, offset: int = 0, map_negative_numbers: bool = False):
        """
        By default it only supports natural numbers (>0).
        However, it can be extended to all numbers by using the offset and mapping the negative numbers.

        Args:
            offset (int, optional): Offset to be added to all compression inputs. Defaults to 0.
            map_negative_numbers (bool, optional): If true, then the negative numbers are mapped to positive numbers. Defaults to False.
        """
        super().__init__(offset, map_negative_numbers)

    def compress(self, array: np.ndarray) -> np.ndarray:
        """
        Compresses an input array of integers with the Elias Omega algorithm.

        Args:
            array (np.ndarray): Array of integers to be compressed. The array must be of types Union[np.int64, np.uint64, np.int32, np.uint32].

        Returns:
            np.ndarray: Array of compressed integers. The dtype of the array is np.uint8.
        """
        return omega.compress(array, self.offset, self.map_negative_numbers)

    def decompress(
        self,
        array: np.ndarray,
        output_length: int,
        output_dtype: Union[np.int64, np.uint64, np.int32, np.uint32] = np.int64,
    ) -> np.ndarray:
        """
        Decompress an array compressed by the Elias Omega algorithm.

        Args:
            array (np.ndarray): Numpy array containing the compressed data. The dtype must be np.uint8.
            output_length (int): Length of the decompressed output.
            output_dtype (Union[np.int64, np.uint64, np.int32, np.uint32], optional): Dtype of the output array. Defaults to np.int64.

        Raises:
            TypeError: If the `output_dtype` is not supported a TypeError  is raised.

        Returns:
            np.ndarray: Array of integers of type `output_dtype`.
        """
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
