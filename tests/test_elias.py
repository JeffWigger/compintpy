import numpy as np
import pytest

from comppy.elias import *


SUPPORTED_TYPES = [np.int64, np.uint64, np.int32, np.uint32]
SUPPORTED_TYPES_NEGATIVE = [np.int64, np.int32]
POISSON_TEST_SIZE = 5000000

## Gamma Tests


@pytest.mark.parametrize("input_type", SUPPORTED_TYPES)
def test_elias_gamma_compress(input_type):
    arr = np.array([1, 2, 5, 10, 17], dtype=input_type)
    eg = EliasGamma()
    comp = eg.compress(arr)
    output = np.array([162, 138, 8, 128], dtype=np.uint8)
    assert (comp == output).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES)
def test_elias_gamma_decompress(output_dtype):
    input = np.array([162, 138, 8, 128], dtype=np.uint8)
    eg = EliasGamma()
    uncomp = eg.decompress(input, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == output).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES)
def test_elias_gamma_offset(output_dtype):
    arr = np.array([0, 1, 4, 9, 16], dtype=output_dtype)
    eg = EliasGamma(offset=1)
    comp = eg.compress(arr)
    output = np.array([162, 138, 8, 128], dtype=np.uint8)
    assert (comp == output).all()
    uncomp = eg.decompress(output, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == arr).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES_NEGATIVE)
def test_elias_gamma_offset_map_negative(output_dtype):
    arr = np.array([0, -1, 2, -5, 8], dtype=output_dtype)
    eg = EliasGamma(offset=1, map_negative_numbers=True)
    comp = eg.compress(arr)
    output = np.array([162, 138, 8, 128], dtype=np.uint8)
    assert (comp == output).all()
    uncomp = eg.decompress(output, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == arr).all()


@pytest.mark.parametrize("input_type", SUPPORTED_TYPES)
def test_elias_gamma_compress_poisson(input_type):
    arr = np.random.poisson(30, POISSON_TEST_SIZE) + 1  # elias does not work on 0s
    arr = arr.astype(input_type)
    eg = EliasGamma()
    comp = eg.compress(arr)
    arr_dec = eg.decompress(comp, POISSON_TEST_SIZE, output_dtype=input_type)
    assert len(arr) == len(arr_dec)
    assert (arr == arr_dec).all()


## Delta Tests


@pytest.mark.parametrize("input_type", SUPPORTED_TYPES)
def test_elias_delta_compress(input_type):
    arr = np.array([1, 2, 5, 10, 17], dtype=input_type)
    ed = EliasDelta()
    comp = ed.compress(arr)
    output = np.array([163, 72, 138, 32], dtype=np.uint8)
    assert (comp == output).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES)
def test_elias_delta_decompress(output_dtype):
    input = np.array([163, 72, 138, 32], dtype=np.uint8)
    ed = EliasDelta()
    uncomp = ed.decompress(input, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == output).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES)
def test_elias_delta_offset(output_dtype):
    arr = np.array([0, 1, 4, 9, 16], dtype=output_dtype)
    ed = EliasDelta(offset=1)
    comp = ed.compress(arr)
    output = np.array([163, 72, 138, 32], dtype=np.uint8)
    assert (comp == output).all()
    uncomp = ed.decompress(output, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == arr).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES_NEGATIVE)
def test_elias_delta_offset_map_negative(output_dtype):
    arr = np.array([0, -1, 2, -5, 8], dtype=output_dtype)
    ed = EliasDelta(offset=1, map_negative_numbers=True)
    comp = ed.compress(arr)
    output = np.array([163, 72, 138, 32], dtype=np.uint8)
    assert (comp == output).all()
    uncomp = ed.decompress(output, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == arr).all()


@pytest.mark.parametrize("input_type", SUPPORTED_TYPES)
def test_elias_delta_compress_poisson(input_type):
    arr = np.random.poisson(30, POISSON_TEST_SIZE) + 1  # elias does not work on 0s
    arr = arr.astype(input_type)
    ed = EliasDelta()
    comp = ed.compress(arr)
    arr_dec = ed.decompress(comp, POISSON_TEST_SIZE, output_dtype=input_type)
    assert len(arr) == len(arr_dec)
    assert (arr == arr_dec).all()


## Omega Tests


@pytest.mark.parametrize("input_type", SUPPORTED_TYPES)
def test_elias_omega_compress(input_type):
    arr = np.array([1, 2, 5, 10, 17], dtype=input_type)
    eo = EliasOmega()
    comp = eo.compress(arr)
    output = np.array([74, 186, 82, 32], dtype=np.uint8)
    assert (comp == output).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES)
def test_elias_omega_decompress(output_dtype):
    input = np.array([74, 186, 82, 32], dtype=np.uint8)
    eo = EliasOmega()
    uncomp = eo.decompress(input, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == output).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES)
def test_elias_omega_offset(output_dtype):
    arr = np.array([0, 1, 4, 9, 16], dtype=output_dtype)
    eo = EliasOmega(offset=1)
    comp = eo.compress(arr)
    output = np.array([74, 186, 82, 32], dtype=np.uint8)
    assert (comp == output).all()
    uncomp = eo.decompress(output, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == arr).all()


@pytest.mark.parametrize("output_dtype", SUPPORTED_TYPES_NEGATIVE)
def test_elias_omega_offset_map_negative(output_dtype):
    arr = np.array([0, -1, 2, -5, 8], dtype=output_dtype)
    eo = EliasOmega(offset=1, map_negative_numbers=True)
    comp = eo.compress(arr)
    output = np.array([74, 186, 82, 32], dtype=np.uint8)
    assert (comp == output).all()
    uncomp = eo.decompress(output, 5, output_dtype=output_dtype)
    output = np.array([1, 2, 5, 10, 17], dtype=output_dtype)
    assert (uncomp == arr).all()


@pytest.mark.parametrize("input_type", SUPPORTED_TYPES)
def test_elias_omega_compress_poisson(input_type):
    arr = np.random.poisson(30, POISSON_TEST_SIZE) + 1  # elias does not work on 0s
    arr = arr.astype(input_type)
    eo = EliasOmega()
    comp = eo.compress(arr)
    arr_dec = eo.decompress(comp, POISSON_TEST_SIZE, output_dtype=input_type)
    assert len(arr) == len(arr_dec)
    assert (arr == arr_dec).all()
