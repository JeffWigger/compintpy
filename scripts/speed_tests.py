import os

from timeit import default_timer as timer

import numpy as np
import pandas as pd

from comppy.elias import Elias, EliasDelta, EliasGamma, EliasOmega


def poisson_compress_tests(test_object: Elias, input: np.ndarray) -> np.ndarray:
    comp = test_object.compress(input)
    return comp


def poisson_decompress_tests(test_object: Elias, input: np.ndarray, output_length: int, output_dtype) -> np.ndarray:
    decomp = test_object.decompress(input, output_length, output_dtype)
    return decomp


if __name__ == "__main__":
    test_objects = [("Elias Gamma", EliasGamma(1)), ("Elias Delta", EliasDelta(1)), ("Elias Omega", EliasOmega(1))]
    test_sizes = [500000000, 100000000, 10000000]  # reverse order for more efficiant memory allocation
    input_type = np.int64
    input_type_size = 8
    test_arrays = [np.random.poisson(30, size).astype(input_type) for size in test_sizes]
    threads = [1, 2, 4, 8]
    data = {
        "Compression Algorithm": [],
        "Test Size": [],
        "Threads": [],
        "Compute Time AVG": [],
        "Compute Time STD": [],
        "Input Byte Size": [],
        "Output Byte Size": [],
    }
    reruns = 5

    for compression_name, compression_algorithm in test_objects:
        for test_size, array in zip(test_sizes, test_arrays):
            for thread in threads:
                print(
                    f"Experiment: Compression Algorithm {compression_name}, Test Size {test_size}, Number of threads {thread}"
                )
                os.environ["OMP_NUM_THREADS"] = str(thread)
                total_time = []
                for _ in range(reruns):
                    start = timer()
                    res = poisson_compress_tests(compression_algorithm, array)
                    end = timer()
                    total_time.append(end - start)
                # poisson_decompress_tests(compression_algorithm, res, test_size, input_type)
                data["Compression Algorithm"].append(compression_name)
                data["Test Size"].append(test_size)
                data["Threads"].append(thread)
                data["Compute Time AVG"].append(np.mean(total_time))
                data["Compute Time STD"].append(np.std(total_time))
                data["Input Byte Size"].append(test_size * input_type_size)
                data["Output Byte Size"].append(len(res))

    df = pd.DataFrame(data)
    df.to_csv("../data/compression_test_results.csv")
