# CompIntPy


A Python library for compressing integers with variable-length Elias compression algorithms ([universal codes](https://en.wikipedia.org/wiki/Universal_code_(data_compression))). The library was designed for compressing indices of weight matrices in decentralised machine learning. Hence, it supports multithreading for faster encoding of indices to speed up the distribution of the data to neighbouring nodes.


## Supported Algorithms



The library implements the Elias [gamma](https://en.wikipedia.org/wiki/Elias_gamma_coding), [delta](https://en.wikipedia.org/wiki/Elias_delta_coding), and [omega](https://en.wikipedia.org/wiki/Elias_omega_coding) algorithms. An example for `EliasGamma` is given below (Replace `Gamma` with your respective choices):


```
arr = np.array([0, -1, 2, -5, 8], dtype=np.int64)
eg = EliasGamma(offset=1, map_negative_numbers=True)
comp = eg.compress(arr)
uncomp = eg.decompress(output, arr.size, output_dtype=np.int64)
```

## Installation

For Linux the project is available via pip:
```
pip install compintpy
```

Or, install it from the GitHub repository:
```
git clone --recursive https://github.com/JeffWigger/compintpy.git
cd compintpy
pip install .
```

## Documentation
All three algorithms are available as Python objects: `comppy.elias.EliasGamma`, `comppy.elias.EliasDelta`, and `comppy.elias.EliasOmega`.
They are all subclasses of the follwing class:
```
class comppy.elias.Elias(offset: int = 0, map_negative_numbers: bool = False)
    """ Abstract base class of all Elias compression implementations.""""
    abstract compress(array: ndarray) → ndarray
        """ Abstract method defining the compression method used by all Elias implementations.
            Parameters:
                array (np.ndarray) – Array to be compressed. The array must be of types Union[np.int64, np.uint64, np.int32, np.uint32].
        """
    abstract decompress(array: ndarray, output_length: int, output_dtype: int64 | uint64 | int32 | uint32) → ndarray
        """Abstract method defining the decompression method used by all Elias implementations.
            Parameters:
                array (np.ndarray) – Numpy array containing data compressed with an Elias algorithm. The dtype must be np.uint8.
                output_length (int) – Length of the decompressed output.
                output_dtype (Union[np.int64, np.uint64, np.int32, np.uint32]) – Dtype of the output array.
        """"
```


## Multi-threading
The compress function is parallelized with OpenMP. You can set the number of threads by setting the OMP_NUM_THREADS environment variable, e.g.,


export OMP_NUM_THREADS=10


## Bindings


This library binds the high-performance C++ code from our sister Project [ComIntC](https://github.com/JeffWigger/compintc).
