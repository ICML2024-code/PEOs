PEOs
======

How to use
------
### Dataset

* DEEP10M and GloVe200: https://ann-benchmarks.com/index.html#datasets
* SIFT10M, GIST, Tiny5M and GloVe300: https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html
* DEEP100M: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search

### Compile

* Prerequisite : openmp, cmake, boost
* Prepare and Compile:
    1. Go to the root directory of PEOs.
	2. Put the base set(.fvecs), query set(.fvecs) and groundtruth set(.ivecs) into XXX folder, where XXX is the name of dataset.
	3. Check the parameter setting in the script run_XXX.sh.
    4. Execute the following commands:

```bash
$ cd /path/to/project
$ mkdir -p build && cd build
$ cmake .. && make -j
```

### Building HNSW-PEOs Index (only once)

```bash
$ bash run_XXX.sh
```

### Searching

```bash
$ bash run_XXX.sh
```