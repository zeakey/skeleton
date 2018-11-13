1. Cannot find `ImageLablemapDataLayer`
    
    You MUST use the [associated caffe code](https://github.com/zeakey/skeleton/tree/master/caffe) in `skeleton/caffe`.

2. Compile Error with cudnn library
    
    The caffe is developped based on a very old branch therefore
    may not support modern cudnn library. Please disable *cudnn* during building
    via `cmake -DUSE_CUDNN=OFF ..` (or alternatively change `USE_CUDNN=ON` to `USE_CUDNN=OFF` in `Makefile.config` if you are building with makefile). 

