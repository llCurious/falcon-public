# MixQT: Secure Mixed-Precision Quantized Training of Deep Neural Networks

This is an anonymous repo for MixQT. MixQT is based on the code base of Falcon: Honest-Majority Maliciously Secure Framework for Private Deep Learning.

> We are sorry that the current running script is not formatted very well. We will provide a more detailed description soon.

## Running the code
1. **Build the code**: `make all -j$(nproc)`
2. **Prepare the data**: Please follow the `README` in MNIST directory to prepare the dataset.
3. **Set configuration**: Please check the configs in `makefile` (choose NETWORK and DATASET). Note that, if you do not have GPU, set `USE_CUDA := 0` in `makefile`. Besides, set `#define USE_GPU false` in `src/globals.h`
4. **Run the exp**: Execute `make command -j` in the root directory of this repo. Three threads will be used to run the 3PC experiments locally.

