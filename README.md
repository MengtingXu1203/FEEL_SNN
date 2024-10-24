# FEEL-SNN

Code for "FEEL-SNN: Robust Spiking NeuralNetworks with FrequencyEncoding and Evolutionary LeakFactor (NeurIPS 2024)"

## Prerequisites

The Following Setup is tested and it is working:

- Python>=3.5

- Pytorch>=1.9.0

- Cuda>=10.2

## Data preparation

- CIFAR10: `def build_cifar(use_cifar10=True)` in `data_loaders.py`

- CIFAR100: `def build_cifar(use_cifar10=False)` in `data_loaders.py`
 
 - Tiny-ImageNet: 
 
   (1) Download Tiny-ImageNet dataset
   
   (2)`def build_tiny_imagenet()` in `data_loaders.py`

## Description

- Use a triangle-like surrogate gradient `ZIF` in `layers.py` for step function forward and backward.

- Use FE method `def ft(x,freq_filter)` in `layers.py`.

- Use EL employed spiking neuron `LIFSpikeTau` in `layers.py`.


## FEEL Training \& Testing

Take CIFAR10, VGG11 as an example

   + train: run `bash script/cifar10_vgg11.sh`; 
   
   + test: run `bash script/cifar10_vgg11_test.sh`.

## Citation
```
@article{xu2024feel,
  title={FEEL-SNN: Robust Spiking Neural Networks with Frequency Encoding and  Evolutionary Leak Factor},
  author={Xu, Mengting and Ma, De and Tang, Huajin and Zheng, Qian and Pan, Gang},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
      
 