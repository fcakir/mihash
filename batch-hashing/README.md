# Batch (Deep) Hashing

**This portion of the repo is outdated with inferior results. Please refer to the [new repo](https://github.com/fcakir/deep-mihash) for the latest results for deep/batch learning of MIHash!**

This folder contains MatConvNet implementation of MIHash (batch learning version), described in the following paper:
- F. Cakir, K. He, S. A. Bargal, S. Sclaroff "MIHash: Online Hashing with Mutual Information" International Conference on Computer Vision (ICCV) 2017 ([pdf](https://arxiv.org/abs/1703.08919))


## Setup

If you have not done so, initialize runtime:
```Matlab
>> run ../startup.m
```

## Datasets/Models
The CIFAR dataset is supported. See `+IMDB` folder.

Two model types are supported:  single-layer (`+Models/fc1.m`) and VGG-F (`+Models/vggf.m`).

## Example Usage
Train 32-bit single-layer MIHash model on CIFAR, split 1. 
Learning rate: initial 0.1, decay by 0.5 every 10 epochs,  for 100 epochs.
Do not use GPU.
```Matlab
>> demo_cifar(32,'fc1','split',1,'lr',0.1,'lrdecay',0.5,'lrstep',10,'epochs',100,'gpus',[])
```

Training uses MatConvNet, and example log is not shown. 
The result should be close to this:
```
evaluate: mAP = 0.72766
```
