# Hashing with Mutual Information
This repository contains Matlab implementation of the below paper:

[1] "MIHash: Online Hashing with Mutual Information", Fatih Cakir*, Kun He*, Sarah A. Bargal, and Stan Sclaroff. (* Equal contribution) International Conference on Computer Vision (ICCV), 2017 ([arXiv](https://arxiv.org/abs/1703.08919))

## Preparation
- Create or symlink a directory `cachedir` under the main directory to hold experimental results
- Run `download_data.sh` in the `data` directory
- Install or symlink [VLFeat](http://www.vlfeat.org/)  at `./vlfeat`
-  Install or symlink [MatConvNet](http://www.vlfeat.org/matconvnet/) at `./matconvnet` (For expeirments with deep CNN)

## Usage
- In the main folder, run `startup.m`
- For **online** hashing experiments: `cd online-hashing`, and run `demo_online.m` with appropriate input arguments (see `online-hashing/README.md`)
- For **batch** hashing experiments (including deep CNN): `cd batch-hashing`, and run `demo_cifar.m` with appropriate input arguments (see `batch-hashing/README.md`).

## License
BSD License,  see `LICENSE`

If you use this code in your research, please cite [1].
