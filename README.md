# Hashing with Mutual Information
This repository contains Matlab implementation of the below paper:

"MIHash: Online Hashing with Mutual Information",  
Fatih Cakir*, Kun He*, Sarah A. Bargal, and Stan Sclaroff. (* Equal contribution)  
International Conference on Computer Vision (ICCV), 2017 ([arXiv](https://arxiv.org/abs/1703.08919))

## Preparation
- Create or symlink a directory `cachedir` under the main directory to hold experimental results
- Run `download_data.sh` in the `data` directory
- Install or symlink [VLFeat](http://www.vlfeat.org/)  at `./vlfeat` (for computing performance metrics)
- Install or symlink [MatConvNet](http://www.vlfeat.org/matconvnet/) at `./matconvnet` (for batch hashing experiments)

## Usage
- In the main folder, run `startup.m`
- For **online** hashing experiments: `cd online-hashing`, and run `demo_online.m` with appropriate input arguments (see `online-hashing/README.md`)
- For **batch** hashing experiments: `cd batch-hashing`, and run `demo_cifar.m` with appropriate input arguments (see `batch-hashing/README.md`).

## License
BSD License,  see `LICENSE`

If you use this code in your research, please cite:
```
@inproceedings{mihash,
  title={MIHash: Online Hashing with Mutual Information},
  author={Fatih Cakir and Kun He and Sarah A. Bargal and Stan Sclaroff},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```
