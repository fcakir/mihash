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

## Batch Results
Here we provide the latest results of MIHash and other competing work on CIFAR-10 and NUSWIDE. For reproducibility, we also provide the parameters for MIHash we used to obtain these results (see `batch-hashing/opts_batch.m`). 

### CIFAR-10
The standard setup for CIFAR-10 has two distinct settings (as specified in the papers DTSH and MIHash). The results shown here uses the VGG-F deep learning architecture and learning is done in an end-to-end fashion. For non-deep methods this corresponds to using the features at the penultimate layer of VGG-F. (Note that differently, in the MIHash paper, we do VGG-16 single-layer experiments for setting-1). 

Please refer to the above papers for details regarding setting 1 and 2. 

**Setting 1: Mean Average Precision** 

| Method  | 12-Bits | 24-Bits | 32-Bits | 48-Bits|
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| **BRE**  | 0.361  | 0.448  | 0.502  | 0.533  |
| **MACHash**  | 0.628  | 0.707  | 0.726  | 0.734  |
| **FastHash**  | 0.678  | 0.729  | 0.742  | 0.757  |
| **StructHash**  | 0.664  | 0.693  | 0.691  | 0.700  |
| **DPSH**  | 0.720  | 0.757  | 0.757  | 0.767  |
| **DTSH**  | **0.725**  | 0.773  | 0.781  | 0.810  |
| **MIHash**  | 0.687  | **0.788**  | **0.7899**  | **0.826**  |

**MIHash Parameters:**
- 12: [diary ](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar12-sp1-vggf.txt)
- 24: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar24-sp1-vggf.txt)
- 32: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar32-sp1-vggf.txt)
- 48: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar48-sp1-vggf.txt)

**Setting 2: Mean Average Precision** 

| Method  | 16-Bits | 24-Bits | 32-Bits | 48-Bits|
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| **DPSH**  | 0.908 | 0.909  | 0.917  | 0.932  |
| **DTSH**  | 0.916  | 0.924  | 0.927  | 0.934  |
| **MIHash**  | **0.922**  | **0.931**  | **0.9408**  | **0.942**  |

**MIHash Parameters:**
- 16: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar16-sp2-vggf.txt)
- 24: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar24-sp2-vggf.txt)
- 32: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar32-sp2-vggf.txt)
- 48: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar48-sp2-vggf.txt)

**NOTE:** These diaries are from older versions of the repo, where different parameter names might be used. By inspection the parameters can easily be matched to `opts_batch.m`. Notably **sigscale** is equal to **sigmf(1)**. Please email `fcakir@gmail.com` or `hekun@bu.edu` if you have any questions.  

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
