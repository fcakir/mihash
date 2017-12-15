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
Here we provide the latest results of MIHash and other competing work on CIFAR-10 and NUSWIDE. These results are from our latest hashing paper [Tie-Aware Learning to Rank](https://arxiv.org/pdf/1705.08562.pdf). For reproducibility, we also provide the parameters for MIHash we used to obtain these results (see `batch-hashing/opts_batch.m`). 

### CIFAR-10
The standard setup for CIFAR-10 has two distinct settings. The deep learning architecture is VGG-F and learning is done in and end-to-end fashion. Please refer to the above paper for details. 

**Setting 1: Mean Average Precision** 

| Method  | 12-Bits | 24-Bits | 32-Bits | 48-Bits|
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| **BRE**  | 0.361  | 0.448  | 0.502  | 0.533  |
| **MACHash**  | 0.628  | 0.707  | 0.726  | 0.734  |
| **FastHash**  | 0.678  | 0.729  | 0.742  | 0.757  |
| **StructHash**  | 0.664  | 0.693  | 0.691  | 0.700  |
| **DPSH**  | 0.720  | 0.757  | 0.757  | 0.767  |
| **DTSH**  | **0.725**  | 0.773  | 0.781  | 0.810  |
| **MIHash**  | 0.687  | **0.775**  | **0.786**  | **0.822**  |

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
| **MIHash**  | **0.929**  | **0.933**  | **0.938**  | **0.942**  |

**MIHash Parameters:**
- 16: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar16-sp2-vggf.txt)
- 24: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar24-sp2-vggf.txt)
- 32: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar32-sp2-vggf.txt)
- 48: [diary](https://github.com/fcakir/mihash/blob/master/diary/batch-hashing/MI-cifar48-sp2-vggf.txt)

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
