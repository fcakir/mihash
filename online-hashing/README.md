# Online Hashing Benchmark
This folder contains online hashing benchmark code used in the following paper:
- F. Cakir, K. He, S. A. Bargal, S. Sclaroff "MIHash: Online Hashing with Mutual Information" International Conference on Computer Vision (ICCV) 2017 ([pdf](https://arxiv.org/abs/1703.08919))

The implementations of the below online hashing methods are also available:
- L. K. Huang, Q. Y. Yang and W. S. Zheng "Online Hashing" International Joint Conference on Artificial Intelligence (IJCAI) 2013 ([pdf](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI13/paper/view/6599))
- C. Leng, J. Wu, J. Cheng, X. Bai and H. Lu "Online Sketching Hashing" Computer Vision and Pattern Recognition (CVPR) 2015 ([pdf](http://ieeexplore.ieee.org/document/7298865/))
- F. Cakir, S. Sclaroff "Adaptive Hashing for Fast Similarity Search" International Conference on Computer Vision (ICCV) 2015 ([pdf](http://ieeexplore.ieee.org/document/7410482/?reload=true))
- F. Cakir, S. A. Bargal, S. Sclaroff  "Online Supervised Hashing"  Computer Vision and Image Understanding (CVIU) 2016 ([pdf](http://www.sciencedirect.com/science/article/pii/S1077314216301606))

The hash table is continuously updated and the hash method is evaluated at random locations 
during online learning. Please refer to the code for implementation details. 
Please cite the respective papers along with our ICCV 2017 paper if this code is used in your research.

## Setup

If you have not done so, initialize runtime:
```Matlab
>> run ../startup.m
```

## Datasets/Methods
We provide loaders for the three datasets considered in the paper: CIFAR, Places, LabelMe.
Please refer to the `+Datasets` folder for details.

Online hashing methods are implemented in `+Methods`.

## Example Usage
Train 32-bit MIHash model (ICCV'17) with 2,000 training instances on the CIFAR dataset. 
Use a reservoir of size 200 (for estimating gradients), and update the hash table after 
every 100 examples (no Trigger Update check).
Do a single random trial and use 5 checkpoints for testing. 
Override any previous experiments. 
```Matlab
>> demo_online('MIHash','cifar',32,'ntrials',1,'ntest',5,'numTrain',2000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)
```

### Command Window Output
The command window output for the training portion of this experiment should look like this (with minor alterations):
```
opts_online: identifier: 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX
           dataset: 'cifar'
             epoch: 1
            metric: 'mAP'
             nbits: 32
            ntests: 5
           ntrials: 1
          numTrain: 2000
          override: 1
            prefix: ''
          randseed: 12345
     reservoirSize: 200
         showplots: 0
           trigger: 'fix'
     triggerThresh: 0
    updateInterval: 100
             decay: 0
           no_bins: 16
         normalize: 1
          sigscale: 10
          stepsize: 1
        identifier: '20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX'
         batchSize: 1
          methodID: 'MIHash'
      unsupervised: 0
              dirs: [1×1 struct]

cifar: [CIFAR10_CNN] loaded in 61.69 secs
demo_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX: Training ...
demo_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX: random trial 1
  MIHash with properties:

     no_bins: 16
    sigscale: 10
    stepsize: 1
       decay: 0
      initRS: 20

train_online: 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX: 2000 train_iters
train_online: 
train_online: [T1] CHECKPOINT @ iter 1/2000 (batchSize 1)
train_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX
train_online: W 0.07s, HT 0.00s (0 updates), Res 0.00s. #BR: 0
train_online: 
trigger_update: [Fix] iter 100/2000, update = 1
trigger_update: [Fix] iter 200/2000, update = 1
trigger_update: [Fix] iter 300/2000, update = 1
trigger_update: [Fix] iter 400/2000, update = 1
trigger_update: [Fix] iter 500/2000, update = 1
train_online: 
train_online: [T1] CHECKPOINT @ iter 573/2000 (batchSize 1)
train_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX
train_online: W 1.76s, HT 1.67s (5 updates), Res 0.54s. #BR: 9.44e+06
train_online: 
trigger_update: [Fix] iter 600/2000, update = 1
trigger_update: [Fix] iter 700/2000, update = 1
trigger_update: [Fix] iter 800/2000, update = 1
trigger_update: [Fix] iter 900/2000, update = 1
train_online: 
train_online: [T1] CHECKPOINT @ iter 993/2000 (batchSize 1)
train_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX
train_online: W 3.05s, HT 2.95s (9 updates), Res 0.77s. #BR: 1.7e+07
train_online: 
trigger_update: [Fix] iter 1000/2000, update = 1
trigger_update: [Fix] iter 1100/2000, update = 1
trigger_update: [Fix] iter 1200/2000, update = 1
trigger_update: [Fix] iter 1300/2000, update = 1
trigger_update: [Fix] iter 1400/2000, update = 1
train_online: 
train_online: [T1] CHECKPOINT @ iter 1456/2000 (batchSize 1)
train_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX
train_online: W 4.35s, HT 4.51s (14 updates), Res 1.01s. #BR: 2.64e+07
train_online: 
trigger_update: [Fix] iter 1500/2000, update = 1
trigger_update: [Fix] iter 1600/2000, update = 1
trigger_update: [Fix] iter 1700/2000, update = 1
trigger_update: [Fix] iter 1800/2000, update = 1
trigger_update: [Fix] iter 1900/2000, update = 1
trigger_update: [Fix] iter 2000/2000, update = 1
train_online: 
train_online: [T1] CHECKPOINT @ iter 2000/2000 (batchSize 1)
train_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX
train_online: W 5.73s, HT 6.42s (20 updates), Res 1.24s. #BR: 3.78e+07
train_online: 
train_online: HT updates: 20, bits computed: 37760000
train_online: [Trial 1] Saved: ../cachedir/MIHash/20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX/2000pts_1epochs_5tests/trial1.mat

demo_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX: Training is done.
================================================
       Training Time: 5.73 +/- 0.00
      HT update time: 6.42 +/- 0.00
      Reservoir time: 1.24 +/- 0.00

  Hash Table Updates: 20 +/- 0
  Bit Recomputations: 37760000 +/- 000
================================================
```

The experiment ID is prefixed with the current date, 
e.g. `20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX`.
After priting input parameters, the dataset is loaded and training begins. 
Two major types of events are logged during training.

#### i. When a checkpoint is reached
```
train_online: [T1] CHECKPOINT @ iter 2000/2000 (batchSize 1)
train_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX
train_online: W 5.73s, HT 6.42s (20 updates), Res 1.24s. #BR: 3.78e+07
```
- `[T1]` specifies the trial number.
- `@ iter 2000/2000` is the checkpoint location.
- `W 5.73s` the training time for the hash method.
- `HT 6.42s (20 updates)` time spent in updating the hash table, and the total number of updates up to this checkpoint.
- `Res 1.24s` time spent in maintaining the reservoir.
- `#BR: 3.78e+07` current total amount of bit recomputations.

#### ii. When the hash table is updated
```
trigger_update: [Fix] iter 1500/2000, update = 1
```

- `[Fix]` indicates the update strategy. 
- `iter 1500/2000` is the current iteration.
- `update = 1` result of the update check (always 1 for `fix`).

After training, performance is evaluated and displayed:
```
demo_online: [MIHash] 20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX: Testing ...
Trial 1, Checkpoint     1/2000, evaluate: mAP = 0.20967
Trial 1, Checkpoint   573/2000, evaluate: mAP = 0.32368
Trial 1, Checkpoint   993/2000, evaluate: mAP = 0.37662
Trial 1, Checkpoint  1456/2000, evaluate: mAP = 0.40889
Trial 1, Checkpoint  2000/2000, evaluate: mAP = 0.45064
demo_online: 
demo_online:   AUC mAP: 0.354 +/- 0
demo_online: FINAL mAP: 0.451 +/- 0
```
The default performance metric is mAP, and can be specified by the `metric` parameter.
The instataneous performance at each checkpoint is recorded, and the area under curve (AUC) 
and final performance are reported in the end. 
The std is 0 in this case as there is a single trial. 

Final bits of information:
```
demo_online: All done.
demo_online: Results file: ../cachedir/MIHash/20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX/2000pts_1epochs_5tests/mAP_1trials.mat
demo_online:   Diary file: ../cachedir/MIHash/20170901-cifar-32bit-Bins16Sig10-S1D0-U100-R200FIX/2000pts_1epochs_5tests/diary_001.txt
```
