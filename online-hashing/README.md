### Online Hashing Benchmarking
This repository contains experimental code used in the below paper:
- F. Cakir, K. He, S. A. Bargal, S. Sclaroff "MIHash: Online Hashing with Mutual Information" International Conference on Computer Vision (ICCV) 2017 ([details](https://arxiv.org/abs/1703.08919))

The implementations of the online hashing methods listed below are also available in this repository:
- L. K. Huang, Q. Y. Yang and W. S. Zheng "Online Hashing" International Joint Conference on Artificial Intelligence (IJCAI) 2013 ([details](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI13/paper/view/6599))
- F. Cakir, S. Sclaroff "Adaptive Hashing for Fast Similarity Search" International Conference on Computer Vision (ICCV) 2015 ([details](http://ieeexplore.ieee.org/document/7410482/?reload=true))
- F. Cakir, S. A. Bargal, S. Sclaroff  "Online Supervised Hashing"  Computer Vision and Image Understanding (CVIU) 2016 ([details](http://www.sciencedirect.com/science/article/pii/S1077314216301606))
- C. Leng, J. Wu, J. Cheng, X. Bai and H. Lu "Online Sketching Hashing" Computer Vision and Pattern Recognition (CVPR) 2015 ([details](http://ieeexplore.ieee.org/document/7298865/))

The experimental code includes a benchmark for these hashing methods in which the hash table is continuously updated and the hash method is evaluated at random locations during online learning. Please refer below and the code for implementation details. Please cite the respective papers along with our ICCV 2017 paper if any of the code is used.

### Setup
The `localdir` parameter must be specified (see get_opts.m) and `load_cnn.m` and `load_gist.m` files must be modified to load data. See the respective files for additional details. 

In the main folder, initialize runtime:
```Matlab
>> startup
```
### Example #1
Run the OSH\* method with 2,000 training instances on the CIFAR dataset. Use 32 bit hash codes and CNN features\*. Do a single trial (`ntrials`) and put 5 checkpoints for testing (`ntest`). Override any previously ran identical experimental results (`override`), use a Hinge-loss formulation (`SGDBoost`). 
```Matlab
>> demo_osh('cnn','cifar',32,'ntrials',1, 'ntest', 5, 'noTrainingPoints',2000, 'updateInterval', 1e2, 'reservoirSize', 0, 'override', 1,'SGDBoost', 0)
```
##### Command Window Output
The command window output for the above experiment is below (with minor alterations):
```
@get_opts: identifier: 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100
             dataset: 'cifar'
               epoch: 1
          flipThresh: -1
               ftype: 'cnn'
        labelsPerCls: 0
            localdir: '/research/object_detection/cachedir/online-hashing/osh'
             mapping: 'smooth'
              metric: 'mAP'
               nbits: 32
           no_blocks: 1
    noTrainingPoints: 2000
              ntests: 5
             ntrials: 1
            nworkers: 0
            override: 1
            pObserve: 0
              prefix: ''
            randseed: 12345
       reservoirSize: 0
           showplots: 0
            testFrac: 1
             trigger: 'bf'
         tstScenario: 1
      updateInterval: 100
            val_size: 0
            methodID: 'osh'
            SGDBoost: 0
            stepsize: 0.1000
          identifier: '01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100'
           batchSize: 1
        unsupervised: 0
             windows: 0
              expdir: '/research/object_detection/cachedir/online-hashing/osh/01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100/2000pts_1epochs_5tests'
          diary_name: '/research/object_detection/cachedir/online-hashing/osh/01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100/2000pts_1epochs_5tests/diary_001.txt'
@demo: Loading data for cifar_cnn...
  Name            Size                   Bytes  Class     Attributes

  Xtest        1000x4096              32768000  double              
  Xtrain      59000x4096            1933312000  double              
  Ytest        1000x1                     8000  double              
  Ytrain      59000x1                   472000  double              

@load_cnn: Dataset "cifar" loaded in 13.72 secs
@demo: Training models...
@train: 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100: 2000 train_iters
@make_general_channel/channel_general: 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100: random trial 1
@train_osh: [T01] HT Update#1 @1, #BRs=1.888e+06, bf_all=0, trigger_val=-1(bf)
@train_osh: *checkpoint*
[T01] 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100
     (1/2000) W 0.01s, HT 0.73s(1 updates), Res 0.00s
     total #BRs=1.888e+06, avg #BF=0
@train_osh: [T01] HT Update#2 @100, #BRs=3.776e+06, bf_all=13.8245, trigger_val=-1(bf)
@train_osh: [T01] HT Update#3 @200, #BRs=5.664e+06, bf_all=3.3311, trigger_val=-1(bf)
@train_osh: [T01] HT Update#4 @300, #BRs=7.552e+06, bf_all=1.51875, trigger_val=-1(bf)
@train_osh: [T01] HT Update#5 @400, #BRs=9.44e+06, bf_all=1.98064, trigger_val=-1(bf)
@train_osh: [T01] HT Update#6 @500, #BRs=1.1328e+07, bf_all=1.20429, trigger_val=-1(bf)
@train_osh: *checkpoint*
[T01] 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100
     (573/2000) W 0.56s, HT 5.57s(6 updates), Res 0.21s
     total #BRs=1.1328e+07, avg #BF=21.8593
@train_osh: [T01] HT Update#7 @600, #BRs=1.3216e+07, bf_all=1.35307, trigger_val=-1(bf)
@train_osh: [T01] HT Update#8 @700, #BRs=1.5104e+07, bf_all=1.06468, trigger_val=-1(bf)
@train_osh: [T01] HT Update#9 @800, #BRs=1.6992e+07, bf_all=0.942661, trigger_val=-1(bf)
@train_osh: [T01] HT Update#10 @900, #BRs=1.888e+07, bf_all=1.17314, trigger_val=-1(bf)
@train_osh: *checkpoint*
[T01] 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100
     (993/2000) W 0.96s, HT 9.45s(10 updates), Res 0.37s
     total #BRs=1.888e+07, avg #BF=26.3928
@train_osh: [T01] HT Update#11 @1000, #BRs=2.0768e+07, bf_all=0.91378, trigger_val=-1(bf)
@train_osh: [T01] HT Update#12 @1100, #BRs=2.2656e+07, bf_all=0.865441, trigger_val=-1(bf)
@train_osh: [T01] HT Update#13 @1200, #BRs=2.4544e+07, bf_all=0.782797, trigger_val=-1(bf)
@train_osh: [T01] HT Update#14 @1300, #BRs=2.6432e+07, bf_all=1.10295, trigger_val=-1(bf)
@train_osh: [T01] HT Update#15 @1400, #BRs=2.832e+07, bf_all=0.852542, trigger_val=-1(bf)
@train_osh: *checkpoint*
[T01] 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100
     (1456/2000) W 1.32s, HT 13.59s(15 updates), Res 0.52s
     total #BRs=2.832e+07, avg #BF=30.9103
@train_osh: [T01] HT Update#16 @1500, #BRs=3.0208e+07, bf_all=0.846373, trigger_val=-1(bf)
@train_osh: [T01] HT Update#17 @1600, #BRs=3.2096e+07, bf_all=0.745153, trigger_val=-1(bf)
@train_osh: [T01] HT Update#18 @1700, #BRs=3.3984e+07, bf_all=0.601119, trigger_val=-1(bf)
@train_osh: [T01] HT Update#19 @1800, #BRs=3.5872e+07, bf_all=0.776441, trigger_val=-1(bf)
@train_osh: [T01] HT Update#20 @1900, #BRs=3.776e+07, bf_all=0.668542, trigger_val=-1(bf)
@train_osh: [T01] HT Update#21 @2000, #BRs=3.9648e+07, bf_all=0.605881, trigger_val=-1(bf)
@train_osh: *checkpoint*
[T01] 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100
     (2000/2000) W 1.70s, HT 19.11s(21 updates), Res 0.72s
     total #BRs=3.9648e+07, avg #BF=35.1538
@train_osh: 21 Hash Table updates, bits computed: 3.9648e+07
@train_osh: [T01] Saved: /research/object_detection/cachedir/online-hashing/osh/01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100/2000pts_1epochs_5tests/trial1.mat

@train:  Training time (total): 1.70 +/- 0.00
@train: HT_update time (total): 19.11 +/- 0.00
@train: Reservoir time (total): 0.72 +/- 0.00
@train:     Hash Table Updates (per): 21 +/- 0
@train:     Bit Recomputations (per): 3.965e+07 +/- 0
@train:              Bit flips (per): 35.15 +/- 0
@demo: 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100: Training is done.
@demo: Testing models...
Trial 1, Checkpoint     1/2000, @evaluate: mAP = 0.17385
Trial 1, Checkpoint   573/2000, @evaluate: mAP = 0.457
Trial 1, Checkpoint   993/2000, @evaluate: mAP = 0.50891
Trial 1, Checkpoint  1456/2000, @evaluate: mAP = 0.53164
Trial 1, Checkpoint  2000/2000, @evaluate: mAP = 0.55005
@test:   FINAL mAP: 0.55 +/- 0
@test:     AUC mAP: 0.444 +/- 0
@demo: 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100: Testing is done.
```

After printing out the experiment ID (`@get_opts: identifier:01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100`) and the input parameters, the data is loaded (`@demo: Loading data for cifar_cnn...
@load_cnn: Dataset "cifar" loaded in 13.72 secs`). 
Afterwards, the training begins. Two primary information are printed during learning.

##### i. When a checkpoint (for testing) is reached
```
@train_osh: *checkpoint*
[T01] 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100
     (1456/2000) W 1.32s, HT 13.59s(15 updates), Res 0.52s
     total #BRs=2.832e+07, avg #BF=30.9103
```
- `[T01]` again specifies the trial number.
- `01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100` denotes the experiment ID.
- `(1456/2000)` is the checkpoint location.
- `W 1.32s` specifies the training time for the hash method.
- `HT 13.59s(15 updates)` specifies the hash table update time and the total number of hash table updates till this checkpoint
- `Res 0.52s` specifies the reservoir maintainance time.
- `total #BRs=2.832e+07` specifies the current total amount of bit recomputations.
- `avg #BF=30.9103` specifies the current total bit flips per iteration.
##### ii. When the hash table is updated (only for OSH)
`@train_osh: [T01] HT Update#7 @600, #BRs=1.3216e+07, bf_all=1.35307, trigger_val=-1(bf)`

- `[T01]` indicates the trial number. 
- `HT Update#7` indicates it is the 7th hash table update. 
- `@600` specifies the training iteration location for the hash table update. 
- `#BRs=1.3216e+07` specifies the current total number of bit recomputations in the hash table 
- `bf_all=1.35307` specifies the current total number of bit flips in the hash table
- `trigger_val=-1(bf)` specifies the trigger threshold value in determining whether to perform an update to the hash table. `bf` denotes that the trigger type is based on `bit flips`.

Note that some of the information such as `Bit flips`, `Bit recomputations` and `Trigger Type` are rudimentary and primarily for future release purposes. 

After training the method, the performance is evaluated and displayed on the command windows as below:
```
@demo: Testing models...
Trial 1, Checkpoint     1/2000, @evaluate: mAP = 0.17385
Trial 1, Checkpoint   573/2000, @evaluate: mAP = 0.457
Trial 1, Checkpoint   993/2000, @evaluate: mAP = 0.50891
Trial 1, Checkpoint  1456/2000, @evaluate: mAP = 0.53164
Trial 1, Checkpoint  2000/2000, @evaluate: mAP = 0.55005
@test:   FINAL mAP: 0.55 +/- 0
@test:     AUC mAP: 0.444 +/- 0
@demo: 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100: Testing is done.
```
Notice that the performance metric is mAP by default. Other performance measures can be used, please see `get_opts.m` and the `metric` parameter. 
mAP is evaluated at every checkpoint and for each trial (here there is a single trial). The performance values, among other information, are stored under the folder specified by  `opts.expdir`. Afterwards, the average mAP and the AUC values of all the trials are reported as `FINAL mAP` and `AUC mAP`, respectively. Notice the std is 0 as there is a single trial. 

### Example #2
Train SketchHash on CIFAR dataset with GIST descriptors. Use 10,000 training instances. Set batch size (data chunk) to 50 and sketch size to 200. Do 3 trials with 50 test checkpoints. Update the hash table after processing every 100 training instances. Use `/research` as the results directory. Override any past results.
```Matlab 
>> [result_file_path diary_path] = demo_sketch('cnn','cifar',32,'ntrials',3, 'ntest', 50, 'noTrainingPoints',20000, 'updateInterval', 1e2,'override', 1, 'sketchSize', 200, 'batchSize', 50, 'localdir','/research')
```

Results are stored at: `result_file_path='/research/sketch/01-Jul-2017-cifar-cnn-32smooth-Ske200Bat50-U100/20000pts_1epochs_50tests/mAP_3trials.mat'`

Command window log is at: `diary_path='/research/sketch/01-Jul-2017-cifar-cnn-32smooth-Ske200Bat50-U100/20000pts_1epochs_50tests/diary_001.txt'`

### Example #3
Train AdaptHash on LabelMe dataset with GIST descriptors. Use 10,000 training instances with 2 epochs (20,000 instances in total). Use 16 bit hash codes. Set performance metric to prec@N=5. Update hash table at every 500 instances. Do 2 trials.
```Matlab
>> [result_file_path diary_path] = demo_adapthash('gist','labelme',16,'ntrials',2, 'ntest', 5, 'noTrainingPoints', 10000, 'updateInterval', 5e2, 'override', 1, 'metric','prec_n5', 'alpha', 0.9, 'stepsize', 1e-1, 'epochs', 2)
```
Results are stored at: `result_file_path='/research/object_detection/cachedir/online-hashing/adapt/01-Jul-2017-labelme-gist-16smooth-A0.9B0.01S0.1-U500/10000pts_1epochs_5tests/prec_n5_2trials.mat'`

Command window log is at: `diary_path='/research/object_detection/cachedir/online-hashing/adapt/01-Jul-2017-labelme-gist-16smooth-A0.9B0.01S0.1-U500/10000pts_1epochs_5tests/diary_001.txt'`

### Example #4
Train OKH on LabelMe dataset with GIST descriptors. Use 24 bit hash codes. Single trial. Use 2,000 training instances. Update the hash table at every 200 instances. Set performance metric to prec@K=100.
```Matlab
>> [result_file_path diary_path] = demo_okh('gist','labelme',24,'ntrials',1, 'ntest', 10, 'noTrainingPoints',2000, 'updateInterval', 2e2, 'override', 1, 'metric','prec_k100','alpha', 0.7, 'c', 1e-4,'localdir','/research/object_detection/cachedir/online-hashing')
```

Results are stored at: `result_file_path='/research/object_detection/cachedir/online-hashing/okh/01-Jul-2017-labelme-gist-24smooth-C0.0001A0.7-U200/2000pts_1epochs_10tests/prec_k100_1trials.mat'`

Command window log is at: `diary_path='/research/object_detection/cachedir/online-hashing/okh/01-Jul-2017-labelme-gist-24smooth-C0.0001A0.7-U200/2000pts_1epochs_10tests/diary_001.txt'`

