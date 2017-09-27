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
Update the hash table after every 100 examples (no Trigger Update check).
Do a single random trial and use 5 checkpoints for testing. 
Override any previous experiments. 
```Matlab
>> demo_online('MIHash','cifar',32,'ntrials',1,'ntest',5,'numTrain',2000,'updateInterval',100,'reservoirSize',0,'override',1)
```

### Command Window Output
The command window output for the above experiment should look like this (with minor alterations):
```
TODO: training log
```

After printing out the experiment ID (`get_opts: identifier:01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100`) 
and the input parameters, the data is loaded (`@demo: Loading data for cifar_cnn...
@load_cnn: Dataset "cifar" loaded in 13.72 secs`). 
Afterwards, the training begins. Two primary information are printed during learning.

#### i. When a checkpoint (for testing) is reached
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

#### ii. When the hash table is updated (only for OSH)
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
Notice that the performance metric is mAP by default. 
Other performance measures can be used, please check the `metric` parameter in `get_opts.m`. 
mAP is evaluated at every checkpoint and for each trial. 
The performance values, among other information, are stored under the folder specified by `opts.expdir`. 
Afterwards, the average mAP and the AUC values of all the trials are reported as `FINAL mAP` and `AUC mAP`, respectively. 
Notice the std is 0 as there is a single trial. 


Results are stored at: `result_file_path='/research/object_detection/cachedir/online-hashing/okh/01-Jul-2017-labelme-gist-24smooth-C0.0001A0.7-U200/2000pts_1epochs_10tests/prec_k100_1trials.mat'`

Command window log is at: `diary_path='/research/object_detection/cachedir/online-hashing/okh/01-Jul-2017-labelme-gist-24smooth-C0.0001A0.7-U200/2000pts_1epochs_10tests/diary_001.txt'`

