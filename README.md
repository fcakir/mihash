# online-hashing

Online hashing with reservoir sampling

by: SAB, FC, KH, (SS)

## running experiments

For example, CIFAR GIST features, 16 bits, smooth mapping (default), 3 random trials, use 5,000 training points, update hash table every 200 iters, and test at 10 checkpoints.
``` matlab
demo_osh('gist', 'cifar', 16, 'ntrials', 3, 'noTrainingPoints', 5000, 'updateInterval', 200, 'ntests', 10);
```
for more parameters, refer to `get_opts.m`


## intermediate results

Results are cached by default to: `/research/object_detection/cachedir/online-hashing`

If really necessary, change it by setting opts.localdir
### Example
In the main folder, initialize runtime:
```Matlab
>> startup
```
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
Afterwards, the training begins. Two primary information is printed.
##### i. When the hash table is updated (only for OSH)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Trial No.`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Iteration No.`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`# of Bit flips`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Trigger Type`
`--------------↓-----------------↓----------------------↓----------------------------↓--`
`@train_osh: [T01] HT Update#7 @600, #BRs=1.3216e+07, bf_all=1.35307, trigger_val=-1(bf)`
`----↑-----------------↑---------------↑---------------------------------↑--------------`
`In method`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Hash Table Update No.` `# of Bit Recomputations`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Trigger threshold value`
##### ii. When a checkpoint (for testing) is reached
`Trial No.`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Experiment ID`
`--↓------------------↓---------------------------`
`[T01] 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100`

`Checkpoint location`&nbsp;&nbsp;&nbsp;&nbsp;`Hash Table Update time` `Reservoir Update time`
`--↓----------------------↓--------------------↓------`
`(1456/2000) W 1.32s, HT 13.59s(15 updates), Res 0.52s`
`-------------↑--------------------↑------------------`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Training time`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Number of Hash Table Updates till this checkpoint`
`Total Bit recomputations`
`-------↓-----------------------------`
`total #BRs=2.832e+07, avg #BF=30.9103`
`---------------------------↑---------`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Bit Flips per Iteration`

Note that some of the information such as `Bit flips`, `Bit recomputations` and `Trigger Type` are rudimentary and primarily for future release purposes.

After training the method the testing is done:
`
@demo: Testing models...`
`Trial 1, Checkpoint     1/2000, @evaluate: mAP = 0.17385`
`Trial 1, Checkpoint   573/2000, @evaluate: mAP = 0.457`
`Trial 1, Checkpoint   993/2000, @evaluate: mAP = 0.50891`
`Trial 1, Checkpoint  1456/2000, @evaluate: mAP = 0.53164`
`Trial 1, Checkpoint  2000/2000, @evaluate: mAP = 0.55005`
`@test:   FINAL mAP: 0.55 +/- 0`
`@test:     AUC mAP: 0.444 +/- 0`
`@demo: 01-Jul-2017-cifar-cnn-32smooth-B0_S0.1-U100: Testing is done.`

