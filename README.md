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
