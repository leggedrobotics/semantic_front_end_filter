# AdaBins

> Migrated from [shariqfarooq123/AdaBins](https://github.com/shariqfarooq123/AdaBins.git)

## Training on euler

### Dataset

Dataset is uploaded on `/cluster/work/riner/users/PLR-2022/semantic_frontend_filter/extract_trajectories.tar`

### Environment

The code can run in euler official installation of python, plus a local package msgpack_numpy

**Load Environment Module**

```bash
# on euler: Execute this every time logging in
env2lmod
module load gcc/8.2.0 python/3.8.5
```

**Install msgpack_numpy**

```bash
# After loading python/3.8.5 on euler
python3 -m pip install --user msgpack-numpy
python3 -m pip install --user geffnet
python3 -m pip install --user simple-parsing
```

this will install the package `msgpack-numpy` to your `$HOME/.local/lib/python3.8/site-packages` 

### Run jobs

**script for copying dataset to $TMPDIR**

Save the following script into a file "run_train_adabins.sh"

```bash
#!/usr/bin/env bash
tar -xf extract_trajectories.tar -C $TMPDIR
cd semantic_front_end_filter/adabins/
python3 train.py  args_train.txt
```

**submit jobs onto cluster**

```bash
bsub -n 32 \
-R "rusage[mem=1000,ngpus_excl_p=1]" \
-W 04:00 \
-R "select[gpu_mtotal0>=9000]" \
-R "rusage[scratch=200]" \
-R "select[gpu_driver>=470]" \
bash ./run_train_adabins.sh
```

- `-W` specifies the time
- `-n` specifies the number of cores
- `mem=xxx` the total memory is `n*mem`
- `scratch=200` is the space in `$TMPDIR`

## Evaluation on local Machine

1. scp checkpoints from euler to local machine

### Visulize input and output of the model

Run `vismodel.py`, a simple script to load a data from test loader, run it through model and plot.

