#!/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --account=def-bakhshai
#SBATCH --mem=32G            # memory per node
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6      # CPU cores/threads
#SBATCH --ntasks-per-node=12
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python cuda cudnn

# Activate your enviroment
source ~/envs/hello_world/bin/activate

# Variables for readability
logdir=/home/arshiat/scratch/saved
datadir=/home/arshiat/scratch/data
# datadir=$SLURM_TMPDIR

tensorboard --logdir=${logdir}/lightning_logs --host 0.0.0.0 --load_fast false & \
    python ~/projects/pl_mnist_example/train.py \
    --model Conv \
    --dataloader MNIST \
    --batch_size 32 \
    --epoch 10 \
    --num_workers 10 \
    --logdir ${logdir} \
    --data_dir  ${datadir}