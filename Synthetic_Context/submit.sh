#!/bin/sh
#BSUB -q gpua100
#BSUB -J mixes
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[gpu80gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
#BSUB -N
# -- end of LSF options --

nvidia-smi

source ../../envs/thesis/bin/activate

# Options
# Run main.py --help to get options

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name UNet_mix --batch_size 12 --max-epochs 300 --num-workers 12 --mix >| outputs/UNet.out 2>| error/UNet.err
