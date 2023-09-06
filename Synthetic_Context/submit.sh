#!/bin/sh
#BSUB -q gpua100
#BSUB -J UNet
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
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

CUDA_LAUNCH_BLOCKING=1 python3 main.py --name UNet_test --batch_size 4 --max-epochs 10 --num-workers 6 >| outputs/UNet_focal_loss.out 2>| error/UNet_focal_loss.err
