#!/bin/sh
#BSUB -q gpua100
#BSUB -J mixes
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:45
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

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name UNet_mixes_and_originals --batch_size 1 --max-epochs 300 --num-workers 12 --mix >| outputs/UNet.out 2>| error/UNet.err

python3 test.py --n 0
python3 test.py --n 1
python3 test.py --n 2
python3 test.py --n 3
python3 test.py --n 4
python3 test.py --n 5
python3 test.py --n 6
python3 test.py --n 7
python3 test.py --n 8
python3 test.py --n 9