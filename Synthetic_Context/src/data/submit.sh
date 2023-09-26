#!/bin/sh
#BSUB -q compute
#BSUB -J mixes
#BSUB -n 24
#BSUB -R "span[hosts=1]"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=2GB]"
##BSUB -R "select[gpu80gb]" #options gpu40gb or gpu80gb
##BSUB -o outputs/gpu_%J.out
##BSUB -e outputs/gpu_%J.err
##BSUB -N
# -- end of LSF options --

source ../../../../envs/thesis/bin/activate

# Options
# Run main.py --help to get options

# python3 prepare_cyclegan_data.py
python3 create_mixes.py