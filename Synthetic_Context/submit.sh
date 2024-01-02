#!/bin/sh
#BSUB -q gpua100
#BSUB -J pca_subset
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 15:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu80gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
##BSUB -N
# -- end of LSF options --

nvidia-smi

source ../../envs/thesis/bin/activate

# Options
# Run main.py --help to get options

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name CycleGAN_TPWN_large --batch_size 1 --max-epochs 200 --num-workers 4 --gan --compiled >| outputs/CycleGAN.out 2>| error/CycleGAN.err

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name UNet_large --batch_size 20 --max-epochs 500 --num-workers 16 --mix --seed 1997 --pca_subset >| outputs/UNet_pca.out 2>| error/UNet_pca.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name UNet_large --batch_size 20 --max-epochs 500 --num-workers 16 --mix --seed 1997 --umap_subset >| outputs/UNet_umap.out 2>| error/UNet_umap.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --real
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --nn
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --old
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --new
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --new --train
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --gan


# python3 test.py --n 0
# python3 test.py --n 1
# python3 test.py --n 2
# python3 test.py --n 3
# python3 test.py --n 4
# python3 test.py --n 5
# python3 test.py --n 6
# python3 test.py --n 7
# python3 test.py --n 8
# python3 test.py --n 9
