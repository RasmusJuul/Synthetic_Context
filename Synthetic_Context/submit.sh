#!/bin/sh
#BSUB -q gpua100
#BSUB -J swin50v3
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
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

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small5000v1 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --size 5000 --version v1  >| outputs/small5000v1.out 2>| error/small5000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small20000v1 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --size 20000 --version v1  >| outputs/small20000v1.out 2>| error/small20000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small50000v1 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --version v1  >| outputs/small50000v1.out 2>| error/small50000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small5000v2 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --size 5000 --version v2  >| outputs/small5000v2.out 2>| error/small5000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small20000v2 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --size 20000 --version v2  >| outputs/small20000v2.out 2>| error/small20000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small50000v2 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --version v2  >| outputs/small50000v2.out 2>| error/small50000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small5000v3 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --size 5000 --version v3  >| outputs/small5000v3.out 2>| error/small5000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small20000v3 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --size 20000 --version v3  >| outputs/small20000v3.out 2>| error/small20000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small50000v3 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --version v3  >| outputs/small50000v3.out 2>| error/small50000v3.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large5000v1 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --size 5000 --version v1  >| outputs/large5000v1.out 2>| error/large5000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large20000v1 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --size 20000 --version v1  >| outputs/large20000v1.out 2>| error/large20000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large50000v1 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --version v1  >| outputs/large50000v1.out 2>| error/large50000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large5000v2 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --size 5000 --version v2  >| outputs/large5000v2.out 2>| error/large5000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large20000v2 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --size 20000 --version v2  >| outputs/large20000v2.out 2>| error/large20000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large50000v2 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --version v2  >| outputs/large50000v2.out 2>| error/large50000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large5000v3 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --size 5000 --version v3  >| outputs/large5000v3.out 2>| error/large5000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large20000v3 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --size 20000 --version v3  >| outputs/large20000v3.out 2>| error/large20000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large50000v3 --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --version v3  >| outputs/large50000v3.out 2>| error/large50000v3.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin5000v1 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --size 5000 --version v1  >| outputs/swin5000v1.out 2>| error/swin5000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin20000v1 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --size 20000 --version v1  >| outputs/swin20000v1.out 2>| error/swin20000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin50000v1 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --version v1  >| outputs/swin50000v1.out 2>| error/swin50000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin5000v2 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --size 5000 --version v2  >| outputs/swin5000v2.out 2>| error/swin5000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin20000v2 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --size 20000 --version v2  >| outputs/swin20000v2.out 2>| error/swin20000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin50000v2 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --version v2  >| outputs/swin50000v2.out 2>| error/swin50000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin5000v3 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --size 5000 --version v3  >| outputs/swin5000v3.out 2>| error/swin5000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin20000v3 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --size 20000 --version v3  >| outputs/swin20000v3.out 2>| error/swin20000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin50000v3 --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --version v3  >| outputs/swin50000v3.out 2>| error/swin50000v3.err


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small_pca --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --pca_subset >| outputs/smallpca.out 2>| error/smallpca.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large_pca --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --pca_subset >| outputs/largepca.out 2>| error/largepca.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin_pca --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --pca_subset >| outputs/swinpca.out 2>| error/swinpca.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small_umap --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --umap_subset >| outputs/smallumap.out 2>| error/smallumap.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large_umap --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --umap_subset >| outputs/largeumap.out 2>| error/largeumap.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin_umap --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --umap_subset >| outputs/swinumap.out 2>| error/swinumap.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name small_fd --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model small --feature_distance_subset >| outputs/smallfd.out 2>| error/smallfd.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name large_fd --batch_size 20 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model large --feature_distance_subset >| outputs/largefd.out 2>| error/largefd.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 main.py --name swin_fd --batch_size 1 --max-epochs 800 --num-workers 16 --mix --seed 1997 --model swin --feature_distance_subset >| outputs/swinfd.out 2>| error/swinfd.err

