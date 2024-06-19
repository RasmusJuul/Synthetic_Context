#!/bin/sh
#BSUB -q gpuv100
#BSUB -J results
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu32gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
##BSUB -N
# -- end of LSF options --

nvidia-smi

source ../../envs/thesis/bin/activate

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --version v1 --batch_size 32
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --version v2 --batch_size 32
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 extract_features.py --version v3 --batch_size 32

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 5000 --version v1  >| results/small5000v1.out 2>| results/small5000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 20000 --version v1  >| results/small20000v1.out 2>| results/small20000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 50000 --version v1  >| results/small50000v1.out 2>| results/small50000v1.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 5000 --version v2  >| results/small5000v2.out 2>| results/small5000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 20000 --version v2  >| results/small20000v2.out 2>| results/small20000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 50000 --version v2  >| results/small50000v2.out 2>| results/small50000v2.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 5000 --version v3  >| results/small5000v3.out 2>| results/small5000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 20000 --version v3  >| results/small20000v3.out 2>| results/small20000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 50000 --version v3  >| results/small50000v3.out 2>| results/small50000v3.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 5000 --version v1  >| results/large5000v1.out 2>| results/large5000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 20000 --version v1  >| results/large20000v1.out 2>| results/large20000v1.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 50000 --version v1  >| results/large50000v1.out 2>| results/large50000v1.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 5000 --version v2  >| results/large5000v2.out 2>| results/large5000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 20000 --version v2  >| results/large20000v2.out 2>| results/large20000v2.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 50000 --version v2  >| results/large50000v2_2.out 2>| results/large50000v2_2.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 5000 --version v3  >| results/large5000v3.out 2>| results/large5000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 20000 --version v3  >| results/large20000v3.out 2>| results/large20000v3.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 50000 --version v3  >| results/large50000v3.out 2>| results/large50000v3.err

#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 5000 --version v1  >| results/swin5000v1.out 2>| results/swin5000v1.err
#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 20000 --version v1  >| results/swin20000v1.out 2>| results/swin20000v1.err
#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 50000 --version v1  >| results/swin50000v1.out 2>| results/swin50000v1.err

#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 5000 --version v2  >| results/swin5000v2.out 2>| results/swin5000v2.err
#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 20000 --version v2  >| results/swin20000v2.out 2>| results/swin20000v2.err
#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 50000 --version v2  >| results/swin50000v2.out 2>| results/swin50000v2.err

#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 5000 --version v3  >| results/swin5000v3.out 2>| results/swin5000v3.err
#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 20000 --version v3  >| results/swin20000v3.out 2>| results/swin20000v3.err
#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 50000 --version v3  >| results/swin50000v3.out 2>| results/swin50000v3.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --version single  >| results/smallsingle.out 2>| results/smallsingle.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --version single  >| results/largesingle.out 2>| results/largesingle.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --version single  >| results/swinsingle.out 2>| results/swinsingle.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --size 50000 --version v3 --fixed >| results/small50000v3_fixed.out 2>| results/small50000v3_fixed.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --size 50000 --version v3 --fixed >| results/large50000v3_fixed.out 2>| results/large50000v3_fixed.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 20000 --version v3 --fixed >| results/swin20000v3_fixed.out 2>| results/swin20000v3_fixed.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --version fd  >| results/smallfd.out 2>| results/smallfd.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --version fd  >| results/largefd.out 2>| results/largefd.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --version fd  >| results/swinfd.out 2>| results/swinfd.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --version single --fixed >| results/smallsingle_fixed.out 2>| results/smallsingle_fixed.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --version single --fixed >| results/largesingle_fixed.out 2>| results/largesingle_fixed.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --version single --fixed >| results/swinsingle_fixed.out 2>| results/swinsingle_fixed.err


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --version umap  >| results/smallumap.out 2>| results/smallumap.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --version umap  >| results/largeumap.out 2>| results/largeumap.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --version umap  >| results/swinumap.out 2>| results/swinumap.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --version gan  >| results/smallgan.out 2>| results/smallgan.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --version gan  >| results/largegan.out 2>| results/largegan.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --version gan  >| results/swingan.out 2>| results/swingan.err

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model small --version pca  >| results/smallpca.out 2>| results/smallpca.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model large --version pca  >| results/largepca.out 2>| results/largepca.err
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --version pca  >| results/swinpca.out 2>| results/swinpca.err

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model swin --size 20000 --version v3 --fixed --save >| results/swin20000v3_fixed.out 2>| results/swin20000v3_fixed.err