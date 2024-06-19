#!/bin/sh
#BSUB -q compute
#BSUB -J create
#BSUB -n 24
#BSUB -R "span[hosts=1]"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu32gb]" #options gpu40gb or gpu80gb
#BSUB -o ../../outputs/gpu_%J.out
#BSUB -e ../../outputs/gpu_%J.err
##BSUB -N
# -- end of LSF options --

source ../../../../envs/thesis/bin/activate

# Options
# Run main.py --help to get options

python3 create_mixes.py --kaggle
# python3 create_mixes_v1.py
python3 create_mapping_csv_for_dataloader.py --train --test --validation --version kaggle_noisy
# python3 prepare_cyclegan_data.py
# python3 create_mixed_metric_labels.py --synthetic
# python3 use_cyclegan.py
# python3 prepare_data.py
# python3 create_noise_files.py