#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate latent-wampeters
which python
which torchrun
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29592 src/train.py --config conf/autoencoder.yaml