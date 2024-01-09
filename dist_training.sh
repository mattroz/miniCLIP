#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,3 torchrun --rdzv-backend=c10d \
                                    --rdzv-endpoint=localhost:0 \
                                    --nnodes=1 \
                                    --nproc-per-node=3 \
                                    tools/dist_train.py --path_to_config=configs/clip_base.yaml --path_to_log=logs/ 