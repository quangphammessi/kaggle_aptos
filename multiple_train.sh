#!/bin/bash
# nohup python train_efficientnet.py > ./logs/20190829_data3k_effib4_aug_huber.log &
# nohup python train_efficientnet_clean.py > ./logs/20190829_data3k_effib4_aug_clean.log &
nohup bash -c "python train_efficientnet.py; python train_efficientnet_clean.py" > ./logs/20190829_data3k_effib4_aug_huber_clean.log &
exit 0