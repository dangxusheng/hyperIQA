#!/bin/bash

## step1: train based koniq_pretrained.pkl
#python3 train_test_IQA.py --dataset 'tid2013' --train_patch_num 25 --test_patch_num 25 \
#--lr 0.000025 --weight_decay 0.0008 --batch_size 32 --epochs 50 --patch_size 224 \
#--save_path './train_result/20221201_tid2013' --resume_ckpt './pretrained/koniq_pretrained.pkl'


## step2: train based  (trained on tid2013)
#python3 train_test_IQA.py --dataset 'csiq' --train_patch_num 25 --test_patch_num 25 \
#--lr 0.000025 --weight_decay 0.0008 --batch_size 48 --epochs 50 --patch_size 224 \
#--save_path './train_result/20221201_tid2013_csiq' --resume_ckpt './train_result/20221201_tid2013/epoch_best.pth'


## failed: 模型过拟合
## step3: train based  (trained on tid2013)
#python3 train_test_IQA.py --dataset 'tid2013' --train_patch_num 25 --test_patch_num 25 \
#--lr 0.000025 --weight_decay 0.0008 --batch_size 32 --epochs 50 --patch_size 224 \
#--save_path './train_result/20221201_tid2013_resnet18' --resume_ckpt './train_result/20221201_tid2013/epoch_best.pth'


## step4: train based  (trained on tid2013)
#python3 train_test_IQA.py --dataset 'tid2013' --train_patch_num 25 --test_patch_num 25 \
#--lr 0.000025 --weight_decay 0.0008 --batch_size 32 --epochs 50 --patch_size 224 \
#--save_path './train_result/20221201_tid2013_mobilenetv2' \
#--resume_ckpt './train_result/20221201_tid2013_mobilenetv2/epoch_best.pth'


#python3 train_test_IQA.py --dataset 'csiq' --train_patch_num 25 --test_patch_num 25 \
#--lr 0.000015 --weight_decay 0.0008 --batch_size 32 --epochs 50 --patch_size 224 \
#--save_path './train_result/20221201_tid2013_mobilenetv2_csiq' \
#--resume_ckpt './train_result/20221201_tid2013_mobilenetv2/epoch_best.pth'


#python3 train_test_IQA.py --dataset 'tid2013' --train_patch_num 25 --test_patch_num 25 \
#--lr 0.000015 --weight_decay 0.0008 --batch_size 64 --epochs 50 --patch_size 224 \
#--save_path './train_result/20221201_tid2013_mobilenetv2_sigmoid2' \
#--resume_ckpt './train_result/20221201_tid2013_mobilenetv2_sigmoid/epoch_best.pth'


#
# mix_dataset train
python3 train_test_IQA.py --dataset 'tid2013' --train_patch_num 25 --test_patch_num 25 \
--lr 0.00015 --weight_decay 0.0008 --batch_size 64 --epochs 50 --patch_size 224 \
--save_path './train_result/20221201_csiq_mobilenetv2' \
--resume_ckpt './train_result/20221201_csiq_mobilenetv2/epoch_best.pth'
