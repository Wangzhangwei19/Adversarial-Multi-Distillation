#!/bin/bash
for dataset in 128;
do
#	CUDA_VISIBLE_DEVICES=6 python train.py --model GRU --dataset $dataset --num_workers 8 --epochs 50
#	CUDA_VISIBLE_DEVICES=1 python train.py --model CNN1D --dataset $dataset --num_workers 8 --epochs 50 &
#	CUDA_VISIBLE_DEVICES=3 python train.py --model CNN2D --dataset $dataset --num_workers 8 --epochs 50 &
#	CUDA_VISIBLE_DEVICES=6 python train.py --model LSTM --dataset $dataset --num_workers 8 --epochs 50 &
#	CUDA_VISIBLE_DEVICES=5 python train.py --model GRU  --dataset $dataset --num_workers 8 --epochs 50 ;
#	CUDA_VISIBLE_DEVICES=2 python train.py --model MCLDNN --dataset $dataset --num_workers 8 --epochs 50 &
#	CUDA_VISIBLE_DEVICES=3 python train.py --model Lenet --dataset $dataset --num_workers 8 --epochs 50 &
#	CUDA_VISIBLE_DEVICES=7 python train.py --model Vgg16 --dataset $dataset --num_workers 8 --epochs 50 &
#	CUDA_VISIBLE_DEVICES=5 python train.py --model Alexnet --dataset $dataset --num_workers 8 --epochs 50 ;
done
#
#for dataset in 3040;
#do
#	CUDA_VISIBLE_DEVICES=2 python train.py --model CNN2D --dataset $dataset --num_workers 8 --epochs 50 &
#	CUDA_VISIBLE_DEVICES=4 python train.py --model Lenet --dataset $dataset --num_workers 8 --epochs 50 ;
#done
#
#for dataset in 512;
#do
#	CUDA_VISIBLE_DEVICES=2 python train.py --model CNN2D --dataset $dataset --num_workers 8 --epochs 50;
#done
#
#
#

