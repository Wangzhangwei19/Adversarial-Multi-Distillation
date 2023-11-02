#!/bin/bash
for model in CNN1D CNN2D LSTM GRU MCLDNN Lenet Vgg16 Alexnet;
do
	for dataset in 128 1024 3040;
	do
		CUDA_VISIBLE_DEVICES=3 python train.py --model $model --dataset $dataset --num_workers 8 --epochs 1;
	done
done


