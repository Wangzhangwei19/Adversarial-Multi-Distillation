#!/bin/bash
for model in CNN1D CNN2D LSTM GRU MCLDNN Lenet Vgg16 Alexnet;
do
	CUDA_VISIBLE_DEVICES=2 python test.py --model $model --dataset 128 --num_workers 5 &
	CUDA_VISIBLE_DEVICES=3 python test.py --model $model --dataset 512 --num_workers 7 &
	CUDA_VISIBLE_DEVICES=4 python test.py --model $model --dataset 1024 --num_workers 8 &
	CUDA_VISIBLE_DEVICES=6 python test.py --model $model --dataset 3040 --num_workers 8;
done


