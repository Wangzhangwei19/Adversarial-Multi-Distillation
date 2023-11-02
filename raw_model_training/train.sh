#!/bin/bash
#for model in CNN1D CNN2D LSTM GRU MCLDNN Lenet Vgg16 Alexnet;
#for model in CNN1D MCLDNN r8conv1;
#do
#	for dataset in 1024;
#	do
#		CUDA_VISIBLE_DEVICES=0 python train.py --model $model --dataset $dataset --num_workers 8 --epochs 50
#		CUDA_VISIBLE_DEVICES=1 python train.py --model $model --dataset $dataset --num_workers 8 --epochs 50
#		CUDA_VISIBLE_DEVICES=2 python train.py --model $model --dataset $dataset --num_workers 8 --epochs 50
#		CUDA_VISIBLE_DEVICES=3 python train.py --model $model --dataset $dataset --num_workers 8 --epochs 50
#	done
#done

#!/bin/bash
for dataset in 1024;
do
	CUDA_VISIBLE_DEVICES=0 python train.py --model vgg11_bn --dataset $dataset --num_workers 8 --epochs 50 &
	CUDA_VISIBLE_DEVICES=0 python train.py --model r8conv1 --dataset $dataset --num_workers 8 --epochs 50

done
