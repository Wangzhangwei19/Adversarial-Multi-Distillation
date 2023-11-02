#00.010.01 0.5 1
cd ..
for K in 0;
do
  python AMD.py \
         --save_root "./results/para/0.1/mobilenet128" \
         --t1_model "" \
         --t2_model "" \
         --s_init "/base-mobilenet128/initial_rmobilenet.pth.tar" \
         --dataset 128  --t1_name MCLDNN --t2_name Vgg16  --s_name mobilenet \
         --kd_mode logits --lambda_kd1 0.5 --lambda_kd2 1.2 --lambda_kd3 6 \
         --epsilon 0.06 --step 5 --step_size 0.03 \
         --note 0.5-1.2-6\
          --gpu_index 4 &




done