for ((i=0;i<1;i++))
#for ((i=-4;i<19;i=i+2))
do
  j=$i
  cd ../CleanDatasets
#  python CandidatesSelection.py --dataset=128 --number=22000  --gpu_index $1 --model $2

  python CandidatesSelection.py --dataset=128 --number=19000  --gpu_index $1 --model $2  # 11
#  python CandidatesSelection.py --dataset=128 --number=1100  --gpu_index $1 --model $2 --db $j
  cd ../Attacks
#  python PGD_Generation.py --dataset=128 --epsilon=0.15 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2 --db $j
# pgd
  python PGD_Generation.py --dataset=128 --epsilon=0.02 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.04 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.06 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.08 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.10 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.12 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.14 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.16 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.18 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.20 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.22 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.24 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.26 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.28 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2
  python PGD_Generation.py --dataset=128 --epsilon=0.30 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2



#  python FGSM_Generation.py --dataset=128 --epsilon=0.15 --gpu_index $1 --model $2 --db $j
  python  FGSM_Generation.py --dataset=128 --epsilon=0.02 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.04 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.06 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.08 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.10 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.12 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.14 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.16 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.18 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.20 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.22 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.24 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.26 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.28 --gpu_index $1 --model $2
  python  FGSM_Generation.py --dataset=128 --epsilon=0.30 --gpu_index $1 --model $2

#  UMIfgsm
#  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.15 --gpu_index $1 --model $2 --db $j
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.02 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.04 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.06 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.08 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.10 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.12 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.14 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.16 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.18 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.20 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.22 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.24 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.26 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.28 --gpu_index $1 --model $2
  python UMIFGSM_Generation.py --dataset=128 --epsilon=0.30 --gpu_index $1 --model $2



done