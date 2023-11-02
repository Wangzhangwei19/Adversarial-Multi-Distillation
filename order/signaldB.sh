location=(
"/results/para/0.1/mobilenet128/1-1-1/checkpoint.pth_200.tar"
""

)

for modelL in ${location[@]}
  do
  for ((i=-4;i<19;i=i+2))
    do
      j=$i
      cd ../CleanDatasets

      python CandidatesSelection.py --dataset=128 --number=1100  --gpu_index $1 --model $2 --db $j --note $modelL --location $modelL
      cd ../Attacks
      # pgd
      python PGD_Generation.py --dataset=128 --epsilon=0.15 --epsilon_iter=0.03 --num_steps 15 --gpu_index $1 --model $2 --db $j --note $modelL --location $modelL
      # UMIfgsm
#      python UMIFGSM_Generation.py --dataset=128 --epsilon=0.15 --gpu_index $1 --model $2 --db $j --note $modelL --location $modelL
      # AA
#      python AutoAttack.py --dataset=128 --epsilon=0.15 --gpu_index $1 --model $2 --location $modelL --note $modelL --db $j
      # df
#      python DeepFool_Generation.py --dataset=128  --gpu_index $1 --model CNN1D --location $modelL --note $modelL --db $j

    done

done