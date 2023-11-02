# Learn to Defend: Adversarial Multi-Distillation for Automatic Modulation Recognition Models
### by [Zhuangzhi Chen], [Zhangwei Wang], [Dongwei Xu], [Jiawei Zhu], [Weiguo Shen], [Shilian Zheng], [Qi Xuan], [Xiaoniu Yang]



### Main components

| File name                              | Explanation                                                  |
|:---------------------------------------| :----------------------------------------------------------- |
| `AMD.py`                               | Execute the main code, with distillation model training.     |
| `CleanDatasets/CandidatesSelection.py` | Code for extracting samples with correct model classification . |
| `Attacks/*`                            | Code for the adversarial attacks we use.                     |
| `Defenses/*`                           | Code for generating defense model.                           |
| `model/*`                              | Code for all model architecture we use in experiments.       |
| `raw_model_training/*`                 | Code for model architecture and baseline training scripts.   |
| `Utils/dataset.py`                     | Code for dataset preparation.                                |
| `order/*`                              | Code for running script .                                    |
| `args.py`                              | Code for configuration.                                      |



## Running Codes

### How to train baseline models

```python
python train.py --model CNN1D --dataset 128 --num_workers 8 --epochs 50
```

Alternatively, scripts can be used to batch train the model

```bash
bash raw_model_training/train.sh
```

### How to train adversarial distillation models

To run the experiments, enter the following command.
```python
python AMD.py \
         --save_root "model save root" \
         --t1_model "clean teacher model root" \
         --t2_model "adversarial teacher model root" \
         --s_init "student model root" \
         --dataset 128   --t1_name MCLDNN --t2_name Vgg16  --s_name mobilenet \
         --kd_mode logits --lambda_kd1 1 --lambda_kd2 1 --lambda_kd3 1 \
         --note \
         --gpu_index  
```



### How to generate defense model

|  Defenses  | Commands with default parameters                             |
| :--------: | ------------------------------------------------------------ |
|  **PAT**   | python PAT_Test.py --dataset=128--eps=0.06 --step_num=5 --step_size=0.03 --model CNN1D |
| **TRADES** | python train_trades.py --dataset=128--eps=0.06 --step_num=5 --step_size=0.03 --model CNN1D |
|   **DD**   | python DD_Test.py --dataset=128 --temp 30.0 --model CNN1D    |



### How to evaluate model with attack method

1. Obtain samples with correct model classifacation. Clean samples will saved at `/CleanDatasets`

```pyhton
cd ./CleanDatasets
python CandidatesSelection.py --dataset=128 --number=$1  --gpu_index $2 --model $3  --location $4
```

2. Use the attack methods provided below. Adversarial samples will saved at `/AdversarialExampleDatasets`

|   Attacks   | Commands with default parameters                             |
| :---------: | ------------------------------------------------------------ |
|   **PGD**   | PGD_Generation.py --dataset=128 --epsilon=0.15 --epsilon_iter=0.03 --num_steps 15  --model=CNN1D |
| **UMIFGSM** | python UMIFGSM_Generation.py --dataset=128 --epsilon=0.15  --model=CNN1D |
|   **AA**    | python AutoAttack.py --dataset=128 --epsilon=0.15  --model=CNN1D |
|   **DF**    | python DeepFool_Generation.py  --dataset=128 --max_iters=15  -- overshoot=0.02 --model=CNN1D |
| **SQUARE**  | python square.py --Linf=0.3 --num_queries=3000 --model CNN1D |



### Package Requirements

To ensure success running of the program, the versions Python packages we used are listed in `requirements.txt`.To align the versions of your packages to this file, simply run:

```
pip install -r requirements.txt
```

