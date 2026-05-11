#!/usr/bin/env bash
set -e

# JPype / KoNLPy libstdc++ ������ ��ȸ�ߴ� ������ �ʿ��ϸ� ����
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="dataset/"
TRAIN_FILE="train/train.csv"
VALID_FILE="valid/valid.csv"
TEST_FILE="test/test.csv"

MODEL_NAME="kobert"
LRS=("3e-5")
SEEDS=("42" "99" "123" "166" "1")
WARMUP_RATIO=("0.2")
SCHEDULER=("linear_warmup")

mkdir -p logs

for LR in "${LRS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for SCHED in "${SCHEDULER[@]}"; do
      for WARMRATIO in "${WARMUP_RATIO[@]}"; do
        EXP_NAME="Aug1.3_kobert_lr_${LR}_scheduler_${SCHED}_warmupRatio_${WARMRATIO}_seed_${SEED}"
        
        echo "========================================"
        echo "LIME: ${EXP_NAME}"
        echo "========================================"
    
        python LIME_Analysis.py \
          --data_directory="$DATA_DIR" \
          --test_file="$TEST_FILE" \
          --model_directory="training_result/${EXP_NAME}" \
          --model="kobert" \
          --num_explain=100 \
          --selection="all" \
          --lime_num_features=12 \
          --lime_num_samples=1000 \
          --seed="$SEED"
      done
    done
  done
done

MODELS=("LR" "RF" "SVM" "XGBoost")
for MODEL_NAME in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    EXP_NAME="Aug1.3_${MODEL_NAME}_seed_${SEED}"

    echo "========================================"
    echo "LIME: ${EXP_NAME}"
    echo "========================================"

    python LIME_Analysis.py \
      --data_directory="$DATA_DIR" \
      --test_file="$TEST_FILE" \
      --model_directory="training_result/${EXP_NAME}" \
      --model="$MODEL_NAME" \
      --selection="all" \
      --lime_num_features=12 \
      --lime_num_samples=1000 \
      --seed="$SEED"
  done
done