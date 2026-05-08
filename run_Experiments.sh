#!/usr/bin/env bash
set -e

# JPype / KoNLPy libstdc++ ������ ��ȸ�ߴ� ������ �ʿ��ϸ� ����
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="/home/hpc-ssu/PythonProjects/mimic_preprocessing/hajin/ChemicalAccident-test/kiwook/python/dataset/"
TRAIN_FILE="train/train.csv"
VALID_FILE="valid/valid.csv"
TEST_FILE="test/test.csv"

MODEL_NAME="kobert"
LRS=("4e-5" "3e-5" "2e-5" "1e-5")
SEEDS=("42" "99" "123" "166" "1")
WARMUP_RATIO=("0.0" "0.1" "0.2")
SCHEDULER=("linear_warmup")

mkdir -p logs

for LR in "${LRS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for SCHED in "${SCHEDULER[@]}"; do
      for WARMRATIO in "${WARMUP_RATIO[@]}"; do
        EXP_NAME="Aug1.3_kobert_lr_${LR}_scheduler_${SCHED}_warmupRatio_${WARMRATIO}_seed_${SEED}"

        echo "========================================"
        echo "Running: ${EXP_NAME}"
        echo "========================================"
    
        python main.py \
          --data_directory="$DATA_DIR" \
          --train_file="$TRAIN_FILE" \
          --valid_file="$VALID_FILE" \
          --test_file="$TEST_FILE" \
          --model="$MODEL_NAME" \
          --lr="$LR" \
          --seed="$SEED" \
          --epoch=30 \
          --optimizer="adamW" \
          --scheduler=$SCHED \
          --warmup_ratio=$WARMRATIO \
          --model_directory="training_result/${EXP_NAME}" \
          2>&1 | tee "logs/${EXP_NAME}.log"
      done
    done
  done
done


MODELS=("LR" "RF" "SVM" "XGBoost")
for MODEL_NAME in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    EXP_NAME="Aug1.3_${MODEL_NAME}_seed_${SEED}"

    echo "========================================"
    echo "Running: ${EXP_NAME}"
    echo "========================================"

    python main.py \
      --data_directory="$DATA_DIR" \
      --train_file="$TRAIN_FILE" \
      --valid_file="$VALID_FILE" \
      --test_file="$TEST_FILE" \
      --model="$MODEL_NAME" \
      --seed="$SEED" \
      --model_directory="training_result/${EXP_NAME}" \
      2>&1 | tee "logs/${EXP_NAME}.log"
  done
done