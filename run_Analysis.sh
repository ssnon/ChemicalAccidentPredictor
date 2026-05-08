#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="/home/hpc-ssu/PythonProjects/mimic_preprocessing/hajin/ChemicalAccident-test/kiwook/python/dataset/"
TEST_FILE="test/test.csv"

MODEL_NAME="kobert"
LRS=("4e-5" "3e-5" "2e-5" "1e-5")
SEEDS=("42" "99" "123" "166" "1")
WARMUP_RATIO=("0.0" "0.1" "0.2")
SCHEDULER=("linear_warmup")

for LR in "${LRS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for SCHED in "${SCHEDULER[@]}"; do
      for WARMRATIO in "${WARMUP_RATIO[@]}"; do
        EXP_NAME="Aug1.3_kobert_lr_${LR}_scheduler_${SCHED}_warmupRatio_${WARMRATIO}_seed_${SEED}"
    
        python lime_analysis_kobert.py \
          --data_directory="$DATA_DIR" \
          --test_file="$TEST_FILE" \
          --model_directory="training_result/${EXP_NAME}" \
          --model="kobert" \
          --num_explain=100 \
          --selection="first" \
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
    EXP_NAME="${MODEL_NAME}_seed_${SEED}"

    echo "========================================"
    echo "LIME: ${EXP_NAME}"
    echo "========================================"

    python lime_analysis_classicML.py \
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