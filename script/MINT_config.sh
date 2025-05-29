#!/usr/bin/env bash

# ---------- Global environment ----------
export CUDA_VISIBLE_DEVICES=0
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

# change to the working directory(../)
cd "$(dirname "$0")/.."

# ---------- Common paths ----------
SEED=42
BASE_DIR="."
DATA_FILE="${BASE_DIR}/data/alpaca_gpt4_data_test.json"
MODEL_TRAIN_PATH="${BASE_DIR}/../cache/Llama3-8B-Base"
RESULTS_FILE="None"           # placeholder for optional eval results JSON
LOG_DIR="log"                 # all logs under ./log
mkdir -p "${LOG_DIR}"

# ---------- Warm‑up (step‑1) ----------
WARM_JOB="5-warmup-gpt4-lora"
JOB_NAME_WARM="Llama3-8B-${WARM_JOB}"
SAMPLE_RATIO=0.05
mkdir -p "${BASE_DIR}/output/model"
WARM_OUTPUT_PATH="${BASE_DIR}/output/model/${JOB_NAME_WARM}"
WARM_SCRIPT_PATH="${BASE_DIR}/mint/scripts/lora_warmup.sh"

# ---------- Gradient extraction (step‑2) ----------
TRAINING_DATA_NAME="alpaca"
GRADIENT_TYPE="adam"
INFO_TYPE="ifl"
DIM=8192                     # embedding / projection dimension, reused later
MODEL_PATH_GRAD="${WARM_OUTPUT_PATH}"
# Search for the latest model checkpoint
LATEST_MODEL=$(ls -t "${MODEL_PATH_GRAD}" | grep -E 'checkpoint-[0-9]+$' | head -n 1)
if [ -z "$LATEST_MODEL" ]; then
  echo "No model checkpoint found in ${MODEL_PATH_GRAD}"
else
  echo "Latest model checkpoint found: ${LATEST_MODEL}"
fi
MODEL_PATH_GRAD="${MODEL_PATH_GRAD}/${LATEST_MODEL}"
OUTPUT_PATH_GRAD="${BASE_DIR}/output/grads/${JOB_NAME_WARM}/"
GRAD_SCRIPT_PATH="${BASE_DIR}/mint/scripts/get_train_lora_grads.sh"

# ---------- MINT coreset selection (step‑3) ----------
MODEL_MINT="${JOB_NAME_WARM}"
GRADIENT_PATH="${OUTPUT_PATH_GRAD}"
RATIOS_LIST="0.05 0.1 0.15"
PREHEAT_RATIO=0.3
SAVE_DIR="${BASE_DIR}/output/coreset/${DIM}_${MODEL_MINT}"
MINT_PYTHON_PATH="${BASE_DIR}/mint/coreset/run_MINT.py"
# Uncomment to tune alpha
# ALPHA=0.10
# ALPHA_OPT="--alpha ${ALPHA}"

# ---------- Final training on coreset (step‑4) ----------
TRAIN_RATIO=0.05
FINAL_JOB="gpt4-MINT-${TRAIN_RATIO}"
JOB_NAME_FINAL="Llama3-8B-${FINAL_JOB}"
CORESET_FILE="${SAVE_DIR}_${TRAIN_RATIO}.npz"
OUTPUT_PATH_FINAL="${BASE_DIR}/output/model/${JOB_NAME_FINAL}"
FINAL_SCRIPT_PATH="${BASE_DIR}/mint/scripts/lora_train_coreset.sh"

# ---------- Helpers ----------
# Usage: LOG_FILE=$(log_file "$JOB_NAME")
log_file() {
  local job_name="$1"
  echo "${LOG_DIR}/${job_name}.log"
}
