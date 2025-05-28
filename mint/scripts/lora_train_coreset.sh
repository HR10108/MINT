#!/usr/bin/env bash
set -euo pipefail
child_pid=0
cleanup(){
  if [[ $child_pid -ne 0 ]]; then
    echo "Received signal, terminating training (PID=$child_pid)…"
    kill "$child_pid" 2>/dev/null || true
  fi
  exit
}
trap cleanup SIGINT SIGTERM

source mint/scripts/base_training_args.sh

train_files=$1
model_path=$2
job_name=$3
results_file=$4
coreset_file=$5
output_dir=$6
seed=$7
mkdir -p "$output_dir"

echo "Model path：$model_path"
echo "Save output directory：$output_dir"


echo $base_training_args
nvidia-smi

LOG_FILE="$output_dir/train.log"

ID=$RANDOM
PORT=$((12000 + RANDOM % 20000)) 
export header="torchrun --nproc_per_node 1 --nnodes 1 --master_port $PORT \
-m mint.train.train_coreset"

bash -c "\
  $header $base_training_args \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --train_files $train_files \
    --coreset_file $coreset_file \
    --seed $seed \
  2>&1 | tee \"$LOG_FILE\"" &
child_pid=$!

echo "🚀 Training started (PID=$child_pid), logs written to $LOG_FILE"
wait "$child_pid"
exit_code=$?

echo "Training process (PID=$child_pid) has exited with code $exit_code"
exit $exit_code