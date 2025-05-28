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

train_files=$1
model_path=$2
job_name=$3
results_file=$4
output_dir=$5
sample_ratio=$6
seed=$7

mkdir -p "$output_dir"

echo "Model path：$model_path"
echo "Save output directory：$output_dir"

echo $base_training_args
nvidia-smi


LOG_FILE="$output_dir/train.log"
bash -c "\
  $header $base_training_args \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --train_files $train_files \
    --percentage $sample_ratio \
    --seed $seed \
  2>&1 | tee \"$LOG_FILE\"" &
child_pid=$!

echo "🚀 Training started (PID=$child_pid), logs written to $LOG_FILE"
wait "$child_pid"
exit_code=$?

echo "Training process (PID=$child_pid) has exited with code $exit_code"
exit $exit_code