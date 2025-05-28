#!/usr/bin/env bash
source ./MINT_config.sh

echo "Beginning warm-up training for job: ${JOB_NAME_WARM}"
LOG_FILE=$(log_file "$JOB_NAME_WARM")
bash "$WARM_SCRIPT_PATH" \
  "$DATA_FILE" \
  "$MODEL_TRAIN_PATH" \
  "$JOB_NAME_WARM" \
  "$RESULTS_FILE" \
  "$WARM_OUTPUT_PATH" \
  "$SAMPLE_RATIO" \
  "$SEED" \
  > "$LOG_FILE"