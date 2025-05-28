#!/usr/bin/env bash
source ./MINT_config.sh
echo "Beginning warm-up training for job: ${JOB_NAME_FINAL}"
LOG_FILE=$(log_file "$JOB_NAME_FINAL")
bash "$FINAL_SCRIPT_PATH" \
  "$DATA_FILE" \
  "$MODEL_TRAIN_PATH" \
  "$JOB_NAME_FINAL" \
  "$RESULTS_FILE" \
  "$CORESET_FILE" \
  "$OUTPUT_PATH_FINAL" \
  "$SEED" \
  > "$LOG_FILE"