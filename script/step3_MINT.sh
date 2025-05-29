#!/usr/bin/env bash
source ./MINT_config.sh
python "${MINT_PYTHON_PATH}" \
    --grad-dir "${GRADIENT_PATH}" \
    --proj-dim "${DIM}" \
    --ratios ${RATIOS_LIST} \
    --preheat-ratio "${PREHEAT_RATIO}" \
    --save "${SAVE_DIR}" \
    --seed "${SEED}" \
    ${ALPHA_OPT:-}
