#!/usr/bin/env bash
set -euo pipefail

export ENABLE_RENDER="${ENABLE_RENDER:-0}"
export ROLLOUT_A3C_TEST_EP="${ROLLOUT_A3C_TEST_EP:-300}"
export HP_EXTRA_DIR="${HP_EXTRA_DIR:-additional}"
export VAE_EXTRA_DIR="${VAE_EXTRA_DIR:-additional}"
export RNN_EXTRA_DIR="${RNN_EXTRA_DIR:-additional}"

export VAE_EXTRA=1
export RNN_EXTRA=1

export VAE_MAX_STEP="${VAE_MAX_STEP:-200000}"
export VAE_LOG_INTERVAL="${VAE_LOG_INTERVAL:-1000}"
export VAE_SAVE_INTERVAL="${VAE_SAVE_INTERVAL:-5000}"
export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-64}"

export RNN_MAX_STEP="${RNN_MAX_STEP:-100000}"
export RNN_LOG_INTERVAL="${RNN_LOG_INTERVAL:-1000}"
export RNN_SAVE_INTERVAL="${RNN_SAVE_INTERVAL:-5000}"
export RNN_SEQ_LEN="${RNN_SEQ_LEN:-32}"

export HP_MAX_EP="${HP_MAX_EP:-1000}"
export HP_SAVE_START_SCORE="${HP_SAVE_START_SCORE:-100}"
export HP_SCORE_CUT="${HP_SCORE_CUT:-300}"
export A3C_N_PROCESSES="${A3C_N_PROCESSES:-3}"
export A3C_UPDATE_TERM="${A3C_UPDATE_TERM:-100}"

echo "[1/4] Generating additional rollout data with pretrained controller"
python rollout-a3c.py

echo "[2/4] Fine-tuning VAE on additional data"
python train-vae.py

echo "[3/4] Fine-tuning RNN on additional data"
python train-rnn.py

echo "[4/4] Retraining A3C controller with updated VAE/RNN"
python train-a3c.py

echo "Stage II long training complete. Run evaluation with: python test.py 5 999 0"
