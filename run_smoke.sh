#!/usr/bin/env bash
set -euo pipefail

export HP_N_ROLLOUT="${HP_N_ROLLOUT:-2}"
export HP_SEQ_LEN="${HP_SEQ_LEN:-32}"
export VAE_MAX_STEP="${VAE_MAX_STEP:-2}"
export VAE_LOG_INTERVAL="${VAE_LOG_INTERVAL:-1}"
export VAE_SAVE_INTERVAL="${VAE_SAVE_INTERVAL:-1}"
export RNN_MAX_STEP="${RNN_MAX_STEP:-2}"
export RNN_LOG_INTERVAL="${RNN_LOG_INTERVAL:-1}"
export RNN_SAVE_INTERVAL="${RNN_SAVE_INTERVAL:-1}"
export HP_MAX_EP="${HP_MAX_EP:-1}"
export HP_SAVE_START_SCORE="${HP_SAVE_START_SCORE:--1000}"
export HP_SCORE_CUT="${HP_SCORE_CUT:--1000}"
export A3C_N_PROCESSES="${A3C_N_PROCESSES:-1}"
export A3C_UPDATE_TERM="${A3C_UPDATE_TERM:-10}"
export ENABLE_RENDER="${ENABLE_RENDER:-0}"

python rollout.py
python train-vae.py
python train-rnn.py
python train-a3c.py
python test.py 1 999 0
