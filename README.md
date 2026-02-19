# World Models A3C

## Quick Start

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    bash run_smoke.sh

For full training/retraining flows, see `docs/RUNBOOK.ko.md`.

## Implementation of a variant of World Models

![](/assets/world-models.png)

## Note
- Replaced MDN-RNN to LSTM for Memory
- Replaced CMA-ES to A3C for Controller
- Trained over two stages
    - Stage 1: V and M were trained on dataset with random rollout
    - Stage 2: V and M were trained on dataset with a3c rollout


## Training Result

<b>Result with dataset using random rollout</b>

<!-- ![](/assets/scores.png) -->
<p><img src="/assets/scores.png" width="400"></p>

<b>Result with dataset using the pretrained model rollout</b>

<p><img src="/assets/scores-additional.png" width="400"></p>
<!-- ![](/assets/scores-additional.png) -->

<b>Play Demo</b>

<p><img src="/assets/a3c.gif" width="400"></p>
<!-- ![](/assets/a3c.gif) -->


---

## Documentation

- Korean runbook (recommended): `docs/RUNBOOK.ko.md`

## Environment Setting
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

## Training Stage I
### Dataset Generation using Rollout with random policy
    python rollout.py

### Vision model with VAE
    python train-vae.py

### Memory model with LSTM-RNN
    python train-rnn.py

### Controller with A3C
    python train-a3c.py


## Training Stage II
### Rollout with the pretrained model
    python rollout-a3c.py

### Fine-tuning V and M with new dataset
    vi hparams.py
        extra = True

    python train-vae.py
    python train-rnn.py

### Train new C with the improved V and M
    python train-a3c.py

## Test
    # <# of plays> <seed> <is_record>
    python test.py 2 999 False


## Reproducible Run (Docker)
Build image:

    docker build -t worldmodels-a3c .

Run end-to-end smoke training + test:

    docker run --rm -it -v $(pwd):/workspace worldmodels-a3c bash run_smoke.sh

The smoke script intentionally uses very small steps so you can verify the full pipeline works:

- rollout -> train-vae -> train-rnn -> train-a3c -> test

If you want to increase training time, pass environment variables:

    docker run --rm -it \
      -e HP_N_ROLLOUT=20 \
      -e VAE_MAX_STEP=2000 \
      -e RNN_MAX_STEP=2000 \
      -e HP_MAX_EP=20 \
      -v $(pwd):/workspace worldmodels-a3c bash run_smoke.sh

## Reproducible Run (venv)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    bash run_smoke.sh

## Long Training Run

    source .venv/bin/activate
    bash run_long.sh

To override defaults:

    HP_N_ROLLOUT=400 VAE_MAX_STEP=500000 RNN_MAX_STEP=200000 HP_MAX_EP=2000 bash run_long.sh

## Stage II Long Training Run

    source .venv/bin/activate
    bash run_stage2_long.sh

This script assumes Stage I checkpoints already exist in `ckpt/`.

To override defaults:

    ROLLOUT_A3C_TEST_EP=500 VAE_MAX_STEP=500000 RNN_MAX_STEP=200000 HP_MAX_EP=2000 bash run_stage2_long.sh


## Reference
- https://arxiv.org/abs/1803.10122
- https://github.com/ctallec/world-models
