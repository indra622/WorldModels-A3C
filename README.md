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


## FAQ

**Q. 이 프로젝트의 최종 목표는 무엇인가요?**

CarRacing-v2 환경에서 어떤 형태의 랜덤 서킷이 주어져도 스스로 주행할 수 있는 자동차 에이전트를 학습시키는 것입니다. 학습 결과는 `test.py`로 시각적으로 확인할 수 있습니다.

```bash
python test.py 2 999 False  # <플레이 횟수> <시드> <녹화 여부>
```

**Q. CarRacing-v2는 미리 만들어진 데이터셋인가요?**

아닙니다. CarRacing-v2는 OpenAI Gym(현 Gymnasium)의 **시뮬레이션 환경**입니다. 별도로 다운로드하는 데이터가 아니라, `rollout.py`를 실행하면 환경을 직접 구동하면서 데이터를 생성합니다. 매 에피소드마다 서킷이 절차적으로 랜덤 생성됩니다.

생성되는 데이터의 구성은 스텝마다 다음과 같습니다:

| 필드 | 내용 | 크기 |
|---|---|---|
| `obs` | 현재 프레임 (RGB) | 96×96×3 |
| `action` | [핸들, 가속, 브레이크] | 3개 연속값 |
| `reward` | 이 스텝의 보상 | 숫자 1개 |
| `next_obs` | 다음 프레임 (RGB) | 96×96×3 |
| `done` | 에피소드 종료 여부 | bool |

**Q. `rollout.py`에서 데이터를 생성할 때 학습된 모델이 주행하나요?**

Stage 1에서는 **아닙니다**. `env.action_space.sample()`로 핸들/가속/브레이크를 완전히 무작위로 입력해 데이터를 수집합니다. 학습된 모델 없이 마구잡이로 달리며 녹화한 것입니다.

Stage 2(`rollout-a3c.py`)에서는 Stage 1에서 학습된 A3C 에이전트가 직접 주행하며 더 질 좋은 데이터를 생성합니다. 이 데이터로 V와 M을 재학습하면 성능이 향상됩니다.

**Q. 에이전트(A3C)는 화면 픽셀을 직접 보고 행동하나요?**

아닙니다. 에이전트는 **픽셀을 전혀 보지 않습니다**. V(VAE)가 96×96 이미지를 128차원 잠재벡터 `z`로 압축하고, M(LSTM)이 과거 상태를 기억한 hidden state `h`를 제공합니다. 에이전트는 오직 `[z, h]`만을 입력으로 받아 행동을 결정합니다.

```
raw 픽셀 → V(VAE) → z (128차원)
[z, action] → M(LSTM) → h (256차원)
[z, h] → A3C → 핸들/가속/브레이크
```

A3C 학습 중 V와 M은 고정(`torch.no_grad()`)되어 있으며, 오직 C(컨트롤러)만 강화학습으로 업데이트됩니다.

## Reference
- https://arxiv.org/abs/1803.10122
- https://github.com/ctallec/world-models
