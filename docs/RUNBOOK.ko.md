# WorldModels-A3C 실행/학습 런북

이 문서는 현재 저장소 기준으로 실제 동작 확인된 실행 경로를 정리한 가이드입니다.

## 1) 현재 기준 핵심 변경점

- Python 3.11 + `gymnasium[box2d]` 조합으로 실행 경로를 정리했습니다.
- `CarRacing-v3/v2/v1/v0`를 순차 시도하는 호환 레이어(`env_compat.py`)를 추가했습니다.
- Gym/Gymnasium API 차이(`reset`, `step`)를 공통 함수로 흡수했습니다.
- 하이퍼파라미터는 `hparams.py`에서 환경변수로 덮어쓸 수 있게 변경했습니다.
- 실행 스크립트 3종을 추가했습니다.
  - `run_smoke.sh`: 파이프라인 스모크 테스트
  - `run_long.sh`: Stage I 장기 학습
  - `run_stage2_long.sh`: Stage II 추가 데이터 기반 재학습

## 2) 빠른 시작 (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

스모크 실행:

```bash
bash run_smoke.sh
```

## 3) 학습 실행 시나리오

### Stage I 장기 학습

```bash
source .venv/bin/activate
bash run_long.sh
```

오버라이드 예시:

```bash
HP_N_ROLLOUT=400 VAE_MAX_STEP=500000 RNN_MAX_STEP=200000 HP_MAX_EP=2000 bash run_long.sh
```

### Stage II 장기 재학습

전제: `ckpt/`에 Stage I 결과(VAE/RNN/A3C)가 존재해야 합니다.

```bash
source .venv/bin/activate
bash run_stage2_long.sh
```

오버라이드 예시:

```bash
ROLLOUT_A3C_TEST_EP=500 VAE_MAX_STEP=500000 RNN_MAX_STEP=200000 HP_MAX_EP=2000 bash run_stage2_long.sh
```

### 평가/플레이

```bash
python test.py 5 999 0
```

- 인자 형식: `python test.py <n_play> <seed> <is_record>`
- `is_record`를 `1`로 주면 `a3c.gif`를 생성합니다.

## 4) 스크립트별 역할

- `run_smoke.sh`
  - 매우 작은 step으로 전체 경로 점검
  - `rollout -> train-vae -> train-rnn -> train-a3c -> test`
- `run_long.sh`
  - Stage I 실제 학습 러닝
  - 랜덤 정책 rollout 데이터로 VAE/RNN/A3C 학습
- `run_stage2_long.sh`
  - Stage II 실제 학습 러닝
  - 기존 A3C로 추가 rollout(`additional/`) 생성 후 VAE/RNN 파인튜닝 + A3C 재학습

## 5) 자주 조정하는 환경변수

- 공통
  - `ENABLE_RENDER` (기본 `0`)
  - `A3C_N_PROCESSES`, `A3C_UPDATE_TERM`
- Stage I 데이터/에피소드
  - `HP_N_ROLLOUT`, `HP_SEQ_LEN`, `HP_MAX_EP`
- VAE
  - `VAE_MAX_STEP`, `VAE_BATCH_SIZE`, `VAE_LOG_INTERVAL`, `VAE_SAVE_INTERVAL`
- RNN
  - `RNN_MAX_STEP`, `RNN_SEQ_LEN`, `RNN_LOG_INTERVAL`, `RNN_SAVE_INTERVAL`
- Stage II 추가 데이터
  - `ROLLOUT_A3C_TEST_EP`
  - `HP_EXTRA_DIR`, `VAE_EXTRA_DIR`, `RNN_EXTRA_DIR`

## 6) 포크/리모트 운영 방식

권장 리모트 구조:

- `origin`: 내 포크 저장소
- `upstream`: 원본 저장소

확인:

```bash
git remote -v
```

원본 최신 반영:

```bash
git fetch upstream
git merge upstream/master
```

내 포크로 푸시:

```bash
git push origin master
```

## 7) 트러블슈팅

- Docker 데몬이 안 떠 있으면 venv 경로를 우선 사용합니다.
- `test.py`에서 체크포인트를 못 찾으면 Stage I 학습(`run_long.sh`)을 먼저 완료해야 합니다.
- macOS headless 환경에서는 `ENABLE_RENDER=0`으로 실행하세요.
