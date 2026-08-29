# Commerce Purchase Prediction

사용자 이벤트 로그를 기반으로 구매할 상품을 추천하는 프로젝트입니다.  
시간 기반 로컬 검증(Local CV), 제출 파일 생성, MLflow 실험 추적, ALS 모델 학습을 지원합니다.

---

## 프로젝트 구성

```text
.
├── local_cv.py          # 시간 기반 Local CV 및 기본 추천 모델
├── make_submission.py   # 전체 학습 데이터로 제출 CSV 생성
├── log_lb.py            # 대회 리더보드 점수를 MLflow run에 기록
├── train_als.py         # implicit ALS 기반 추천 모델 학습 및 제출 생성
├── requirements.txt     # 패키지 목록
├── data/
│   └── train.parquet
└── output/
    └── submission.csv
```

> 제공된 파일명이 `requirements_1_.txt`라면, 필요에 따라 `requirements.txt`로 변경해서 사용하세요.

---

## 설치

Python 3 환경에서 실행합니다.

```bash
pip install -r requirements.txt
```

MLflow 실험 추적을 사용하려면 별도로 설치합니다.

```bash
pip install mlflow
```

---

## 데이터 형식

기본 입력 파일 경로는 다음과 같습니다.

```text
../data/train.parquet
```

학습 데이터에는 최소한 아래 컬럼이 필요합니다.

| 컬럼 | 설명 |
|---|---|
| `user_id` | 사용자 ID |
| `item_id` | 상품 ID |
| `event_time` | 이벤트 발생 시간 |
| `event_type` | 이벤트 유형 |

`event_type`은 다음과 같은 값을 사용합니다.

```text
view
cart
purchase
```

기본 이벤트 가중치는 다음과 같습니다.

| 이벤트 | 가중치 |
|---|---:|
| `view` | 1.0 |
| `cart` | 3.0 |
| `purchase` | 10.0 |

---

## 1. Local CV

`local_cv.py`는 마지막 기간을 검증 데이터로 사용하는 시간 기반 홀드아웃 검증을 수행합니다.

- 검증 기간 이전의 데이터로 추천 모델을 구성합니다.
- 검증 기간의 `purchase` 이벤트를 정답으로 사용합니다.
- 공식 지표를 기준으로 `NDCG@10`을 계산합니다.
- 추가로 `Recall@10`, `MAP@10`, `HitRate@10`을 출력합니다.

### 기본 실행

```bash
python3 local_cv.py
```

기본 모델은 `popularity`입니다.

```bash
python3 local_cv.py --model popularity
```

### 지원 모델

| 모델 | 설명 |
|---|---|
| `popularity` | 전체 이벤트 가중치 기준 인기 상품 추천 |
| `popularity_decay` | 최근 이벤트에 더 큰 비중을 주는 인기 상품 추천 |
| `personal` | 사용자 과거 상호작용 상품을 개인화 재정렬하고 인기 상품으로 보완 |

### 개인화 모델 검증

```bash
python3 local_cv.py --model personal
```

### 최근 1주일을 검증 데이터로 사용

```bash
python3 local_cv.py \
  --model personal \
  --val_days 7 \
  --n_folds 1
```

### 여러 폴드로 검증

마지막 주부터 과거 방향으로 여러 개의 검증 폴드를 생성합니다.

```bash
python3 local_cv.py \
  --model personal \
  --val_days 7 \
  --n_folds 3
```

### 데이터 경로 지정

```bash
python3 local_cv.py \
  --data_path ../data/train.parquet \
  --model personal
```

### IDCG 계산 방식

기본값은 표준 NDCG 방식입니다.

```bash
python3 local_cv.py --idcg_mode standard
```

대회 평가 방식과의 차이를 비교해야 하는 경우 `full_k`를 사용할 수 있습니다.

```bash
python3 local_cv.py --idcg_mode full_k
```

---

## 2. 제출 파일 생성

`make_submission.py`는 Local CV와 달리 학습 데이터를 분리하지 않습니다.

- 전체 `train.parquet` 데이터를 사용합니다.
- 학습 데이터에 존재하는 모든 사용자에 대해 추천합니다.
- 사용자마다 중복 없는 `k`개의 상품을 생성합니다.
- 기본적으로 사용자당 10개 상품을 추천합니다.
- 출력 컬럼은 `user_id`, `item_id`입니다.

### 기본 제출 파일 생성

```bash
python3 make_submission.py --model personal
```

기본 출력 경로는 다음과 같습니다.

```text
../output/submission.csv
```

### 출력 파일 경로 지정

```bash
python3 make_submission.py \
  --model personal \
  --out ../output/personal_submission.csv
```

### 추천 상품 수 변경

```bash
python3 make_submission.py \
  --model personal \
  --k 10
```

### 인기 상품 모델로 제출 생성

```bash
python3 make_submission.py \
  --model popularity \
  --out ../output/popularity_submission.csv
```

### 제출 파일 형식

```csv
user_id,item_id
10001,20011
10001,30012
10001,40013
...
```

각 사용자는 정확히 `k`개의 추천 상품을 가져야 하며, 동일 사용자 내 추천 상품은 중복되지 않습니다.

---

## 3. MLflow 실험 추적

MLflow를 사용하면 Local CV 결과와 실제 리더보드 점수를 하나의 실험 단위로 관리할 수 있습니다.

### Local CV 결과 기록

```bash
python3 local_cv.py \
  --model personal \
  --mlflow
```

실행 후 출력되는 `run_id`를 저장해 두세요.

```text
[mlflow] run_id=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 실행 이름 지정

```bash
python3 local_cv.py \
  --model personal \
  --mlflow \
  --mlflow_run exp-001-personal
```

### Experiment 이름 지정

```bash
python3 local_cv.py \
  --model personal \
  --mlflow \
  --mlflow_experiment commerce-purchase-prediction
```

### MLflow UI 실행

기본 로컬 저장소를 사용하는 경우 다음 명령으로 UI를 실행할 수 있습니다.

```bash
mlflow ui
```

브라우저에서 아래 주소에 접속합니다.

```text
http://127.0.0.1:5000
```

### MLflow 저장 경로 지정

```bash
python3 local_cv.py \
  --model personal \
  --mlflow \
  --mlflow_uri "file:../mlruns"
```

`local_cv.py`, `make_submission.py`, `log_lb.py`에서 동일한 MLflow URI를 사용해야 같은 실험 기록을 조회할 수 있습니다.

---

## 4. 제출 파일을 MLflow Run에 연결

Local CV를 수행한 MLflow run에 제출 파일을 artifact로 연결할 수 있습니다.

```bash
python3 make_submission.py \
  --model personal \
  --out ../output/personal.csv \
  --mlflow_run_id <RUN_ID>
```

MLflow URI를 지정한 경우:

```bash
python3 make_submission.py \
  --model personal \
  --out ../output/personal.csv \
  --mlflow_run_id <RUN_ID> \
  --mlflow_uri "file:../mlruns"
```

기본적으로 제출 CSV는 gzip으로 압축된 뒤 MLflow artifact에 업로드됩니다.

원본 CSV를 그대로 artifact로 기록하려면 다음 옵션을 사용합니다.

```bash
python3 make_submission.py \
  --model personal \
  --mlflow_run_id <RUN_ID> \
  --log_raw
```

---

## 5. 리더보드 점수 기록

제출 후 대회 리더보드 점수를 확인하면 `log_lb.py`를 통해 기존 MLflow run에 기록할 수 있습니다.

### Run ID로 public score 기록

```bash
python3 log_lb.py \
  --run_id <RUN_ID> \
  --public 0.31
```

### Run 이름으로 public score 기록

동일한 이름의 run이 여러 개라면 가장 최근 run을 사용합니다.

```bash
python3 log_lb.py \
  --run_name exp-001-personal \
  --public 0.31
```

### Public / Private 점수 모두 기록

```bash
python3 log_lb.py \
  --run_id <RUN_ID> \
  --public 0.31 \
  --private 0.32
```

### 제출 메모 추가

```bash
python3 log_lb.py \
  --run_id <RUN_ID> \
  --public 0.31 \
  --note "personal model, decay 14 days"
```

### 최근 MLflow Run 목록 조회

```bash
python3 log_lb.py --list
```

---

## 6. ALS 모델 학습

`train_als.py`는 `implicit` 라이브러리의 Alternating Least Squares(ALS) 모델을 사용합니다.

현재 구현은 모든 이벤트에 동일한 값 `1`을 부여하고, 사용자-상품 상호작용 횟수를 confidence 값으로 사용합니다.

### 기본 실행

```bash
python3 train_als.py
```

기본 입력 및 출력 경로는 다음과 같습니다.

```text
입력:  ../data/train.parquet
출력:  ../output/output.csv
```

### Factor 수 변경

```bash
python3 train_als.py --num_factor 64
```

### Regularization 및 Alpha 변경

```bash
python3 train_als.py \
  --num_factor 64 \
  --regularization 0.001 \
  --alpha 10
```

### 데이터 및 출력 경로 변경

```bash
python3 train_als.py \
  --dir_path ../data/ \
  --data_dir train.parquet \
  --output_dir ../output/
```

### ALS 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---:|---|
| `--num_factor` | 32 | 잠재 요인(latent factor) 수 |
| `--regularization` | 0.001 | 정규화 계수 |
| `--alpha` | 10 | 암묵적 피드백 confidence 강도 |
| `--seed` | 42 | 랜덤 시드 |

---

## 권장 실험 워크플로우

### 1) 모델별 Local CV 비교

```bash
python3 local_cv.py --model popularity --mlflow --mlflow_run popularity-baseline
python3 local_cv.py --model popularity_decay --mlflow --mlflow_run popularity-decay
python3 local_cv.py --model personal --mlflow --mlflow_run personal-baseline
```

### 2) 가장 좋은 모델로 제출 파일 생성

```bash
python3 make_submission.py \
  --model personal \
  --out ../output/personal_submission.csv \
  --mlflow_run_id <RUN_ID>
```

### 3) 대회 제출

생성된 CSV 파일을 대회 플랫폼에 제출합니다.

```text
../output/personal_submission.csv
```

### 4) 리더보드 점수 기록

```bash
python3 log_lb.py \
  --run_id <RUN_ID> \
  --public <PUBLIC_SCORE>
```

### 5) MLflow에서 Local CV와 LB 결과 비교

```bash
mlflow ui
```

MLflow UI에서 다음 값을 비교합니다.

- `cv/NDCG_10`
- `cv/Recall_10`
- `cv/MAP_10`
- `cv/HitRate_10`
- `lb_public`
- `lb_private`

---

## 참고 사항

- `local_cv.py`의 기본 추천기는 빠른 베이스라인 및 실험용입니다.
- `personal` 모델은 사용자의 과거 상호작용 상품을 시간 감쇠와 이벤트 가중치로 재정렬합니다.
- `purchase` 이벤트는 `view`, `cart`보다 높은 가중치를 가집니다.
- 실제 제출 생성 시에는 검증 홀드아웃 없이 전체 학습 데이터를 사용합니다.
- ALS 실행을 위해서는 `train_als.py`가 import하는 `utils.py` 파일이 프로젝트에 필요합니다.
- 제출 조건상 사용자별 추천 상품 수와 중복 여부를 반드시 확인해야 합니다.
