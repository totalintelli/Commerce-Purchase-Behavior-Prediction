"""
Local CV for Commerce Purchase Behavior Prediction.

공식 평가지표(대회 평가방법 탭) = Binary relevance 기반 NDCG@10.
  - relevance: test set의 해당 user가 예측 item을 구매했으면 1, 아니면 0.
  - DCG  = sum_i rel_i / log2(rank_i + 1)            (rank는 1부터)
  - IDCG = 이상적 배치의 DCG. 표준은 min(|gt|, k)개의 1을 위에서부터 채운 값.
    (평가방법 PDF 예시는 |gt|>=3을 가정해 IDCG@3=1/log2(2)+1/log2(3)+1/log2(4)로 표기 →
     표준 NDCG와 동일. LB 첫 제출과 어긋나면 --idcg_mode full_k 로 전환해 대조.)

test 셋(3/1~3/7 구매)과 동일 구조의 "시간 기반 홀드아웃"으로 검증한다.
  - 컷오프 T 이전 데이터로만 추천 → T 이후 1주간 실제 purchase와 비교.
  - 정답 유저 = 검증 주에 구매했고 & 컷오프 이전 이력이 있는 유저(실제 test 조건과 동일).

추천기(recommender)는 plug-in: recommend(train_df, target_users, k) -> {user_id: [item_id,...]}
(점수 내림차순, 최대 k개, 중복 없음). 새 모델은 동일 시그니처 함수만 추가하면 CV에 그대로 태움.

실험 추적은 선택(--mlflow). 오픈소스/로컬(계정·API 키 불필요, ./mlruns 에 기록, `mlflow ui`로 확인).
미설치여도 CV 자체는 그대로 동작한다.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# 이벤트 가중치(설계 선택). purchase가 가장 강한 구매 의도 신호.
EVENT_WEIGHT = {"view": 1.0, "cart": 3.0, "purchase": 10.0}


# --------------------------------------------------------------------------- #
# 데이터 로딩 / 시간 분할
# --------------------------------------------------------------------------- #
def load_events(path, columns=None):
    """train.parquet 로드 + event_time 파싱(" UTC" 접미사 제거, 모두 UTC라 tz 무시)."""
    if columns is None:
        columns = ["user_id", "item_id", "event_time", "event_type"]
    df = pd.read_parquet(path, columns=columns)
    df["t"] = pd.to_datetime(df["event_time"].str.slice(0, 19))
    df["w"] = df["event_type"].map(EVENT_WEIGHT).fillna(1.0)
    return df


def make_folds(df, val_days=7, n_folds=1):
    """
    마지막 날부터 val_days 길이의 검증창을 n_folds개 만든다(뒤에서 앞으로).
    각 폴드: (train_df = 창 시작 이전, truth = {user: set(items)} 검증창 내 purchase)
    """
    max_day = df["t"].dt.normalize().max()           # 2020-02-29
    end_excl = max_day + pd.Timedelta(days=1)         # 2020-03-01 (배타적 상한)
    folds = []
    for i in range(n_folds):
        val_end = end_excl - pd.Timedelta(days=val_days * i)
        val_start = val_end - pd.Timedelta(days=val_days)

        train_df = df[df["t"] < val_start]
        val = df[(df["t"] >= val_start) & (df["t"] < val_end) & (df["event_type"] == "purchase")]

        # 컷오프 이전 이력이 있는 유저만 정답으로(실제 test 조건: 모든 test 유저는 train에 존재).
        known = set(train_df["user_id"].unique())
        truth = defaultdict(set)
        for u, it in zip(val["user_id"].values, val["item_id"].values):
            if u in known:
                truth[u].add(it)

        folds.append({
            "name": f"{val_start.date()}~{(val_end - pd.Timedelta(days=1)).date()}",
            "train": train_df,
            "truth": dict(truth),
        })
    return folds


# --------------------------------------------------------------------------- #
# 평가 지표 (@k) — 공식 지표 NDCG@10 중심
# --------------------------------------------------------------------------- #
def _dcg_weights(k):
    """위치별 할인 가중치 1/log2(rank+1), rank=1..k."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate(preds, truth, k=10, idcg_mode="standard"):
    """
    정답이 있는 유저 전체에 대한 평균 지표.
    idcg_mode:
      "standard" -> IDCG = top-min(|gt|,k) 위치 합 (표준 NDCG, 권장)
      "full_k"   -> IDCG = top-k 위치 합 (상수; 평가방법 PDF 문자 그대로 해석)
    """
    disc = _dcg_weights(k)
    full_idcg = disc.sum()
    rec = ndcg = ap = hit = 0.0
    n = 0
    for u, gt in truth.items():
        if not gt:
            continue
        n += 1
        p = preds.get(u, [])[:k]
        hits = 0
        dcg = 0.0
        ap_u = 0.0
        for rank, item in enumerate(p):          # rank 0-indexed
            if item in gt:
                hits += 1
                dcg += disc[rank]
                ap_u += hits / (rank + 1)
        if idcg_mode == "full_k":
            idcg = full_idcg
        else:
            idcg = disc[: min(len(gt), k)].sum()
        denom = min(len(gt), k)
        rec += hits / len(gt)
        ndcg += dcg / idcg if idcg > 0 else 0.0
        ap += ap_u / denom if denom else 0.0
        hit += 1.0 if hits > 0 else 0.0
    if n == 0:
        return {"users": 0}
    return {
        "users": n,
        f"NDCG@{k}": ndcg / n,         # 공식 지표
        f"Recall@{k}": rec / n,
        f"MAP@{k}": ap / n,
        f"HitRate@{k}": hit / n,
    }


# --------------------------------------------------------------------------- #
# 베이스라인 추천기 (plug-in). 새 모델은 이 시그니처만 맞추면 됨.
# --------------------------------------------------------------------------- #
def _global_top(train_df, k, decay_days=None):
    """전역 인기 아이템 top-k (이벤트 가중 + 선택적 시간감쇠)."""
    w = train_df["w"].values
    if decay_days:
        tmax = train_df["t"].max()
        age = (tmax - train_df["t"]).dt.total_seconds().values / 86400.0
        w = w * np.exp(-age / decay_days)
    s = pd.Series(w, index=train_df["item_id"].values).groupby(level=0).sum()
    return s.sort_values(ascending=False).index[:k].tolist()


def recommend_popularity(train_df, target_users, k=10, decay_days=None):
    """모든 유저에게 동일한 전역 인기 top-k."""
    top = _global_top(train_df, k, decay_days)
    return {u: top for u in target_users}


def recommend_personal(train_df, target_users, k=10, decay_days=14):
    """유저가 과거에 상호작용한 아이템을 가중·시간감쇠로 재랭킹 + 부족분은 인기로 채움."""
    targets = set(target_users)
    sub = train_df[train_df["user_id"].isin(targets)].copy()
    tmax = train_df["t"].max()
    w = sub["w"].values
    if decay_days:
        age = (tmax - sub["t"]).dt.total_seconds().values / 86400.0
        w = w * np.exp(-age / decay_days)
    sub["score"] = w
    g = sub.groupby(["user_id", "item_id"])["score"].sum().reset_index()
    g = g.sort_values("score", ascending=False)

    fallback = _global_top(train_df, k, decay_days)
    preds = {}
    for u, grp in g.groupby("user_id", sort=False):
        items = grp["item_id"].tolist()[:k]
        if len(items) < k:
            for it in fallback:
                if it not in items:
                    items.append(it)
                if len(items) == k:
                    break
        preds[u] = items
    for u in targets:                       # 이력이 전혀 없는 타깃은 인기로
        preds.setdefault(u, fallback)
    return preds


RECOMMENDERS = {
    "popularity": recommend_popularity,
    "popularity_decay": lambda tr, tu, k=10: recommend_popularity(tr, tu, k, decay_days=14),
    "personal": recommend_personal,
}


# --------------------------------------------------------------------------- #
# 실험 추적 (MLflow, 오픈소스/로컬) — 선택. 미설치/실패해도 CV는 계속.
# --------------------------------------------------------------------------- #
class MLflowTracker:
    def __init__(self, mlflow, run_id):
        self.mlflow = mlflow
        self.run_id = run_id

    @staticmethod
    def _key(k):
        return k.replace("@", "_")           # mlflow metric 키는 '@' 미허용

    def log(self, metrics, step=None):
        self.mlflow.log_metrics(
            {self._key(k): float(v) for k, v in metrics.items()
             if isinstance(v, (int, float))},
            step=step,
        )

    def finish(self):
        self.mlflow.end_run()
        # 제출 후 이 run_id로 LB 점수를 같은 런에 기록 (CV vs LB 대조)
        print(f"\n[mlflow] run_id={self.run_id}")
        print(f"[mlflow] 제출 후 LB 점수 기록: "
              f"python3 log_lb.py --run_id {self.run_id} --public <점수>")


def maybe_init_tracker(args):
    """--mlflow 일 때만 초기화. 미설치/실패 시 경고 후 None 반환(CV는 계속)."""
    if not args.mlflow:
        return None
    try:
        import mlflow
    except ImportError:
        print("[mlflow] 미설치 → 로깅 생략. `pip install mlflow` 후 사용하세요.")
        return None
    try:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)   # 기본: ./mlruns
        mlflow.set_experiment(args.mlflow_experiment)
        run = mlflow.start_run(run_name=args.mlflow_run
                               or f"{args.model}_vd{args.val_days}_nf{args.n_folds}")
        mlflow.log_params({
            "model": args.model, "k": args.k, "val_days": args.val_days,
            "n_folds": args.n_folds, "idcg_mode": args.idcg_mode,
            "event_weight": str(EVENT_WEIGHT), "seed": args.seed,
        })
        print(f"[mlflow] run started: {run.info.run_id} "
              f"(experiment={args.mlflow_experiment})")
        return MLflowTracker(mlflow, run.info.run_id)
    except Exception as e:
        print(f"[mlflow] init 실패 → 로깅 생략: {e}")
        return None


# --------------------------------------------------------------------------- #
# 러너
# --------------------------------------------------------------------------- #
def run_cv(df, recommender, k=10, val_days=7, n_folds=1, idcg_mode="standard", tracker=None):
    folds = make_folds(df, val_days=val_days, n_folds=n_folds)
    rows = []
    for i, f in enumerate(folds):
        targets = list(f["truth"].keys())
        if not targets:
            print(f"[fold {f['name']}] 정답 유저 0명 → 건너뜀")
            continue
        preds = recommender(f["train"], targets, k)
        m = evaluate(preds, f["truth"], k=k, idcg_mode=idcg_mode)
        m["fold"] = f["name"]
        rows.append(m)
        print(f"[fold {f['name']}] users={m['users']:>5}  "
              f"NDCG@{k}={m[f'NDCG@{k}']:.4f}  Recall@{k}={m[f'Recall@{k}']:.4f}  "
              f"MAP@{k}={m[f'MAP@{k}']:.4f}  HitRate@{k}={m[f'HitRate@{k}']:.4f}")
        if tracker:
            tracker.log({f"fold/{c}": m[c] for c in m if c != "fold"}, step=i)

    if not rows:
        return rows
    keys = [c for c in rows[0] if c not in ("fold", "users")]
    avg = {c: float(np.mean([r[c] for r in rows])) for c in keys}
    if len(rows) > 1:
        print("[mean] " + "  ".join(f"{c}={avg[c]:.4f}" for c in keys))
    if tracker:                              # CV 평균(런 비교 기준)
        tracker.log({f"cv/{c}": avg[c] for c in keys})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="../data/train.parquet")
    ap.add_argument("--model", default="popularity", choices=list(RECOMMENDERS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--val_days", type=int, default=7)
    ap.add_argument("--n_folds", type=int, default=1)
    ap.add_argument("--idcg_mode", default="standard", choices=["standard", "full_k"])
    ap.add_argument("--seed", type=int, default=42)
    # 실험 추적 (MLflow)
    ap.add_argument("--mlflow", action="store_true", help="MLflow 로깅 활성화(오픈소스/로컬)")
    ap.add_argument("--mlflow_experiment", default="commerce-purchase-prediction")
    ap.add_argument("--mlflow_run", default=None)
    ap.add_argument("--mlflow_uri", default=None, help='예: "file:../mlruns" (기본: ./mlruns)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    tracker = maybe_init_tracker(args)

    print(f"loading {args.data_path} ...")
    df = load_events(args.data_path)
    print(f"  rows={len(df):,}  users={df['user_id'].nunique():,}  "
          f"range={df['t'].min().date()}~{df['t'].max().date()}")
    print(f"model={args.model}  k={args.k}  val_days={args.val_days}  "
          f"n_folds={args.n_folds}  idcg_mode={args.idcg_mode}\n")

    run_cv(df, RECOMMENDERS[args.model], k=args.k, val_days=args.val_days,
           n_folds=args.n_folds, idcg_mode=args.idcg_mode, tracker=tracker)

    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()
