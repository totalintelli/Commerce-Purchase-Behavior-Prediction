"""
제출 파일 생성기 (분리형).

local_cv.py 의 추천기를 그대로 재사용해, 선택한 모델로 제출 CSV를 만든다.
local_cv 와의 차이(중요):
  - 학습 데이터: 홀드아웃 없이 **전체** train.parquet (최신 1주까지 모두 사용).
  - 예측 대상: 검증 유저가 아니라 **train의 모든 유저(638,257명)**.
출력: sample_submission 포맷 (헤더 user_id,item_id / 유저당 10행, score 내림차순).

사용:
  python3 make_submission.py --model personal
  python3 make_submission.py --model personal --out ../output/personal.csv

워크플로우: local_cv 로 모델 비교 → 가장 좋은 설정 선택 → 이 스크립트로 제출 파일 생성
→ 제출 후 log_lb.py 로 LB 점수를 해당 CV 런에 기록.

--mlflow_run_id 를 주면 생성한 제출 파일을 그 CV 런에 artifact로 연결한다(어떤 run이 어떤
제출을 만들었는지 추적). 기본은 gzip 압축본을 올린다(원본 ~450MB → 수십 MB). 원본 CSV는
업로드용으로 --out 위치에 그대로 남는다.
"""
import argparse
import gzip
import os
import shutil

import numpy as np
import pandas as pd

from local_cv import RECOMMENDERS, load_events


def build_submission(df, recommender, k=10):
    """전체 데이터·전체 유저로 추천 → {user: [items]} 반환."""
    all_users = df["user_id"].unique().tolist()
    print(f"  추천 생성: 유저 {len(all_users):,}명 × {k}개 ...")
    preds = recommender(df, all_users, k)
    return all_users, preds


def to_frame(all_users, preds, k=10):
    """{user: [items]} → 제출 DataFrame(user_id,item_id). 유저당 정확히 k개·중복 없음 검증."""
    users_col, items_col = [], []
    bad = 0
    for u in all_users:
        items = preds.get(u, [])
        if len(items) != k or len(set(items)) != k:   # 10개·중복없음 보장(채점 조건)
            bad += 1
        users_col.extend([u] * len(items))
        items_col.extend(items)
    if bad:
        raise ValueError(f"{bad}명의 유저가 '중복없는 {k}개' 조건을 위반 → 추천기 점검 필요")
    return pd.DataFrame({"user_id": users_col, "item_id": items_col})


def log_submission_artifact(args, sub):
    """생성한 제출 파일을 기존 MLflow CV 런(--mlflow_run_id)에 artifact로 연결."""
    try:
        import mlflow
    except ImportError:
        print("[mlflow] 미설치 → artifact 연결 생략. `pip install mlflow`.")
        return
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)        # local_cv 와 동일해야 같은 런

    tmp_gz = None
    if args.log_raw:
        art_path = args.out                              # 원본(~450MB) 그대로
    else:
        art_path = tmp_gz = args.out + ".gz"             # gzip 압축본(권장)
        with open(args.out, "rb") as fi, gzip.open(art_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)

    try:
        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_artifact(art_path, artifact_path="submission")
            mlflow.set_tags({                            # 런에서 바로 보이는 메타데이터
                "submission_model": args.model,
                "submission_file": os.path.basename(art_path),
                "submission_rows": len(sub),
                "submission_users": int(sub["user_id"].nunique()),
                "submission_k": args.k,
            })
        print(f"[mlflow] run {args.mlflow_run_id} 에 제출 artifact 연결 완료 "
              f"(submission/{os.path.basename(art_path)})")
    except Exception as e:
        print(f"[mlflow] artifact 연결 실패: {e}")
    finally:
        if tmp_gz and os.path.exists(tmp_gz):            # 업로드용 원본만 남기고 gz 임시본 제거
            os.remove(tmp_gz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="../data/train.parquet")
    ap.add_argument("--model", default="personal", choices=list(RECOMMENDERS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="../output/submission.csv")
    ap.add_argument("--seed", type=int, default=42)
    # MLflow 연결 (선택): 생성한 제출 파일을 해당 CV 런에 artifact로 붙임
    ap.add_argument("--mlflow_run_id", default=None,
                    help="local_cv.py --mlflow 가 출력한 run_id. 주면 제출 파일을 그 런에 연결")
    ap.add_argument("--mlflow_uri", default=None, help="기본 ./mlruns. local_cv 와 동일하게")
    ap.add_argument("--log_raw", action="store_true",
                    help="압축 없이 원본 CSV(~450MB)를 artifact로 올림(기본: gzip)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    print(f"loading {args.data_path} ...")
    df = load_events(args.data_path)
    print(f"  rows={len(df):,}  users={df['user_id'].nunique():,}  model={args.model}")

    all_users, preds = build_submission(df, RECOMMENDERS[args.model], k=args.k)
    sub = to_frame(all_users, preds, k=args.k)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\n저장 완료: {args.out}")
    print(f"  행수={len(sub):,} (= 유저 {sub['user_id'].nunique():,} × {args.k})")

    if args.mlflow_run_id:
        log_submission_artifact(args, sub)


if __name__ == "__main__":
    main()
