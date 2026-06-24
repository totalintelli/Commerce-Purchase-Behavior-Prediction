"""
제출 후 받은 대회 리더보드(LB) 점수를 기존 MLflow 런에 기록한다.

워크플로우:
  1) python3 local_cv.py --model personal --mlflow   # 끝에 run_id 출력
  2) 같은 설정으로 만든 제출 파일을 대회에 제출 → LB public 점수 확인
  3) python3 log_lb.py --run_id <위 run_id> --public 0.31
     (대회 종료 후 private 점수가 나오면 --private 로 추가)

그러면 MLflow UI에서 cv/NDCG_10(로컬) 과 lb_public(실제)이 한 런에 나란히 보여
"내 로컬 CV가 LB와 얼마나 잘 맞는지" 대조할 수 있다.

주의: --mlflow_uri 는 local_cv.py 실행 때와 같아야 같은 런을 찾는다(둘 다 기본 ./mlruns
이면 같은 폴더에서 실행하면 됨).
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="local_cv.py --mlflow 가 출력한 run_id")
    ap.add_argument("--public", type=float, default=None, help="LB public 점수")
    ap.add_argument("--private", type=float, default=None, help="LB private 점수(대회 종료 후)")
    ap.add_argument("--note", default=None, help="제출 메모(태그로 기록)")
    ap.add_argument("--mlflow_uri", default=None, help='기본 ./mlruns. local_cv 와 동일하게.')
    args = ap.parse_args()

    if args.public is None and args.private is None:
        ap.error("--public 또는 --private 중 최소 하나는 입력하세요.")

    try:
        import mlflow
    except ImportError:
        raise SystemExit("[log_lb] mlflow 미설치. `pip install mlflow` 후 사용하세요.")

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    # run_id 로 기존 런을 다시 열어 metric/tag 를 덧붙인다.
    with mlflow.start_run(run_id=args.run_id):
        if args.public is not None:
            mlflow.log_metric("lb_public", args.public)
        if args.private is not None:
            mlflow.log_metric("lb_private", args.private)
        if args.note:
            mlflow.set_tag("lb_note", args.note)

    print(f"[log_lb] run {args.run_id} 기록 완료 "
          f"(public={args.public}, private={args.private})")


if __name__ == "__main__":
    main()
