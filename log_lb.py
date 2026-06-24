"""
제출 후 받은 대회 리더보드(LB) 점수를 기존 MLflow 런에 기록한다.

run 지정 방법(둘 중 하나):
  --run_id <해시>     : local_cv.py --mlflow 가 출력하는 32자리 run_id (정확)
  --run_name <이름>   : 런 이름으로 찾기(같은 이름이면 가장 최근). 예: popularity_vd7_nf1
                        (local_cv.py --mlflow --mlflow_run <이름> 으로 이름을 직접 지정 가능)
run_id를 모르면 먼저: python3 log_lb.py --list

워크플로우:
  1) python3 local_cv.py --model personal --mlflow [--mlflow_run exp-001-personal]
  2) 같은 설정으로 만든 제출 파일을 대회에 제출 → LB public 점수 확인
  3) python3 log_lb.py --run_name exp-001-personal --public 0.31
그러면 MLflow UI에서 cv/NDCG_10(로컬)과 lb_public(실제)이 한 런에 나란히 보인다.

주의: tracking 저장소(MLFLOW_TRACKING_URI 또는 --mlflow_uri)는 local_cv.py 와 같아야 한다.
"""
import argparse


def _get_mlflow(args):
    try:
        import mlflow
    except ImportError:
        raise SystemExit("[log_lb] mlflow 미설치. `pip install mlflow` 후 사용하세요.")
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    return mlflow


def list_runs(mlflow, args):
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"experiment '{args.experiment}' 없음. 먼저 local_cv.py --mlflow 실행.")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                            order_by=["attribute.start_time DESC"], max_results=30)
    if len(df) == 0:
        print("런이 없습니다. local_cv.py --mlflow 를 먼저 실행하세요.")
        return
    cols = ["run_id", "tags.mlflow.runName", "metrics.cv/NDCG_10",
            "metrics.lb_public", "metrics.lb_private"]
    have = [c for c in cols if c in df.columns]
    show = df[have].rename(columns={"tags.mlflow.runName": "name",
                                    "metrics.cv/NDCG_10": "cv_NDCG10"})
    print(show.to_string(index=False))


def resolve_run_id(mlflow, args):
    if args.run_id:
        return args.run_id
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"experiment '{args.experiment}' 없음. 먼저 local_cv.py --mlflow 실행.")
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f'tags.`mlflow.runName` = "{args.run_name}"',
        order_by=["attribute.start_time DESC"], max_results=1)
    if len(df) == 0:
        raise SystemExit(
            f"run_name '{args.run_name}' 인 런을 못 찾음.\n"
            f"  - 이름 확인: python3 log_lb.py --list\n"
            f"  - 또는 먼저: python3 local_cv.py --model <m> --mlflow --mlflow_run {args.run_name}")
    return df.iloc[0]["run_id"]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run_id", default=None, help="local_cv.py --mlflow 가 출력한 32자리 run_id")
    g.add_argument("--run_name", default=None, help="런 이름(같은 이름이면 최신)")
    ap.add_argument("--public", type=float, default=None, help="LB public 점수")
    ap.add_argument("--private", type=float, default=None, help="LB private 점수(대회 종료 후)")
    ap.add_argument("--note", default=None, help="제출 메모(태그로 기록)")
    ap.add_argument("--experiment", default="commerce-purchase-prediction",
                    help="run_name/--list 조회 대상 experiment")
    ap.add_argument("--mlflow_uri", default=None, help="local_cv 와 동일하게(기본: 환경 설정)")
    ap.add_argument("--list", action="store_true", help="최근 런 목록(run_id/이름/점수) 출력 후 종료")
    args = ap.parse_args()

    mlflow = _get_mlflow(args)

    if args.list:
        list_runs(mlflow, args)
        return

    if not args.run_id and not args.run_name:
        ap.error("--run_id 또는 --run_name 중 하나를 지정하세요(모르면 --list).")
    if args.public is None and args.private is None:
        ap.error("--public 또는 --private 중 최소 하나는 입력하세요.")

    run_id = resolve_run_id(mlflow, args)
    with mlflow.start_run(run_id=run_id):       # 기존 런을 다시 열어 metric/tag 덧붙임
        if args.public is not None:
            mlflow.log_metric("lb_public", args.public)
        if args.private is not None:
            mlflow.log_metric("lb_private", args.private)
        if args.note:
            mlflow.set_tag("lb_note", args.note)

    print(f"[log_lb] run {run_id} 기록 완료 (public={args.public}, private={args.private})")


if __name__ == "__main__":
    main()
