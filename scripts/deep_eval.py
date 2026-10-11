"""Measure the deep recognition engine's real accuracy on your own labels.

Retrains the full engine per fold (every gate, calibration and the
meta-classifier included) with a share of the persons held out entirely as
"strangers", and reports precision / coverage / false-accept rate overall and
by face size and quality.  Also prints the precision measured from your
review decisions (confirmed vs corrected / reverted automatic assignments).

Read-only: the database and the saved model are never modified.

Usage::

    python -m scripts.deep_eval
    python -m scripts.deep_eval --config config.yaml --folds 3 --holdout 0.2
    python -m scripts.deep_eval --json test-reports/deep_eval.json

Run it before and after changing a ``deep_recognition`` setting to see
whether the change really helps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pct(value) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _print_bucket(name: str, data: dict) -> None:
    print(
        f"  {name:<18} known={data['n_known']:>5}  coverage={_pct(data['coverage'])}"
        f"  known-precision={_pct(data['known_precision'])}"
        f"  strangers={data['n_strangers']:>5}  false-accept={_pct(data['false_accept_rate'])}"
        f"  precision={_pct(data['precision'])}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--db", default=None, help="database path override")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--holdout", type=float, default=None,
                        help="share of persons held out as strangers")
    parser.add_argument("--unknown-ratio", type=float, default=None,
                        help="assumed share of strangers among unknown faces")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="also write the report here")
    args = parser.parse_args(argv)

    from app.config import load_config
    from app.db.database import init_db, session_scope
    from app.deep.dataset import build_training_dataset
    from app.deep.evaluation import evaluate_open_set, feedback_precision
    from app.services.embedding_service import embedder_identity

    config = load_config(args.config)
    deep = config.deep_recognition
    init_db(args.db or str(config.db_path_resolved))

    with session_scope() as session:
        dataset = build_training_dataset(
            session,
            auto_min_confidence=deep.auto_training_min_confidence,
            use_auto_assignments=deep.use_auto_assignments_for_training,
            exclude_low_quality=deep.strict_quality_filter,
            embedder_id=embedder_identity(config),
            collect_unknown=deep.use_unknown_class,
            unknown_max_similarity=deep.unknown_class_max_similarity,
            max_unknown=deep.max_unknown_examples,
        )
        reviewed_precision, n_reviewed = feedback_precision(session, deep.feedback_window)
        # Detach everything the evaluation needs from the session.
        session.expunge_all()

    print(
        f"Dataset: {dataset.n_examples} labeled face(s) of {dataset.n_persons} "
        f"person(s), {dataset.n_negatives} hard negative(s), "
        f"{dataset.n_unknown} unknown background face(s)"
    )
    report = evaluate_open_set(
        dataset, deep,
        n_folds=args.folds,
        holdout_fraction=(
            args.holdout if args.holdout is not None else deep.evaluation_holdout_fraction
        ),
        unknown_ratio=(
            args.unknown_ratio if args.unknown_ratio is not None
            else deep.expected_unknown_ratio
        ),
        seed=args.seed,
        progress_cb=lambda cur, tot, detail: print(f"  … {detail} ({cur}/{tot})"),
    )
    data = report.to_dict()
    data["reviewed_precision"] = reviewed_precision
    data["n_reviewed"] = n_reviewed

    print(
        f"\nOpen-set evaluation: {data['n_folds']} fold(s), "
        f"{data['n_known_persons']} known / {data['n_holdout_persons']} held-out person(s), "
        f"assumed stranger share {data['unknown_ratio']:.0%}"
    )
    _print_bucket("overall", data["overall"])
    print("By face size:")
    for name, bucket in sorted(data["by_size"].items()):
        _print_bucket(name, bucket)
    print("By quality:")
    for name, bucket in sorted(data["by_quality"].items()):
        _print_bucket(name, bucket)
    if data["rejection_reasons"]:
        print("Why known faces stayed unassigned:")
        for reason, count in sorted(data["rejection_reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<18} {count}")
    print(
        f"\nPrecision measured from your reviews: {_pct(reviewed_precision)} "
        f"over {n_reviewed} decision(s) (target {deep.target_precision:.0%})"
    )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
