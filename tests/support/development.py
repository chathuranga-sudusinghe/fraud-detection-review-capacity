"""Valid example records and table rows of the development contracts (no data read)."""

from __future__ import annotations

import copy
from typing import Any

from review_capacity.config import load_dataset_spec
from review_capacity.development.records import (
    CONFIGURATION_NAMES,
    RECORD_CONTRACTS,
    REFERENCE,
    DevelopmentPartitions,
    contract_versions,
)
from support.synthetic import REPO_PATHS

# The development partitions of the real config/dataset.yaml (no data read).
PARTITIONS = DevelopmentPartitions.from_spec(load_dataset_spec(REPO_PATHS), REPO_PATHS)
DIGEST = "a" * 64
V2_ROWS, V2_FRAUDS = 77_822, 2_638
TRAIN_ROWS, TRAIN_FRAUDS = 368_632, 12_500

IDENTITY: dict[str, Any] = {
    "run_id": "2026-10-20T091530Z",
    "run_type": "development",
    "code_commit": "b" * 40,
    "uncommitted_changes": False,
    "source_sha256": "c" * 64,
    "settings_sha256": "d" * 64,
    "declaration_sha256": "e" * 64,
    "dataset_manifest_sha256": "f" * 64,
    "seeds": {"main": 42, "additional": [43, 44], "sampler": 42, "preprocessing": 0},
}


def file(path: str) -> dict[str, str]:
    return {"path": path, "sha256": DIGEST}


def history(best: int, stopped: int, best_value: float = 0.41) -> list[dict[str, Any]]:
    """A trial's checkpoint history: rising to ``best_value`` at ``best``, lower afterwards."""
    rows = []
    for checkpoint in range(1, stopped + 1):
        if checkpoint < best:
            objective = best_value * checkpoint / (best + 1)
        else:
            objective = best_value if checkpoint == best else best_value / 2
        rows.append(
            {
                "checkpoint": checkpoint,
                "objective": objective,
                "improved": checkpoint <= best,
                "best_checkpoint": min(checkpoint, best),
                "seconds": 0.1 * checkpoint,
            }
        )
    return rows


def with_days(record: dict[str, Any], first: int, last: int) -> dict[str, Any]:
    """``record`` (an evaluation) over days ``first``–``last``, with consistent per-day values,
    pooled counts and cutoff ties (a tied group of 3 on the first day if the policy records
    ties). All of V2 (days 127–154) holds its 77,822 transactions."""
    record = copy.deepcopy(record)
    record["days"] = {"first": first, "last": last}
    if record["policy"] == "ranking":
        return record | {"daily": [], "counts": None, "cutoff_ties": None}
    k, tied = record["k"], record["cutoff_ties"] is not None
    selected, frauds, true_positives = (60 if k is None else k), 94, 32
    volume = dict.fromkeys(range(first, last + 1), 2779)
    if (first, last) == (127, 154):
        volume[first] += V2_ROWS - 2779 * 28
    record["daily"] = [
        {
            "day": day,
            "transactions": volume[day],
            "frauds": frauds,
            "selected": selected,
            "true_positives": true_positives,
            "oracle_reviewable": None if k is None else min(k, frauds),
            "overflow": None if k is None else max(selected - k, 0),
            "unused": None if k is None else max(k - selected, 0),
            "load_ratio": None if k is None else selected / k,
            "cutoff_tied_group": 3 if tied and day == first else None,
        }
        for day in range(first, last + 1)
    ]
    days = last - first + 1
    record["counts"] = {
        "days": days,
        "transactions": sum(volume.values()),
        "frauds": frauds * days,
        "selected": selected * days,
        "true_positives": true_positives * days,
        "false_positives": (selected - true_positives) * days,
        "false_negatives": (frauds - true_positives) * days,
    }
    if tied:
        record["cutoff_ties"] = {"days": 1, "group_sizes": [3]}
    return record


def bins(rows: int, frauds: int) -> list[dict[str, Any]]:
    """Two reliability-diagram bins holding ``rows`` transactions and ``frauds`` frauds."""
    low_rows, low_frauds = rows // 2, frauds // 4
    return [
        {
            "bin": number,
            "min_probability": low,
            "max_probability": high,
            "mean_probability": mean,
            "transactions": count,
            "frauds": positives,
            "observed_rate": positives / count,
        }
        for number, low, high, mean, count, positives in (
            (1, 0.0, 0.01, 0.005, low_rows, low_frauds),
            (2, 0.02, 0.9, 0.3, rows - low_rows, frauds - low_frauds),
        )
    ]


def _envelope(kind: str) -> dict[str, Any]:
    return {
        "record": kind,
        "schema_version": RECORD_CONTRACTS[kind].version,
        "identity": copy.deepcopy(IDENTITY),
    }


H3_CAPTURED = {
    "LGB-u": 900,
    "Cat-u": 920,
    "RES-u": 850,
    "FTT-u": 870,
    "LGB-b": 910,
    "Cat-b": 905,
    "RES-b": 860,
    "FTT-b": 880,
}


def _cost_grid() -> list[dict[str, Any]]:
    """The nine-ratio grid of a 2 % cost analysis over V2: R3 reviews 1,568 transactions and
    captures 900 of the V2 frauds (total fraud amount 1,000,000; R3 captures 400,000 of it); R4
    declines 68 reviews (5 frauds, 500 USD) at ratio 2, 8 (1 fraud, 100 USD) at ratio 5 and none
    from ratio 10 on."""
    reviews, frauds, captured = 1568, V2_FRAUDS, 900
    amount, captured_amount = 1_000_000.0, 400_000.0
    declined = {2: (68, 5, 500.0, 0.3), 5: (8, 1, 100.0, 0.1)}
    rows = []
    for ratio in (2, 5, 10, 20, 50, 100, 200, 500, 1000):
        unused, forgone, forgone_amount, mean = declined.get(ratio, (0, 0, 0.0, None))
        cost_r3 = reviews + ratio * (frauds - captured)
        cost_r4 = reviews - unused + ratio * (frauds - captured + forgone)
        rows.append(
            {
                "ratio": float(ratio),
                "threshold": 1 / ratio,
                "reviews": reviews - unused,
                "unused": unused,
                "frauds_forgone": forgone,
                "days_with_unused": 10 if unused else 0,
                "utilisation": (reviews - unused) / reviews,
                "declined_fraud_rate": forgone / unused if unused else None,
                "declined_mean_probability": mean,
                "cost_r3": float(cost_r3),
                "cost_r4": float(cost_r4),
                "cost_difference": float(cost_r4 - cost_r3),
                "relative_difference": (cost_r4 - cost_r3) / cost_r3,
                "cost_no_review": float(ratio * frauds),
                "fixed_status": "finite" if unused else "identical",
                "fixed_ratio": unused / forgone if unused else None,
                "fraud_value_exposed_r3": amount - captured_amount,
                "fraud_value_exposed_r4": amount - captured_amount + forgone_amount,
                "fraud_value_forgone": forgone_amount,
                "forgone_per_unused": forgone_amount / unused if unused else None,
                "amount_weighted_recall_r3": captured_amount / amount,
                "amount_weighted_recall_r4": (captured_amount - forgone_amount) / amount,
            }
        )
    return rows


_BODIES: dict[str, dict[str, Any]] = {
    "candidates": {
        "model": "lightgbm",
        "configurations": ["LGB-u", "LGB-b"],
        "stream": {"index": 0, "spawn_key": [0]},
        "sampler_seed": 42,
        "sampler": {
            "generator": "numpy.random.default_rng",
            "bit_generator": "PCG64",
            "numpy": "2.5.3",
            "python": "3.12.3",
        },
        "trials": [
            {
                "trial": 1,
                "reference": True,
                "parameters": {"learning_rate": 0.1, "num_leaves": 31, "lambda_l2": 0.0},
            },
            {
                "trial": 2,
                "reference": False,
                "parameters": {"learning_rate": 0.0731, "num_leaves": 412, "lambda_l2": 2.5},
            },
            {
                "trial": 3,
                "reference": False,
                "parameters": {"learning_rate": 0.0125, "num_leaves": 9, "lambda_l2": 0.0},
            },
        ],
    },
    "cost_prediction": {
        "candidates": {
            "lightgbm": file("candidates/lightgbm.json"),
            "catboost": file("candidates/catboost.json"),
            "resnet": file("candidates/resnet.json"),
            "ft_transformer": file("candidates/ft_transformer.json"),
            "logistic_regression": file("candidates/logistic_regression.json"),
        },
        "estimates": [
            {
                "model": model,
                "estimated": False,
                "basis": "no cost model in DR-015 §7",
                "epochs_low": None,
                "epochs_high": None,
                "candidate_epoch_minutes": [],
                "search_hours_low": None,
                "search_hours_high": None,
                "refit_hours": None,
            }
            for model in ("lightgbm", "catboost")
        ]
        + [
            {
                "model": model,
                "estimated": True,
                "basis": basis,
                "epochs_low": 7,
                "epochs_high": 15,
                "candidate_epoch_minutes": [minutes, minutes],
                "search_hours_low": low,
                "search_hours_high": high,
                "refit_hours": refit,
            }
            for model, basis, minutes, low, high, refit in (
                ("resnet", "upper bound at the largest measured setting", 0.48, 0.25, 0.5, 0.75),
                ("ft_transformer", "linear in d_token between the anchors", 8.0, 3.5, 7.5, 9.25),
            )
        ]
        + [
            {
                "model": "logistic_regression",
                "estimated": False,
                "basis": "no cost model in DR-015 §7",
                "epochs_low": None,
                "epochs_high": None,
                "candidate_epoch_minutes": [],
                "search_hours_low": None,
                "search_hours_high": None,
                "refit_hours": None,
            }
        ],
        "totals": {"search_hours_low": 3.75, "search_hours_high": 8.0, "refit_hours": 10.0},
        "decision": "pending_approval",
    },
    "run": {
        "started": "2026-10-20T09:15:30.123456Z",
        "code": {
            "package": {
                "name": "review_capacity",
                "version": "0.1.0",
                "location": "/project/src/review_capacity",
                "source_sha256": "c" * 64,
            },
            "git_commit": "b" * 40,
            "git_uncommitted_changes": False,
            "python": "3.12.11",
            "packages": {"numpy": "2.5.3", "lightgbm": None},
        },
        "environment": {"gpu": {"name": "GPU"}, "threads": 24},
        "settings": {"path": "config/models.yaml", "sha256": "d" * 64},
        "declaration": {"path": "config/training.yaml", "sha256": "e" * 64},
        "plan": file("outputs/search_plan/2026-10-20T080000Z/cost_prediction.json"),
        "dataset": {
            "manifest": {"path": "data/datapackage.json", "sha256": "f" * 64},
            "processed": {"train": file("data/processed/train.parquet")},
        },
        "configurations": [*CONFIGURATION_NAMES, REFERENCE],
        "contracts": contract_versions(),
        "source": None,
    },
    "preprocessing": {
        "artifact": file("outputs/preprocessing/2026-10-20T080000Z/manifest.json"),
        "artifact_sha256": DIGEST,
        "seed": 0,
        "train": {"rows": TRAIN_ROWS, "transaction_ids_sha256": DIGEST, "file_sha256": DIGEST},
        "inputs": {
            "numeric": 384,
            "categorical": 49,
            "indicators": 324,
            "total": 757,
            "order_sha256": DIGEST,
        },
        "settings": {"n_quantiles": 1000, "category_dropout": 0.01},
        "software": {"scikit-learn": "1.9.1"},
    },
    "preflight": {
        "model": "ft_transformer",
        "trial": 1,
        "candidate_list": file("candidates.json"),
        "effective_batch": 1024,
        "evaluation_batch": 16,
        "margin_mib": 1024,
        "attempts": [
            {
                "phase": "training",
                "rows": 1024,
                "outcome": "out_of_memory",
                "peak_used_mib": None,
                "free_mib": None,
            },
            {
                "phase": "training",
                "rows": 512,
                "outcome": "below_margin",
                "peak_used_mib": 7500.0,
                "free_mib": 651.0,
            },
            {
                "phase": "training",
                "rows": 256,
                "outcome": "pass",
                "peak_used_mib": 6000.0,
                "free_mib": 2151.0,
            },
            {
                "phase": "evaluation",
                "rows": 16,
                "outcome": "pass",
                "peak_used_mib": 3000.0,
                "free_mib": 5151.0,
            },
        ],
        "micro_batch": 256,
        "status": "pass",
        "error": None,
    },
    "fit": {
        "configuration": "FTT-b",
        "purpose": "search_trial",
        "trial": 3,
        "seed": 42,
        "device": "gpu",
        "fresh_process": False,
        "early_stopping": True,
        "length": None,
        "fitted_iterations": 7,
        "train": {
            "rows": TRAIN_ROWS,
            "frauds": TRAIN_FRAUDS,
            "transaction_ids_sha256": DIGEST,
            "file_sha256": DIGEST,
        },
        "positive_weight": (TRAIN_ROWS - TRAIN_FRAUDS) / TRAIN_FRAUDS,
        "preprocessing": file("preprocessing.json"),
        "preflight": file("preflight/FTT/trial-03.json"),
        "model": file("models/FTT-b/trial-03/model.pt"),
        "model_record": file("models/FTT-b/trial-03/run.json"),
        "fit_seconds": 3600.5,
        "status": "completed",
        "error": None,
    },
    "checkpoint": {
        "configuration": "LGB-u",
        "trial": 1,
        "unit": "round",
        "index": 120,
        "objective": 0.41,
        "model": file("models/LGB-u/trial-01/model.txt"),
        "model_record": file("models/LGB-u/trial-01/run.json"),
    },
    "trial": {
        "configuration": "LGB-u",
        "trial": 1,
        "candidate_list": file("candidates.json"),
        "hyperparameters": {"learning_rate": 0.1, "num_leaves": 31},
        "seed": 42,
        "device": "cpu",
        "unit": "round",
        "objective_k": 62,
        "patience": 50,
        "maximum": 2000,
        "best_checkpoint": 120,
        "best_objective": 0.41,
        "stopped_at": 170,
        "stopping_reason": "early_stopping",
        "runtime_seconds": 12.5,
        "checkpoints": history(120, 170),
        "fit": file("fits/LGB-u/trial-01.json"),
        "checkpoint": file("checkpoints/LGB-u/trial-01.json"),
        "status": "completed",
        "error": None,
    },
    "configuration": {
        "configuration": "LGB-u",
        "candidate_list": file("candidates.json"),
        "trials": [
            {
                "trial": number,
                "objective": objective,
                "stopping_reason": "early_stopping",
                "record": file(f"trials/LGB-u/trial-{number:02d}.json"),
            }
            for number, objective in enumerate((0.41, 0.43, 0.43, 0.40), start=1)
        ],
        "selected_trial": 2,
        "selected_objective": 0.43,
        "tied_trials": [2, 3],
        "selected_hyperparameters": {"learning_rate": 0.05},
        "selected_length": 300,
        "selected_stopping_reason": "early_stopping",
        "selected_fit": file("fits/LGB-u/trial-02.json"),
        "selected_checkpoint": file("checkpoints/LGB-u/trial-02.json"),
    },
    "refit": {
        "configuration": "Cat-u",
        "kind": "same_seed_repeat",
        "seed": 42,
        "device": "gpu",
        "selected_trial": 4,
        "length": 800,
        "fit": file("fits/Cat-u/repeat.json"),
        "score": file("scores/Cat-u/repeat-v2.json"),
        "reference_score": file("scores/Cat-u/v2.json"),
        "comparison": {
            "rows": V2_ROWS,
            "identical": False,
            "differing_rows": 15,
            "max_abs_difference": 1.2e-7,
        },
        "status": "completed",
        "error": None,
    },
    "device_comparison": {
        "configuration": "Cat-u",
        "seed": 42,
        "gpu_score": file("scores/Cat-u/v2.json"),
        "cpu_score": file("scores/Cat-u/cpu-v2.json"),
        "refit": file("records/refit-Cat-u-cpu.json"),
        "capacities": [
            {
                "capacity": 0.02,
                "k": 56,
                "differences": {"recall": -0.004, "precision": -0.0003},
            },
            {"capacity": 0.05, "k": 139, "differences": {"recall": 0.001}},
        ],
        "bootstrap": {
            "capacity": 0.02,
            "k": 56,
            "entity_key": "fine",
            "estimate": -0.004,
            "lower": -0.021,
            "upper": 0.012,
            "level": 0.95,
            "replicates": 10_000,
            "undefined_replicates": 0,
            "seed": 42,
            "days": 28,
            "entities": 1353,
            "frauds": V2_FRAUDS,
        },
    },
    "score": {
        "configuration": "RES-u",
        "model_role": "selected",
        "seed": 42,
        "fit": file("fits/RES-u/trial-05.json"),
        "partition": "V2",
        "first_day": 127,
        "last_day": 154,
        "rows": V2_ROWS,
        "file": file("scores/RES-u/v2.parquet"),
        "format": "parquet",
        "columns": ["TransactionID", "day", "score"],
        "score_dtype": "float32",
        "device": "gpu",
        "batch_rows": 4096,
        "min_score": 0.0001,
        "max_score": 0.98,
        "distinct_scores": 76_000,
        "seconds": 3.2,
    },
    "evaluation": {
        "configuration": "LGB-u",
        "model_role": "selected",
        "seed": 42,
        "score": file("scores/LGB-u/v2.json"),
        "partition": "V2",
        "period": "primary",
        "days": {"first": 127, "last": 154},
        "policy": "R3",
        "capacity": 0.02,
        "k": 56,
        "threshold": None,
        "cost_ratio": None,
        "calibrator": None,
        "calibration": None,
        "counts": {
            "days": 28,
            "transactions": V2_ROWS,
            "frauds": V2_FRAUDS,
            "selected": 1568,
            "true_positives": 900,
            "false_positives": 668,
            "false_negatives": V2_FRAUDS - 900,
        },
        "metrics": {
            "recall": 900 / V2_FRAUDS,
            "precision": 900 / 1568,
            "normalised_recall": 0.6,
            "average_precision": None,
        },
        "daily": [],
        "cutoff_ties": {"days": 1, "group_sizes": [3]},
    },
    "timing": {
        "configuration": "RES-u",
        "measurement": "latency",
        "model_role": "selected",
        "subject": file("fits/RES-u/trial-05.json"),
        "partition": None,
        "device": "gpu",
        "boundary": "model_only",
        "threads": None,
        "cpu_affinity": None,
        "synchronised": True,
        "attention_backend": None,
        "row_selection": "10,000 V1 rows with the smallest SHA-256 digests of the TransactionID",
        "rows_sha256": DIGEST,
        "batch_rows": 1,
        "warmup": 100,
        "repetitions": 3,
        "rows": 10_000,
        "seconds": None,
        "rows_per_second": None,
        "latency_ms": {"median": 0.4, "p95": 0.6, "p99": 0.9},
        "peak_gpu_mib": None,
        "peak_host_mib": 512.0,
        "memory_scope": "process resident set",
        "settings": {
            "procedure_version": 1,
            "session": {"load_average_1m": 0.25},
            "memory_problems": [],
            "percentile_method": "linear",
            "timed_calls": 30_000,
            "round_latency_ms": [{"median": 0.4, "p95": 0.6, "p99": 0.9}] * 3,
            "host_memory_method": "clear_refs",
        },
    },
    "calibration": {
        "configuration": "FTT-u",
        "method": "platt",
        "period": "primary",
        "score": file("scores/FTT-u/v2.json"),
        "fitting_days": {"first": 127, "last": 154},
        "rows": V2_ROWS,
        "frauds": V2_FRAUDS,
        "legitimate": V2_ROWS - V2_FRAUDS,
        "clipping": {"epsilon": 1e-15, "clipped_low": 0, "clipped_high": 2},
        "targets": {
            "positive": (V2_FRAUDS + 1) / (V2_FRAUDS + 2),
            "negative": 1 / (V2_ROWS - V2_FRAUDS + 2),
        },
        "parameters": {"intercept": -0.3, "slope": 0.9},
        "solver": {
            "name": "newton",
            "settings": {"tolerance": 1e-12, "max_iterations": 100, "start": [0.0, 1.0]},
            "converged": True,
            "iterations": 7,
        },
        "knots": None,
        "artifact": file("calibration/FTT-u/platt.json"),
        "software": {"scipy": "1.18.1"},
        "reload_verified": True,
        "diagnostics": [
            {
                "rows_used": "fitting",
                "days": {"first": 127, "last": 154},
                "rows": V2_ROWS,
                "frauds": V2_FRAUDS,
                "brier": 0.03,
                "log_loss": 0.12,
                "metric_clipped": 0,
                "intercept": 0.01,
                "slope": 1.02,
                "out_of_range": {"below": 0, "above": 0},
                "reliability": bins(V2_ROWS, V2_FRAUDS),
            }
        ],
        "status": "completed",
        "error": None,
    },
    "cost_analysis": {
        "configuration": "Cat-u",
        "calibrator": "platt",
        "period": "primary",
        "capacity": 0.02,
        "k": 56,
        "days": {"first": 127, "last": 154},
        "score": file("scores/Cat-u/v2.json"),
        "calibration": file("records/calibration-Cat-u-platt.json"),
        "queue": {"reviews": 1568, "frauds": V2_FRAUDS, "captured": 900},
        "saturation": {"ratio": 10.0, "infinite": False},
        "break_even": {
            "classification": "coincidence",
            "ratio": 10.0,
            "single_interval": True,
            "intervals": [{"start": 1.0, "end": 10.0, "end_closed": False}],
            "zero_crossings": [],
        },
        "friction": [{"phi": 1.0, "ratio": 19.0}, {"phi": 5.0, "ratio": 55.0}],
        "grid": _cost_grid(),
    },
    "r1_selection": {
        "configuration": "Cat-b",
        "score": file("scores/Cat-b/v2.json"),
        "period": "primary",
        "selection_days": {"first": 127, "last": 154},
        "transactions": V2_ROWS,
        "frauds": V2_FRAUDS,
        "distinct_scores": 70_000,
        "threshold": 0.7234567890123456,
        "true_positives": 1200,
        "false_positives": 900,
        "false_negatives": V2_FRAUDS - 1200,
        "alerts": 2100,
        "f1_numerator": 2400,
        "f1_denominator": V2_FRAUDS + 2100,
        "rule_version": "1",
        "status": "selected",
        "error": None,
    },
    "h3_selection": {
        "period": "primary",
        "selection_days": {"first": 127, "last": 154},
        "binding": True,
        "capacity": 0.02,
        "k": 56,
        "candidates": [
            {
                "configuration": name,
                "frauds_captured": H3_CAPTURED[name],
                "frauds": V2_FRAUDS,
                "recall": H3_CAPTURED[name] / V2_FRAUDS,
                "average_precision": 0.5,
                "evaluation": file(f"evaluations/{name}/v2-r3-2.json"),
            }
            for name in CONFIGURATION_NAMES
        ],
        "selected": "Cat-u",
        "tie_break_steps": [],
        "rule_version": "1",
    },
}


_BODIES["evaluation"] = with_days(_BODIES["evaluation"], 127, 154)
_BODIES["run_end"] = {
    "finished": "2026-10-20T10:15:30.5Z",
    "outcome": "completed",
    "error": None,
    "records": [file("records/run.json")],
}


def example(kind: str) -> dict[str, Any]:
    """A fresh, valid record of ``kind``."""
    return _envelope(kind) | copy.deepcopy(_BODIES[kind])
