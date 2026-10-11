"""A feature catalog in the strict catalog format, stating the values of a synthetic profile.

``catalog_text`` writes every statement that ``data.catalog`` interprets, so that a test can change
one statement and expect exactly one difference.
"""

from __future__ import annotations

import pandas as pd

FIELD = (
    "| Column | Table | Official meaning | Observed dtype "
    "| Observed range / most frequent values | Interpretation | Model input "
    "| Reason for treatment | Missing (dev) | Levels in Train | Caution |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|\n"
)
TABLES = {"TransactionID": "both (join key)", "id_01": "identity", "id_13": "identity",
          "DeviceType": "identity"}  # fmt: skip
ROLES = {"TransactionID": "identifier", "TransactionDT": "temporal", "ProductCD": "categorical",
         "card1": "categorical", "DeviceType": "categorical", "id_13": "categorical"}  # fmt: skip


def catalog_text(profile: pd.DataFrame, cautions: dict[str, str] | None = None) -> str:
    """A catalog stating the profiled values, in the strict catalog format."""
    cautions = cautions or {}
    main, appendix = [], []
    for column, row in profile.iterrows():
        name = str(column)
        missing = f"{100 * row['missing_dev']:.1f} %"
        if row["kind"] == "text":
            dtype, values = "text", ", ".join(f"`{v}`" for v in str(row["top_values"]).split(" | "))
        else:
            dtype, values = "float64", f"{row['min_dev']:.3f} to {row['max_dev']:.3f}"
        caution = [cautions[name]] if name in cautions else []
        if pd.notna(row.get("v2_unseen")):
            caution.append(f"Validation 2 values unseen in Train: {100 * row['v2_unseen']:.1f} %")
        if row["kind"] == "numeric" and row["negatives_dev"]:
            caution.append(f"{int(row['negatives_dev'])} negative values")
        if name.startswith("V"):
            integer = "yes" if row["integer_valued"] else "no"
            appendix.append(
                f"| `{name}` | {missing} | {row['levels_train']} | {values} | {integer} |"
            )
            continue
        model_input = "exclude" if name in ("TransactionID", "TransactionDT") else "include"
        main.append(
            f"| `{name}` | {TABLES.get(name, 'transaction')} | m | {dtype} | {values} "
            f"| {ROLES.get(name, 'numeric')} | {model_input} | r | {missing} "
            f"| {row['levels_train']} | {'; '.join(caution) or '—'} |"
        )
    main.append(
        "| `isFraud` | transaction | m | binary 0/1 | 1 = fraud | target | exclude | r "
        "| 0.0 % | 2 | — |"
    )
    main.extend(
        f"| `{name}` | derived (from `TransactionDT`) | m | float64 | −1 to 1 (theoretical bounds) "
        "| numeric | include | r | 0.0 % (guaranteed) | 0 | — |"
        for name in ("tod_sin", "tod_cos")
    )
    v_missing = f"{100 * float(profile['missing_dev'].loc['V1']):.1f} %"
    return (
        FIELD
        + "\n".join(main)
        + "\n\n| Role | Columns | Count |\n|---|---|---|\n| Target | t | 1 |\n"
        + "| Excluded (identifier / raw time) | e | 2 |\n| Categorical input | c | 4 |\n"
        + "| Numeric input | n | 3 |\n| **Total** | | **10** |\n\n"
        + "| Property | Value |\n|---|---|\n"
        + "| Table | transaction |\n| Observed dtype | float64 |\n| Interpretation | numeric |\n"
        + "| Model input | include |\n"
        + f"| Missing (dev) | {v_missing} to {v_missing} per column |\n"
        + "| Levels in Train | 3 to 3 per column; 0 columns are two-valued; "
        + "1 are integer-valued |\n"
        + "| Structure | 1 blocks of columns share an identical missing-row pattern in Train |\n\n"
        + "| Block | Columns | Missing in Train |\n|---|---|---|\n"
        + "| 1 | `V1` (1 column) | 25.0 % |\n\n"
        + "| Column | Missing (dev) | Levels in Train | Observed range | Integer-valued |\n"
        + "|---|---|---|---|---|\n"
        + "\n".join(appendix)
        + "\n"
    )
