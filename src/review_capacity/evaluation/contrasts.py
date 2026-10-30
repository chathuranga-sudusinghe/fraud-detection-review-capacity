"""The estimands of H1 and H2 and the descriptive contrasts of DR-005, as coefficient vectors over
the declared configuration order (Issue #189).

Every estimand is a difference of means of Recall@2 % over configurations, so it is a linear
combination of the eight recalls. A coefficient vector applies to the recalls of the data and to
every replicate of the pigeonhole bootstrap alike, so the replicate estimands (delta*_F,
delta*_W,m, ...) follow from the replicate recalls with the same weights. The p-values and
intervals of these estimands belong to the inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from review_capacity.errors import DataValidationError, SpecificationError
from review_capacity.evaluation.declaration import WEIGHTINGS, EvaluationDeclaration

type Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Estimands:
    """Named coefficient vectors: one row per estimand, one column per configuration."""

    names: tuple[str, ...]
    coefficients: Floats

    def row(self, name: str) -> Floats:
        """The coefficient vector of the estimand ``name``."""
        if name not in self.names:
            raise DataValidationError(f"Unknown estimand {name!r}; expected one of {self.names}")
        return np.asarray(self.coefficients[self.names.index(name)], dtype=np.float64)

    def apply(self, values: npt.ArrayLike) -> Floats:
        """The estimands of ``values``, whose last axis is the configurations: a vector of eight
        recalls gives one value per estimand, a (replicates x configurations) matrix gives a
        (replicates x estimands) matrix. NaN replicates stay NaN."""
        data = np.asarray(values, dtype=np.float64)
        if data.ndim < 1 or data.shape[-1] != self.coefficients.shape[1]:
            raise DataValidationError(
                f"values need {self.coefficients.shape[1]} configurations on the last axis, "
                f"not shape {data.shape}"
            )
        return np.asarray(data @ self.coefficients.T, dtype=np.float64)


def estimands(declaration: EvaluationDeclaration) -> Estimands:
    """The estimands of DR-005 over ``declaration.configurations.order``:

    * ``family``: delta_F, the mean of the neural configurations minus the mean of the tree-based
      ones (H1);
    * ``family:<w>``: the family effect at one weighting w, N_w - T_w (the neural mean minus the
      tree-based mean at that weighting), reported separately when the interaction qualifies;
    * ``weighting:<model>``: delta_W,m, balanced minus unweighted, for each model (H2);
    * ``within:<b>-<a>``: the second model of a family minus the first, averaged over weighting
      (descriptive);
    * ``interaction``: (N_b - T_b) - (N_u - T_u), with N_w and T_w the family means at weighting w
      (descriptive);
    * ``cell:<a>-<b>``: each declared cell contrast, a minus b (descriptive).
    """
    configurations = declaration.configurations
    tree, neural = configurations.tree, configurations.neural
    if len(tree) != 2 or len(neural) != 2:
        raise SpecificationError("The within-family contrasts need two models per family (DR-005)")
    unweighted, balanced = WEIGHTINGS
    position = {name: index for index, name in enumerate(configurations.order)}

    def vector(parts: dict[str, float]) -> Floats:
        row = np.zeros(len(position), dtype=np.float64)
        for name, coefficient in parts.items():
            row[position[name]] += coefficient
        return row

    def family(models: tuple[str, ...], weighting: str, coefficient: float) -> dict[str, float]:
        return {f"{model}-{weighting}": coefficient for model in models}

    rows: dict[str, Floats] = {
        "family": vector(
            {
                **{f"{model}-{w}": 0.25 for model in neural for w in WEIGHTINGS},
                **{f"{model}-{w}": -0.25 for model in tree for w in WEIGHTINGS},
            }
        )
    }
    for weighting in WEIGHTINGS:
        rows[f"family:{weighting}"] = vector(
            {**family(neural, weighting, 0.5), **family(tree, weighting, -0.5)}
        )
    for model in (*tree, *neural):
        rows[f"weighting:{model}"] = vector(
            {f"{model}-{balanced}": 1.0, f"{model}-{unweighted}": -1.0}
        )
    for models in (tree, neural):
        first, second = models
        rows[f"within:{second}-{first}"] = vector(
            {
                **{f"{second}-{w}": 0.5 for w in WEIGHTINGS},
                **{f"{first}-{w}": -0.5 for w in WEIGHTINGS},
            }
        )
    rows["interaction"] = vector(
        {
            **family(neural, balanced, 0.5),
            **family(tree, balanced, -0.5),
            **family(neural, unweighted, -0.5),
            **family(tree, unweighted, 0.5),
        }
    )
    for first_cell, second_cell in declaration.contrasts:
        rows[f"cell:{first_cell}-{second_cell}"] = vector({first_cell: 1.0, second_cell: -1.0})
    return Estimands(tuple(rows), np.vstack(list(rows.values())))
