"""The GPU acceptance probe of DR-013 §8, specified in ``docs/gpu-acceptance-probe.md``.

* ``synthetic``: synthetic inputs in the shape of the study's inputs (no research data);
* ``diagnostics``: score hashes, top-K comparisons, timing and memory measurement;
* ``records``: the JSON record and the Markdown summary generated from it;
* ``runner``: the checks C1–C13 and the tolerances T1–T4 of the specification.
"""
