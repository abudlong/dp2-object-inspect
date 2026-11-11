"""Build notebooks/dp2_object_inspect.ipynb from scripts/dp2_object_inspect.py (percent format).

Usage:  python tools/py_to_ipynb.py
(Equivalent to `jupytext --to ipynb`, without needing jupytext installed.)
"""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "scripts" / "dp2_object_inspect.py").read_text()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in header:
        text = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": body.splitlines(True)})
for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"kernelspec": {"name": "LSST", "display_name": "LSST", "language": "python"},
                   "language_info": {"name": "python"}}}
out = root / "notebooks" / "dp2_object_inspect.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out.relative_to(root)} ({len(cells)} cells)")
