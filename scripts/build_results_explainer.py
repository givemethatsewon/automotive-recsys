"""Build the 10-slot shelf results explainer from completed experiment CSVs.

Default output is a standalone offline HTML document. --fragment writes the
body-only page used for the claude.ai Artifact publish (the host adds the skeleton).

UV_CACHE_DIR=.uv-cache uv run --no-project --python .venv/bin/python \
  --with latex2mathml==3.78.1 scripts/build_results_explainer.py [--fragment OUT]
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics as st

from latex2mathml.converter import convert

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "scripts/templates"
SOURCE = ROOT / "docs/results/final"
OUTPUT = ROOT / "docs/explainers/experiment-results-explainer.html"
N_ITEMS = 248186
METRICS = {"ndcg": "ndcg", "recall": "recall", "hit": "hit_rate", "aplt": "aplt", "cov": "catalog_coverage", "arp": "arp"}


def read_csv(name):
    return list(csv.DictReader((SOURCE / name).open()))


def build_data():
    rows = read_csv("all_seed_results.csv")
    models = {}
    for variant in ["popularity", "bpr_mf", "lightgcn", "xsimgcl", "xsimgcl_cl0"]:
        models[variant] = {}
        for group in ["all", "1-5", "6+"]:
            selected = [r for r in rows if r["variant"] == variant and r["group"] == group]
            assert selected, (variant, group)
            stats = {"n": len(selected)}
            for key, column in METRICS.items():
                values = [float(r[column]) for r in selected]
                stats[key] = st.mean(values)
                stats[key + "_sd"] = st.stdev(values) if len(values) > 1 else None
            models[variant][group] = stats
    # Cross-check against the shipped seed aggregate.
    for r in read_csv("model_comparison.csv"):
        m = models[r["variant"]][r["group"]]
        assert abs(m["ndcg"] - float(r["ndcg_mean"])) < 1e-12, r
        assert abs(m["aplt"] - float(r["aplt_mean"])) < 1e-12, r
    depth = [{k: float(v) for k, v in r.items()} for r in read_csv("depth_results.csv")]
    for d in depth:
        d["layers"] = int(d["layers"])
    paired = [{"seed": int(r["seed"]), "left": r["left"], "right": r["right"], "metric": r["metric"],
               "mean": float(r["mean_difference"]), "lo": float(r["ci95_low"]), "hi": float(r["ci95_high"])}
              for r in read_csv("paired_differences.csv")]
    return {"models": models, "depth": depth, "paired": paired, "n_items": N_ITEMS}


def inner(latex):
    """MathML children only, so JS can splice fragments into one <math>."""
    s = convert(latex)
    return re.sub(r"</math>$", "", re.sub(r"^<math[^>]*>", "", s))


def build_tex():
    tex = {f"term{r}": inner(rf"\frac{{1}}{{\log_2({r}+1)}}") for r in range(1, 11)}
    tex.update({
        "dcgL": inner(r"\mathrm{DCG}="),
        "idcgL": inner(r"\mathrm{IDCG}="),
        # 900000x are numeric placeholders replaced in the browser.
        "approx": inner(r"\approx 9000001"),
        "ndcgL": inner(r"\mathrm{NDCG@10}=\frac{9000001}{9000002}\approx 9000003"),
        "recallL": inner(r"\mathrm{Recall@10}=\frac{9000001}{9000002}\approx 9000003"),
        "apltL": inner(r"\mathrm{APLT@10}=\frac{9000001}{10}=9000003"),
        "sep": inner(r"\qquad"),
        "G": inner(r"|G|"),
        "pair_ndcg": inner(r"\bar d=\frac{1}{158805}\sum_{u}\left(\mathrm{NDCG}_u^{\text{B}}-\mathrm{NDCG}_u^{\text{A}}\right)"),
        "pair_aplt": inner(r"\bar d=\frac{1}{158805}\sum_{u}\left(\mathrm{APLT}_u^{\text{B}}-\mathrm{APLT}_u^{\text{A}}\right)"),
    })
    for key in ("approx", "ndcgL", "recallL", "apltL"):
        assert "<mn>9000001</mn>" in tex[key], key
    return tex


def render():
    page = (TEMPLATES / "results_explainer.html").read_text()
    page = re.sub(r'<span data-tex-block="([^"]+)"></span>', lambda m: convert(m.group(1), display="block"), page)
    page = re.sub(r'<span data-tex="([^"]+)"></span>', lambda m: convert(m.group(1)), page)
    assert "data-tex" not in page
    page = page.replace("__DATA__", json.dumps(build_data(), ensure_ascii=False))
    page = page.replace("__TEX__", json.dumps(build_tex(), ensure_ascii=False))
    page = page.replace("__JS__", (TEMPLATES / "results_explainer.js").read_text())
    assert not re.search(r"__(DATA|TEX|JS)__", page)
    return page


def standalone(fragment):
    head, body = fragment.split("</style>", 1)
    return ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'; img-src data:">\n'
            + head + "</style>\n</head>\n<body>" + body + "</body>\n</html>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment", type=Path, help="write the Artifact body-only page here instead")
    args = parser.parse_args()
    page = render()
    out = args.fragment or OUTPUT
    out.write_text(page if args.fragment else standalone(page))
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
