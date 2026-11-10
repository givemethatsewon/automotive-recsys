"""Build the offline beginner explainer from completed experiment CSVs.

UV_CACHE_DIR=.uv-cache uv run --no-project --python .venv/bin/python \
  --with latex2mathml==3.78.1 scripts/build_results_eli5.py
"""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics

from latex2mathml.converter import convert

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/results/final"
OUTPUT = ROOT / "docs/explainers/experiment-results-eli5.html"


def read_csv(name):
    return list(csv.DictReader((SOURCE / name).open()))


def m(tex):
    return '<span class="math" data-latex="' + html.escape(tex, quote=True) + '">' + convert(tex) + '</span>'


def decimal(value, places=8):
    return f"{value:.{places}f}"


def build():
    completion = json.loads((SOURCE / "completion.json").read_text())
    assert completion["status"] == "completed" and completion["runs"] == 32
    models = ["popularity", "bpr_mf", "lightgcn", "xsimgcl"]
    data = {"groups": {}, "models": models, "seedRows": {}, "comparisons": {}, "toy": {}, "tail": {}, "tailValue": {},
            "ci": {}, "depth": [], "cl": {}, "math": {}, "sources": {}}
    metrics = ["ndcg", "recall", "aplt", "catalog_coverage", "arp", "hit_rate", "candidate_recall100"]
    rows = read_csv("model_comparison.csv")
    for group, count in [("all", 158805), ("1-5", 87394), ("6+", 71411)]:
        data["groups"][group] = {"users": count, "rows": {}, "comparisons": {}}
        for variant in models:
            row = next(r for r in rows if r["variant"] == variant and r["group"] == group)
            values = {key: float(row[key + "_mean"]) for key in metrics}
            values["sd"] = float(row["ndcg_seed_sd"]) if row["ndcg_seed_sd"] else None
            values["display"] = {key: m(decimal(values[key] * (100 if key in {"aplt", "catalog_coverage"} else 1),
                                                         8 if key in {"ndcg", "recall", "aplt"} else 4 if key == "catalog_coverage" else 2)
                                          + (r"\%" if key in {"aplt", "catalog_coverage"} else "")) for key in metrics}
            data["groups"][group]["rows"][variant] = values
        for base in models:
            data["groups"][group]["comparisons"][base] = {}
            for target in models:
                a = data["groups"][group]["rows"][base]["ndcg"]
                b = data["groups"][group]["rows"][target]["ndcg"]
                gain = (b / a - 1) * 100
                data["groups"][group]["comparisons"][base][target] = {
                    "gain": gain, "formula": m(r"\left(\frac{" + decimal(b) + "}{" + decimal(a) + r"}-1\right)\times100\%\approx" + f"{gain:.2f}" + r"\%")}
    for rank in range(1, 12):
        hits = [rank] if rank <= 10 else []
        data["toy"][f"one-{rank}"] = toy(hits, 1)
    data["toy"]["two"] = toy([1, 3], 2)
    for count in range(11):
        data["tail"][str(count)] = m(r"\mathrm{APLT}@10=\frac{" + str(count) + r"}{10}=" + f"{count / 10:.1f}" + r"=" + str(count * 10) + r"\%")
        data["tailValue"][str(count)] = m(str(count * 10) + r"\%")
    seed_rows = read_csv("all_seed_results.csv")
    for variant in models[1:]:
        matched = sorted([r for r in seed_rows if r["variant"] == variant and r["group"] == "all"], key=lambda r: r["seed"])
        assert [r["seed"] for r in matched] == ["42", "43", "44"]
        values = [float(r["ndcg"]) for r in matched]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        assert abs(mean - data["groups"]["all"]["rows"][variant]["ndcg"]) < 1e-14
        data["seedRows"][variant] = {"values": values, "mean": mean, "sd": sd,
            "valueMath": [m(decimal(v)) for v in values], "meanMath": m(decimal(mean)), "sdMath": m(decimal(sd)),
            "formula": m(r"\bar{x}=\frac{" + "+".join(decimal(v) for v in values) + r"}{3}\approx" + decimal(mean))}
    for row in read_csv("paired_differences.csv"):
        if row["left"] == "lightgcn" and row["right"] == "xsimgcl" and row["metric"] == "ndcg":
            values = {key: float(row[key]) for key in ["mean_difference", "ci95_low", "ci95_high"]}
            values["formula"] = m(decimal(values["mean_difference"]) + r",\quad\mathrm{CI}_{95\%}=\left[" + decimal(values["ci95_low"]) + "," + decimal(values["ci95_high"]) + r"\right]")
            data["ci"][row["seed"]] = values
    for row in read_csv("depth_results.csv"):
        values = {key: float(v) for key, v in row.items()}
        values["ndcgMath"] = m(decimal(values["ndcg"]))
        values["apltMath"] = m(f"{values['aplt'] * 100:.6f}" + r"\%")
        data["depth"].append(values)
    for variant in ["xsimgcl", "xsimgcl_cl0"]:
        matched = [r for r in seed_rows if r["variant"] == variant and r["group"] == "all"]
        assert len(matched) == 3
        a = statistics.mean(float(r["ndcg"]) for r in matched)
        b = statistics.mean(float(r["aplt"]) for r in matched)
        data["cl"][variant] = {"ndcg": a, "aplt": b, "ndcgMath": m(decimal(a)), "apltMath": m(f"{b * 100:.8f}" + r"\%")}
    gain = (data["cl"]["xsimgcl"]["ndcg"] / data["cl"]["xsimgcl_cl0"]["ndcg"] - 1) * 100
    data["cl"]["gainMath"] = m(f"{gain:.2f}" + r"\%")
    data["math"] = {key: m(tex) for key, tex in {
        "mf": r"s(u,i)=\mathbf{e}_u^\top\mathbf{e}_i",
        "dot": r"[0.8,0.2]\cdot[0.7,0.3]=0.8\times0.7+0.2\times0.3=0.62",
        "vecUser": r"[0.8,0.2]",
        "vecItem": r"[0.7,0.3]",
        "bpr": r"s(u,i^{+})>s(u,i^{-})",
        "logOne": r"2^1=2\quad\Rightarrow\quad\log_2(2)=1",
        "logThree": r"2^2=4\quad\Rightarrow\quad\log_2(4)=2",
        "discount": r"w_r=\frac{1}{\log_2(r+1)}",
        "gcn": r"\mathbf{e}_u^{(\ell+1)}=\sum_{i\in\mathcal{N}(u)}\frac{\mathbf{e}_i^{(\ell)}}{\sqrt{d_ud_i}}",
        "lightAvg": r"\mathbf{e}_{\mathrm{Light}}=\frac{\mathbf{e}^{(0)}+\mathbf{e}^{(1)}+\cdots+\mathbf{e}^{(L)}}{L+1}",
        "xAvg": r"\mathbf{e}_{\mathrm{XSim}}=\frac{\mathbf{e}^{(1)}+\cdots+\mathbf{e}^{(L)}}{L}",
        "lossOn": r"\mathcal{L}=\mathcal{L}_{\mathrm{BPR}}+\mathcal{L}_{\mathrm{L2}}+0.1\mathcal{L}_{\mathrm{CL}}",
        "lossOff": r"\mathcal{L}=\mathcal{L}_{\mathrm{BPR}}+\mathcal{L}_{\mathrm{L2}}",
        "dcg": r"\mathrm{DCG}@10=\sum_{r=1}^{10}\frac{\mathrm{rel}_r}{\log_2(r+1)}",
        "idcg": r"\mathrm{IDCG}@10=\sum_{r=1}^{\min(10,R)}\frac{1}{\log_2(r+1)}",
        "ndcg": r"\mathrm{NDCG}@10=\frac{\mathrm{DCG}@10}{\mathrm{IDCG}@10}",
        "std": r"s=\sqrt{\frac{(x_{42}-\bar{x})^2+(x_{43}-\bar{x})^2+(x_{44}-\bar{x})^2}{3-1}}",
        "arpA": r"\mathrm{ARP}@10=\frac{5\times1200+5\times50}{10}=625",
        "arpB": r"\mathrm{ARP}@10=\frac{10\times50}{10}=50",
        "apltZero": r"\mathrm{APLT}@10=\frac{0}{10}=0\%",
        "coverage": r"\mathrm{Coverage}@10=\frac{14}{248186}\times100\%\approx0.00564\%",
        "recall100": r"\mathrm{Recall}@100\approx3.296\%",
        "hitRate": r"\mathrm{HitRate}@10\approx1.421\%",
        "recallX": r"\mathrm{Recall}@10\approx0.7245\%",
        "tailSlots": r"\frac{2}{158805\times10\times3}\times100\%\approx0.00004198\%",
    }.items()}
    tail_total = sum(float(r["aplt"]) * int(r["users"]) * 10 for r in seed_rows if r["variant"] == "xsimgcl" and r["group"] == "all")
    assert abs(tail_total - 2) < 1e-8
    assert round(data["groups"]["all"]["rows"]["popularity"]["catalog_coverage"] * 248186) == 14
    for name in ["model_comparison.csv", "all_seed_results.csv", "paired_differences.csv", "depth_results.csv", "completion.json"]:
        data["sources"][name] = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    template = (ROOT / "scripts/templates/results_eli5.html").read_text()
    css = (ROOT / "scripts/templates/eli5_reading.css").read_text()
    js = (ROOT / "scripts/templates/results_eli5.js").read_text()
    license_text = (ROOT / "scripts/templates/ELI5-LICENSE.txt").read_text()
    output = template.replace("@@ELI5_LICENSE@@", license_text).replace("@@BASE_CSS@@", css).replace("@@SCRIPT@@", js).replace("@@DATA@@", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    output = re.sub(r"\[\[math:([A-Za-z0-9]+)\]\]", lambda match: data["math"][match[1]], output)
    assert "@@" not in output and "[[math:" not in output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True); OUTPUT.write_text(output)
    print(f"Built {OUTPUT} ({OUTPUT.stat().st_size:,} bytes); verified CSV means and two total tail recommendation slots.")


def toy(hits, truth_count):
    dcg = sum(1 / math.log2(r + 1) for r in hits)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(10, truth_count) + 1))
    score = dcg / ideal
    terms = "+".join(r"\frac{1}{\log_2(" + str(r + 1) + ")}" for r in hits) or "0"
    ideal_terms = "+".join(r"\frac{1}{\log_2(" + str(r + 1) + ")}" for r in range(1, truth_count + 1))
    return {"hits": hits, "truthCount": truth_count, "dcg": dcg, "ideal": ideal, "score": score,
            "macro": (score + 1 / math.log2(3)) / 3,
            "scoreMath": m(f"{score:.5f}"), "recallValueMath": m(f"{len(hits) / truth_count:.1f}"),
            "dcgMath": m(r"\mathrm{DCG}@10=" + terms + r"\approx" + f"{dcg:.5f}"),
            "idealMath": m(r"\mathrm{IDCG}@10=" + ideal_terms + r"\approx" + f"{ideal:.5f}"),
            "ndcgMath": m(r"\mathrm{NDCG}@10=\frac{" + f"{dcg:.5f}" + "}{" + f"{ideal:.5f}" + r"}\approx" + f"{score:.5f}"),
            "recallMath": m(r"\mathrm{Recall}@10=\frac{" + str(len(hits)) + "}{" + str(truth_count) + "}=" + f"{len(hits) / truth_count:.1f}"),
            "macroMath": m(r"\mathrm{mean}=\frac{" + f"{score:.5f}" + r"+0.63093+0}{3}\approx" + f"{(score + 1 / math.log2(3)) / 3:.5f}")}


if __name__ == "__main__": build()
