"""Deterministic tables, paired user bootstrap, figures and a self-contained report."""
import base64
import html
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from .audit import write_json


METRICS = ["ndcg", "recall", "hit_rate", "aplt", "arp", "catalog_coverage", "candidate_recall100", "tail_recall10"]


def flatten(metrics, **identity):
    rows = []
    for group, values in metrics.items():
        row = identity | {"group": group, "users": values["users"]}
        for metric in METRICS:
            entry = values.get(metric)
            row[metric] = entry.get("mean") if isinstance(entry, dict) else entry
        rows.append(row)
    return rows


def paired_bootstrap(differences, seed=20261105, repetitions=500, progress=None):
    difference = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for iteration in range(repetitions):
        means.append(difference[rng.integers(0, len(difference), len(difference))].mean())
        if progress and iteration % 100 == 0: progress()
    return {"mean_difference": float(difference.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)), "users": len(difference), "bootstrap_repetitions": repetitions}


def build_report(output, store, plan, records, progress=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output); report = output / "report"; report.mkdir(exist_ok=True)
    figures = report / "figures"; figures.mkdir(exist_ok=True)
    rows = []
    for run_id, record in records.items():
        result = record["summary"]
        if not result["purpose"].startswith("final"): continue
        settings = result["settings"]
        rows.extend(flatten(result["test_metrics"], run_id=run_id, model=settings["model"], variant=settings["variant"],
                            seed=settings["seed"], layers=settings["layers"], purpose=result["purpose"], best_epoch=result["best_epoch"]))
    baseline = store.get_json("baseline/test/completed.json")["summary"]
    rows.extend(flatten(baseline["metrics"], run_id="popularity", model="popularity", variant="popularity", seed=None,
                        layers=0, purpose="baseline", best_epoch=0))
    frame = pl.DataFrame(rows)
    frame.write_csv(report / "all_seed_results.csv")
    main = frame.filter(pl.col("purpose").is_in(["final_main", "baseline"]))
    aggregate = main.group_by("variant", "group").agg(pl.len().alias("n_runs"), *[expression for metric in METRICS for expression in
        [pl.col(metric).mean().alias(metric + "_mean"), pl.col(metric).std(ddof=1).alias(metric + "_seed_sd")]]).sort("group", "variant")
    aggregate.write_csv(report / "model_comparison.csv")
    frame.filter(pl.col("group") != "all").write_csv(report / "group_results.csv")
    frame.filter(pl.col("purpose") == "final_ablation").write_csv(report / "ablation_results.csv")
    timing = []
    for run_id, record in records.items():
        result = record["summary"]
        timing.append({"run_id": run_id, "purpose": result["purpose"], "model": result["model"],
                       "epochs": result["completed_epochs"], "elapsed_seconds": result["elapsed_seconds"],
                       "training_seconds": sum(row["training_seconds"] for row in result["history"]),
                       "validation_seconds": sum(row.get("validation_seconds", 0) for row in result["history"]),
                       "best_epoch": result["best_epoch"], "peak_allocated_MiB": result["peak_allocated_MiB"]})
    pl.DataFrame(timing).write_csv(report / "timing_cost.csv")

    # Bootstrap is paired within each seed/user cohort; seed variability stays separate.
    cache = {}
    def user_frame(variant, seed):
        key = (variant, seed if variant != "popularity" else None)
        if key in cache: return cache[key]
        if variant == "popularity":
            item = store.find("baseline/test/per_user.parquet")
            path = report / "bootstrap_inputs/popularity.parquet"
        else:
            run_id = f"final_{variant}_seed{seed}"
            artifact = next(item for item in records[run_id]["artifacts"] if item["relative"] == "test/per_user.parquet")
            item = artifact
            path = report / "bootstrap_inputs" / f"{variant}_{seed}.parquet"
        store.download(item["id"], path, item.get("sha256"))
        cache[key] = pl.read_parquet(path).sort("uid")
        return cache[key]
    paired = []
    for seed in plan["seeds"]:
        for left, right in itertools.combinations(["popularity", "bpr_mf", "lightgcn", "xsimgcl"], 2):
            a, b = user_frame(left, seed), user_frame(right, seed)
            if not np.array_equal(a["uid"].to_numpy(), b["uid"].to_numpy()): raise ValueError("Paired comparison user cohorts differ.")
            for metric in ["ndcg", "aplt"]:
                result = paired_bootstrap(b[metric].to_numpy() - a[metric].to_numpy(), progress=progress)
                paired.append({"seed": seed, "left": left, "right": right, "metric": metric, **result})
    pl.DataFrame(paired).write_csv(report / "paired_differences.csv")

    colors = {"bpr_mf":"#2563eb", "lightgcn":"#db2777", "xsimgcl":"#059669", "popularity":"#64748b"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in aggregate.filter(pl.col("group") == "all").iter_rows(named=True):
        name = row["variant"]
        ax.errorbar(row["aplt_mean"], row["ndcg_mean"], xerr=row["aplt_seed_sd"] or 0,
                    yerr=row["ndcg_seed_sd"] or 0, fmt="o", color=colors[name], label=name, capsize=3)
    ax.set(xlabel="APLT@10 (tail exposure)", ylabel="NDCG@10", title="Accuracy and tail exposure; error bars: seed SD")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(figures / "accuracy_tail.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, model in zip(axes, ["bpr_mf", "lightgcn", "xsimgcl"]):
        for seed in plan["seeds"]:
            history = records[f"final_{model}_seed{seed}"]["summary"]["history"]
            observed = [row for row in history if "valid_ndcg_at_10" in row]
            ax.plot([row["epoch"] for row in observed], [row["valid_ndcg_at_10"] for row in observed], label=f"seed {seed}")
        ax.set(title=model, xlabel="Epoch", ylabel="Validation NDCG@10"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "learning_curves.png", dpi=160); plt.close(fig)
    light = frame.filter((pl.col("model") == "lightgcn") & (pl.col("group") == "all")).group_by("layers").agg(pl.col("ndcg").mean(), pl.col("ndcg").std().alias("ndcg_sd"), pl.col("aplt").mean(), pl.col("aplt").std().alias("aplt_sd")).sort("layers")
    light.write_csv(report / "depth_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["ndcg", "aplt"]):
        ax.errorbar(light["layers"], light[metric], yerr=light[metric + "_sd"].fill_null(0), fmt="o-", capsize=4)
        ax.set(xlabel="LightGCN layers", ylabel=metric.upper()+"@10", xticks=[1,2,3]); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "depth_ablation.png", dpi=160); plt.close(fig)

    def table(data, columns):
        result = "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"
        for row in data.iter_rows(named=True):
            result += "<tr>" + "".join("<td>" + (f"{row[c]:.6f}" if isinstance(row[c], float) else html.escape(str(row[c]) if row[c] is not None else "—")) + "</td>" for c in columns) + "</tr>"
        return result + "</tbody></table>"
    overall = aggregate.filter(pl.col("group") == "all").sort("ndcg_mean", descending=True)
    ablation = frame.filter((pl.col("group") == "all") & (pl.col("variant").is_in(["xsimgcl", "xsimgcl_cl0"]))).group_by("variant").agg(pl.col("ndcg").mean(), pl.col("ndcg").std().alias("ndcg_seed_sd"), pl.col("aplt").mean()).sort("variant")
    best = overall.row(0, named=True)
    limited = sum(record["summary"]["best_epoch"] >= plan["final_epochs"] - 5 for record in records.values() if record["summary"]["purpose"].startswith("final"))
    lead = f"test NDCG@10 평균이 가장 높은 모델은 {best['variant']} ({best['ndcg_mean']:.6f})이다. 설정·epoch 선택에는 valid만 사용했다."
    markdown = f"# Automotive 추천 본실험 결과\n\n{lead}\n\n"
    markdown += "## 평가 범위\n\ntrain만 학습한 warm-start 전체 상품 평가다. test 대상은 158,805명, 정답 333,797개이며 원래 test 상호작용의 약 55.2%다.\n\n"
    markdown += f"학습 모델 seed는 {plan['seeds']}이고 Popularity는 결정적 1회 기준선이다. seed SD와 사용자별 paired bootstrap CI는 별도로 제공한다.\n\n"
    markdown += "|모델|NDCG@10 평균|seed SD|Recall@10|APLT@10|\n|---|---:|---:|---:|---:|\n"
    for row in overall.iter_rows(named=True):
        sd = f"{row['ndcg_seed_sd']:.6f}" if row['ndcg_seed_sd'] is not None else "해당 없음"
        markdown += f"|{row['variant']}|{row['ndcg_mean']:.6f}|{sd}|{row['recall_mean']:.6f}|{row['aplt_mean']:.6f}|\n"
    markdown += "\n"
    markdown += "## 저장 파일\n\nmodel_comparison.csv: 평균/seed SD, all_seed_results.csv: seed별 점수, group_results.csv: 활동 그룹, depth_results.csv/ablation_results.csv: 보조 실험, paired_differences.csv: 동일 사용자 차이의 CI, timing_cost.csv: 시간/메모리.\n\n"
    markdown += f"## 해석과 한계\n\n예산 상한 부근에서 best epoch를 기록한 실행은 {limited}개다. 수렴 부족 가능성을 결과와 함께 고려한다. XSimGCL은 공통 squared raw L2를 적용한 recipe variant다.\n\n"
    markdown += "텍스트·차종 특징이 제공되지 않아 재정렬과 차종 불일치는 미실시다. 해당 지표를 0으로 대체하지 않았다.\n\n"
    markdown += "## 재현\n\nexperiment_plan.json, frozen_selection.json, 입력 manifest, 소스 snapshot, 실행별 manifest/epoch CSV/사용자별 Parquet/추천 점수와 best 모델을 같은 Drive 폴더에 보관한다.\n"
    (report / "RESULTS.md").write_text(markdown)
    body = f"<h1>Automotive 추천 본실험 결과</h1><p>{html.escape(lead)}</p>"
    body += "<p>전체 warm-start test · 3 seeds · macro 사용자 평균 · 오차막대는 seed SD</p>"
    body += table(overall, ["variant","n_runs","ndcg_mean","ndcg_seed_sd","recall_mean","aplt_mean","arp_mean","catalog_coverage_mean"])
    for name in ["accuracy_tail", "learning_curves", "depth_ablation"]:
        encoded = base64.b64encode((figures / (name + ".png")).read_bytes()).decode()
        body += f'<img alt="{name}" src="data:image/png;base64,{encoded}" />'
    body += "<h2>활동 그룹별 결과</h2>" + table(aggregate.filter(pl.col("group") != "all"), ["variant","group","ndcg_mean","aplt_mean","catalog_coverage_mean"])
    body += "<h2>XSimGCL 대조 손실 제거 비교</h2><p>동일 encoder/noise에서 CL 가중치를 0으로 바꾼 보조 실험이다.</p>" + table(ablation, ["variant", "ndcg", "ndcg_seed_sd", "aplt"])
    body += "<h2>범위와 한계</h2><p>test 상호작용 유지율 55.2%. 사용자 paired bootstrap와 seed SD를 구분한다. 원본 0-core/텍스트/차종 재정렬은 미실시. XSimGCL은 공통 L2 recipe variant다.</p>"
    body += f"<p>예산 상한 부근의 best epoch: {limited}개 실행. 추가 수렴 검토가 필요할 수 있다.</p>"
    (report / "RESULTS.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Automotive 결과</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172033}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd4e0;padding:8px}th{background:#eef2f8}img{max-width:100%;margin:24px 0}p{line-height:1.7}</style>'+body+"</html>")
    write_json(report / "completion.json", {"status": "completed", "campaign_id": plan["campaign_id"], "runs": len(records), "feature_experiments": "pending missing inputs"})
    for file in sorted(report.rglob("*")):
        if file.is_file() and "bootstrap_inputs" not in file.parts:
            store.put(file, "report/" + file.relative_to(report).as_posix())
