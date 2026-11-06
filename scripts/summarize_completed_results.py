"""Reproduce the reviewed 31/32 snapshot; the final campaign has its own reporter."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
from datetime import datetime
from zoneinfo import ZoneInfo


METRICS = ["ndcg", "recall", "hit_rate", "aplt", "arp", "catalog_coverage", "candidate_recall100", "tail_recall10"]
NAMES = {"popularity": "Popularity", "bpr_mf": "BPR-MF", "lightgcn": "LightGCN", "xsimgcl": "XSimGCL"}


def value(metrics, key):
    result = metrics[key]
    return result["mean"] if isinstance(result, dict) else result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    records = json.loads((root / "completed_final_summaries.json").read_text())
    baseline = json.loads((root / "popularity_test.json").read_text())["summary"]
    selection = json.loads((root / "frozen_selection.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    cost = json.loads((root / "account_cost.json").read_text())
    if progress["completed_runs"] != 31 or len(records) != 17:
        raise ValueError("This snapshot requires the archived 31/32 review inputs. Use the campaign final report for completed studies.")
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    assert selection["plan_hash"] == plan_hash
    assert selection["selection_data"] == "valid only" and selection["test_used_for_selection"] is False
    assert baseline["split"] == "test" and baseline["users"] == 158805
    assert baseline["ground_truth_interactions"] == 333797 and baseline["candidate_items"] == 248186
    inputs = records[0]["inputs"]
    by_variant = {}
    for result in records:
        settings = result["settings"]; variant = settings["variant"]
        assert result["test_evaluated"] and result["campaign_plan_hash"] == plan_hash
        assert result["training_data"] == "train only" and result["selection_data"] == "valid only"
        assert result["inputs"] == inputs and result["gpu"] == "NVIDIA L4"
        assert len(result["history"]) == result["completed_epochs"]
        assert [row["epoch"] for row in result["history"]] == list(range(1, result["completed_epochs"] + 1))
        for key in ["model", "layers", "dim", "learning_rate", "regularization", "cl_weight", "eps", "temperature"]:
            assert settings[key] == selection["choices"][variant][key]
        metrics = result["test_metrics"]
        for group in ["all", "1-5", "6+"]:
            assert metrics[group]["users"] == baseline["metrics"][group]["users"]
            assert metrics[group]["ground_truth_interactions"] == baseline["metrics"][group]["ground_truth_interactions"]
        for key in ["ndcg", "recall", "hit_rate", "aplt", "arp", "candidate_recall100"]:
            weighted = sum(value(metrics[g], key) * metrics[g]["users"] for g in ["1-5", "6+"]) / baseline["users"]
            assert abs(weighted - value(metrics["all"], key)) < 1e-10
        by_variant.setdefault(variant, []).append(result)
    for variant in ["bpr_mf", "lightgcn", "xsimgcl", "lightgcn_l1", "lightgcn_l2"]:
        assert sorted(r["settings"]["seed"] for r in by_variant[variant]) == [42, 43, 44]

    def aggregate(variant, group="all"):
        if variant == "popularity":
            return {"variant": variant, "group": group, "n_runs": 1, "users": baseline["metrics"][group]["users"],
                    **{f"{key}_{suffix}": value(baseline["metrics"][group], key) if suffix == "mean" else None
                       for key in METRICS for suffix in ["mean", "seed_sd"]}}
        rows = by_variant[variant]
        return {"variant": variant, "group": group, "n_runs": len(rows), "users": rows[0]["test_metrics"][group]["users"],
                **{f"{key}_{suffix}": (st.mean if suffix == "mean" else st.stdev)(value(r["test_metrics"][group], key) for r in rows)
                   if suffix == "mean" or len(rows) > 1 else None for key in METRICS for suffix in ["mean", "seed_sd"]}}

    variants = ["popularity", "bpr_mf", "lightgcn", "xsimgcl"]
    overall = {variant: aggregate(variant) for variant in variants}
    assets = args.output.parent / "results" / "2026-11-07"
    assets.mkdir(parents=True, exist_ok=True)
    write_csv(assets / "main_comparison.csv", list(overall.values()))
    write_csv(assets / "group_comparison.csv", [aggregate(v, g) for g in ["1-5", "6+"] for v in variants])
    write_csv(assets / "depth_comparison.csv", [aggregate(v) for v in ["lightgcn_l1", "lightgcn_l2", "lightgcn"]])
    seed_rows = []
    for r in records:
        seed_rows.append({"run_id": r["run_id"], "variant": r["settings"]["variant"], "seed": r["settings"]["seed"],
                          "completed_epochs": r["completed_epochs"], "best_epoch": r["best_epoch"],
                          **{key: value(r["test_metrics"]["all"], key) for key in METRICS},
                          "recorded_training_minutes": sum(h["training_seconds"] for h in r["history"]) / 60})
    write_csv(assets / "completed_seed_results.csv", seed_rows)
    verified = {"completed_final_records": len(records), "main_models_all_three_seeds": True,
                "same_input_fingerprints": True, "plan_hash": plan_hash,
                "validation_only_frozen_selection": True, "same_test_cohort_and_group_aggregation": True,
                "campaign_completed": progress["status"] == "completed", "completed_runs": progress["completed_runs"],
                "snapshot_at": progress["updated_at"]}
    (assets / "verification.json").write_text(json.dumps(verified, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#64748b", "#2563eb", "#db2777", "#059669"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, key, scale, title in zip(axes, ["ndcg", "aplt", "catalog_coverage"], [1, 100, 100],
                                      ["Accuracy: NDCG@10", "Tail exposure: APLT@10 (%)", "Catalog coverage@10 (%)"]):
        means = [overall[v][key + "_mean"] * scale for v in variants]
        errors = [(overall[v][key + "_seed_sd"] or 0) * scale for v in variants]
        ax.bar(range(4), means, yerr=errors, capsize=3, color=colors)
        ax.set_xticks(range(4), [NAMES[v] for v in variants], rotation=20)
        ax.set_title(title); ax.set_ylim(0, max(a + b for a, b in zip(means, errors)) * 1.25)
        for x, y in enumerate(means):
            label = f"{y:.6f}" if key == "ndcg" else (f"{y:.6f}%" if 0 < y < .001 else f"{y:.3f}%")
            ax.annotate(label, (x, y + errors[x]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Completed main-model test results | 3 seeds; error bars: sample seed SD")
    fig.tight_layout()
    for suffix in ["png", "pdf"]: fig.savefig(assets / f"main_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    depth_variants = ["lightgcn_l1", "lightgcn_l2", "lightgcn"]
    for ax, key, scale, title in zip(axes, ["ndcg", "aplt"], [1, 100], ["Test NDCG@10", "Test APLT@10 (%)"]):
        ax.errorbar([1, 2, 3], [aggregate(v)[key + "_mean"] * scale for v in depth_variants],
                    yerr=[aggregate(v)[key + "_seed_sd"] * scale for v in depth_variants], fmt="o-", capsize=4)
        ax.set(xlabel="LightGCN layers", ylabel=title, xticks=[1, 2, 3]); ax.grid(alpha=.2)
    fig.suptitle("Depth comparison | same LR / regularization; validation-selected epochs; 3 seeds")
    fig.tight_layout(); fig.savefig(assets / "depth_comparison.png", dpi=180); plt.close(fig)

    def pct_change(a, b): return (a / b - 1) * 100
    def table(headers, rows):
        return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)
    x, light, pop, mf = (overall[v] for v in ["xsimgcl", "lightgcn", "popularity", "bpr_mf"])
    snapshot = datetime.fromisoformat(progress["updated_at"]).astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")
    text = f"# Automotive 추천 실험 결과 정리\n\n기록 기준: {snapshot}. 캠페인 `{plan['campaign_id']}`.\n\n"
    text += f"**주 비교 3모델×3 seed와 Popularity test 평가는 완료됐다. 전체 캠페인은 {progress['completed_runs']}/32개 완료이며, 마지막 `xsimgcl_cl0` seed 44 보조 실험이 진행 중이다. 이 문서는 완료된 결과의 검증 스냅샷이며 최종 캠페인 완료 보고서가 아니다.**\n\n"
    text += "## 핵심 결과\n\n"
    text += f"XSimGCL의 test NDCG@10이 가장 높다. LightGCN 대비 **{pct_change(x['ndcg_mean'], light['ndcg_mean']):.2f}%**, Popularity 대비 **{pct_change(x['ndcg_mean'], pop['ndcg_mean']):.2f}%** 높다. 반면 비인기 상품 노출은 거의 0이다. 이번 조건에서는 대조학습의 정확도 개선과 비인기 노출 개선이 함께 나타나지 않았다.\n\n"
    text += table(["모델", "반복", "NDCG@10 평균 ± seed SD", "Recall@10 평균", "APLT@10 (%)", "Coverage@10 (%)", "ARP@10"],
                  [[NAMES[v], "결정적 1회" if v == "popularity" else "3 seeds",
                    f"{overall[v]['ndcg_mean']:.8f}" + (" (SD 해당 없음)" if v == "popularity" else f" ± {overall[v]['ndcg_seed_sd']:.8f}"),
                    f"{overall[v]['recall_mean']:.8f}", f"{overall[v]['aplt_mean'] * 100:.8f}",
                    f"{overall[v]['catalog_coverage_mean'] * 100:.4f}", f"{overall[v]['arp_mean']:.2f}"] for v in variants])
    text += "\nNDCG/Recall은 비율 원값, APLT/Coverage는 %다. APLT는 상위 10개 추천 중 비인기 상품 비율, ARP는 추천 상품의 학습 상호작용 수 평균이다. 오차는 사용자 CI가 아니라 seed 3개 간 sample SD(ddof=1)다.\n\n![주 모델 비교](results/2026-11-07/main_comparison.png)\n\n"
    text += f"- BPR-MF는 현재 설정 탐색 범위에서 Popularity보다 NDCG가 {abs(pct_change(mf['ndcg_mean'], pop['ndcg_mean'])):.2f}% 낮다. 개인화 모델이 항상 인기순보다 우수하다는 주장은 지지되지 않는다.\n"
    text += f"- LightGCN은 Popularity 대비 NDCG가 {pct_change(light['ndcg_mean'], pop['ndcg_mean']):.2f}% 높고, 주 비교 중 APLT도 가장 높다. 정확도와 비인기 노출을 함께 볼 때 후보가 된다.\n"
    text += f"- XSimGCL은 정확도가 가장 높지만 APLT는 {x['aplt_mean'] * 100:.8f}%, coverage는 {x['catalog_coverage_mean'] * 100:.2f}%다. 비인기 추천을 늘렸다고 해석할 수 없다.\n"
    text += "- XSimGCL의 ARP는 LightGCN보다 낮지만 APLT도 낮다. ARP 감소를 tail 노출 증가와 같은 뜻으로 해석하면 안 된다.\n\n"
    text += "## 사용자 활동 그룹\n\n"
    text += table(["그룹", "사용자 수", "Popularity NDCG", "BPR-MF NDCG", "LightGCN NDCG", "XSimGCL NDCG"],
                  [[g, baseline['metrics'][g]['users'], *[f"{aggregate(v, g)['ndcg_mean']:.8f}" for v in variants]] for g in ["1-5", "6+"]])
    text += "\n그룹은 train 이력 수로만 정의했다. 두 그룹 모두 XSimGCL의 평균 NDCG가 가장 높다. 이 결과를 cold-start(이력 0개) 성능으로 일반화하지 않는다.\n\n"
    text += "## LightGCN 층 수 비교\n\n"
    text += table(["층 수", "NDCG 평균 ± SD", "APLT (%)", "Coverage (%)"],
                  [[i, f"{aggregate(v)['ndcg_mean']:.8f} ± {aggregate(v)['ndcg_seed_sd']:.8f}",
                    f"{aggregate(v)['aplt_mean'] * 100:.6f}", f"{aggregate(v)['catalog_coverage_mean'] * 100:.4f}"] for i, v in enumerate(depth_variants, 1)])
    one = aggregate("lightgcn_l1")
    text += f"\n1층→3층에서 NDCG는 {pct_change(light['ndcg_mean'], one['ndcg_mean']):.2f}% 높고, APLT는 {abs(pct_change(light['aplt_mean'], one['aplt_mean'])):.2f}% 낮다. 다만 APLT는 2층→3층에서 조금 올라가므로 '깊이에 따라 단조 감소'한다고 쓰면 안 된다. 층별 동일 LR/정규화를 사용하고 각 실행의 best epoch는 valid로 선택했다.\n\n![층 수 비교](results/2026-11-07/depth_comparison.png)\n\n"
    text += "## 대조 손실 제거 보조 실험: 아직 미완료\n\n"
    cl0 = by_variant.get("xsimgcl_cl0", [])
    matched = sorted(r["settings"]["seed"] for r in cl0)
    paired_x = [r for r in by_variant["xsimgcl"] if r["settings"]["seed"] in matched]
    m0 = st.mean(value(r["test_metrics"]["all"], "ndcg") for r in cl0)
    mx = st.mean(value(r["test_metrics"]["all"], "ndcg") for r in paired_x)
    text += table(["완료된 동일 seed만 비교", "seed", "NDCG 평균", "APLT (%)"],
                  [["XSimGCL CL=0", str(matched), f"{m0:.8f}", f"{st.mean(value(r['test_metrics']['all'], 'aplt') for r in cl0) * 100:.8f}"],
                   ["XSimGCL CL=0.1", str(matched), f"{mx:.8f}", f"{st.mean(value(r['test_metrics']['all'], 'aplt') for r in paired_x) * 100:.8f}"]])
    text += f"\n완료된 동일 seed {matched}만 비교하면 CL 사용 시 NDCG가 {pct_change(mx, m0):.2f}% 높지만 tail 노출은 낮다. **이 표는 2-seed 중간 결과이며 3-seed 최종 결론이 아니다.** CL=0 두 실행은 모두 100 epoch 상한에 도달했다(best epoch: {[r['best_epoch'] for r in cl0]}). 충분한 수렴 여부가 미확정이므로 '대조학습이 항상 우월하다'고 일반화하지 않는다. encoder/noise와 나머지 설정을 유지한 비교지만 원 논문의 모든 recipe를 재현한 것은 아니다.\n\n"
    text += "## seed별 결과와 학습 시간\n\n"
    text += table(["모델", "seed", "test NDCG", "종료 epoch", "best epoch", "기록된 GPU 학습(분)"],
                  [[NAMES[s['variant']], s['seed'], f"{s['ndcg']:.8f}", s['completed_epochs'], s['best_epoch'], f"{s['recorded_training_minutes']:.1f}"] for s in seed_rows if s['variant'] in NAMES])
    text += "\n학습 시간은 history의 epoch training_seconds 합계다. 검증·Drive 전송·재시도·유실 후 재계산·test 평가를 포함한 전체 소요 시간이 아니다. 재개 시 summary.elapsed_seconds는 구간 기준일 수 있어 캠페인 전체 시간으로 합산하지 않았다.\n\n"
    text += table(["모델", "LR", "L2 계수", "층 수", "CL 계수"],
                  [[NAMES[v], selection['choices'][v]['learning_rate'], selection['choices'][v]['regularization'],
                    "미사용" if v == 'bpr_mf' else selection['choices'][v]['layers'],
                    selection['choices'][v]['cl_weight'] if v == 'xsimgcl' else "미사용"] for v in variants[1:]])
    text += "\n공통 dim=64, batch=8192, fused Adam, FP32. XSimGCL eps=0.1, temperature=0.2. valid를 1 epoch와 이후 매 5 epoch 평가하고, 8회 연속 개선이 없으면 조기 종료했다. 설정 고정 후 seed 42/43/44 test를 평가했다.\n\n"
    text += "## 범위·검증·남은 일\n\n"
    text += "- 전달받은 전처리 5-core, train만 학습. train 4,818,321개, 사용자 599,742명, 상품 248,186개다. test는 warm-start 158,805명/333,797개 정답이며 원래 test 상호작용의 약 55.2%다.\n"
    text += "- 전체 학습 상품에서 seen_for_test를 제외하고 평가했다. 사용자별 macro 평균이며 sampled-negative 점수가 아니다. test로 설정·epoch를 고르지 않았다.\n"
    text += f"- 완료된 {len(records)}개 final 기록의 입력 fingerprint·plan hash·frozen 설정·연속 epoch history·test cohort·그룹 가중 집계 일치를 검사했다. 주 비교 9개가 모두 학습 seed와 test 평가를 갖춘 것을 확인했다.\n"
    text += f"- XSimGCL 후보 Recall@100의 사용자 평균은 {x['candidate_recall100_mean'] * 100:.3f}%다. 절대 정확도와 tail 정답 회수율은 낮고, 후보 100개 안에서만 재정렬하는 방식에도 누락 한계가 있다.\n"
    text += "- 모든 주 모델의 Tail Recall@10은 사실상 0이다. LightGCN도 3-seed 평균이 매우 작아 비인기 노출 증가를 비인기 정답 추천 성공으로 바꿔 말하면 안 된다.\n"
    text += "- XSimGCL은 공통 raw-embedding squared L2를 적용한 recipe variant다. LightGCN과의 차이를 대조 손실 하나의 인과 효과로 단정하지 않는다. 제한된 4개 설정/모델 coarse 탐색 결과이며 SOTA나 최적 튜닝을 주장하지 않는다.\n"
    text += "- 사용자별 paired bootstrap 95% CI는 아직 전체 캠페인의 최종 자동 보고서 생성 전이다. 여기서는 통계적 유의성을 선언하지 않는다. seed SD와 사용자 CI를 혼동하지 않는다.\n"
    text += "- 텍스트·차종 특징 미제공으로 재정렬·차종 Mismatch는 미실시다. 미계산 값을 0으로 쓰지 않았다.\n"
    text += f"- 비용 관측: {cost['time']}에 계정 잔액 감소 {cost['account_delta_cu']:.2f} CU, 잔액 {cost['current_balance_cu']:.2f} CU. 다른 동시 작업/준비 구간을 포함할 수 있어 모델별 정확한 CU 비용으로 배분하지 않는다.\n"
    text += "- 남은 작업: CL=0 seed 44 완료 → test 평가 → 3-seed 보조 실험 집계·paired CI·자동 최종 보고서 → GPU 반납. 현재 기록 기준 약 2–3시간 예상이나 Colab 재할당·전송 대기로 늘어날 수 있다. 본 문서의 주 모델 확정 결과는 마지막 보조 실험 때문에 다시 학습하지 않는다.\n\n"
    text += "## 발표에 쓸 결론\n\n> 동일한 warm-start 전체 상품 평가에서 XSimGCL이 가장 높은 정확도를 기록했고, LightGCN은 정확도와 비인기 상품 노출을 함께 고려할 때 유용한 비교 대상이었다. 그러나 XSimGCL의 비인기 노출은 거의 0으로, 그래프 대조학습이 인기 편향을 완화한다는 가설은 본 데이터와 설정에서 지지되지 않았다.\n\n"
    text += "주 비교 원본 CSV: [main_comparison.csv](results/2026-11-07/main_comparison.csv). [그룹별](results/2026-11-07/group_comparison.csv), [seed별](results/2026-11-07/completed_seed_results.csv), [층별](results/2026-11-07/depth_comparison.csv), [검증 기록](results/2026-11-07/verification.json).\n\n"
    text += f"동결 선택 시각: `{selection['frozen_at']}`. 모델 소스 revision: `{plan['metadata']['git_revision']}`. plan SHA-256: `{plan_hash}`. raw 모델/추천/사용자별 출력은 Git에 넣지 않고 캠페인 Drive에 보관한다.\n"
    args.output.write_text(text)
    print(json.dumps(verified, indent=2))
    print(args.output)


if __name__ == "__main__": main()
