"""One real, full-data epoch per model; exclude large pilot weights from download."""
import json
from pathlib import Path
import sys
import tarfile

root = Path("/content/automotive-recsys")
sys.path.insert(0, str(root / "src"))
from automotive_recsys.audit import write_json
from automotive_recsys.train_gpu import run_training


def main():
    output = root / "results/train_pilot"
    reports = []
    for model in ["bpr_mf", "lightgcn", "xsimgcl"]:
        report = run_training(root / "data", output / model, model,
                              epochs=1, eval_every=1, verify_resume=True,
                              save_recommendations=False)
        assert report["resume_check"]["passed"] and not report["test_evaluated"]
        reports.append(report)
    summary = {"passed": True, "purpose": "full epoch integration check, not tuned/converged performance comparison", "models": reports}
    write_json(output / "summary.json", summary)
    with tarfile.open("/content/automotive-recsys-pilot-results.tar.gz", "w:gz") as bundle:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file.suffix in {".json", ".parquet"}:
                bundle.add(file, arcname=str(Path("train_pilot") / file.relative_to(output)))
    print("Full-epoch training, complete validation, checkpoint save/load and deterministic optimizer/RNG resume checks passed for all three models.", flush=True)


if __name__ == "__main__":
    main()
