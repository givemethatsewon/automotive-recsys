"""Predeclared validation selection and three-seed study; no test-driven tuning."""
import hashlib
import json


def plan_hash(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def default_plan(campaign_id, metadata=None):
    return {"campaign_id": campaign_id, "seeds": [42, 43, 44], "tuning_seed": 42,
            "tuning_epochs": 15, "final_epochs": 100, "eval_every": 5, "patience": 8,
            "planned_run_count": 32, "cu_budget": 100, "minimum_balance": 30,
            "candidate_top_n": 100, "tuning_metric": "valid/all/ndcg",
            "comparison": "warm-start all-catalog evaluation; train only; frozen validation selection before test",
            "pending_features": ["text reranking: feature tables not supplied", "vehicle reranking/mismatch: feature tables not supplied"],
            "metadata": metadata or {}}


def base_settings(model):
    return dict(model=model, variant=model, dim=64, layers=3, batch_size=8192,
                learning_rate=0.001, regularization=0.0001, cl_weight=0.1,
                eps=0.1, temperature=0.2, validation_batch=1024)


def tuning_jobs(plan):
    jobs = []
    for model in ["bpr_mf", "lightgcn", "xsimgcl"]:
        for lr in [0.001, 0.003]:
            for other in ([0.05, 0.1] if model == "xsimgcl" else [0.00001, 0.0001]):
                settings = base_settings(model) | {"learning_rate": lr}
                settings["cl_weight" if model == "xsimgcl" else "regularization"] = other
                jobs.append({"run_id": f"tune_{model}_{len(jobs):02d}", "purpose": "tuning", "settings": settings,
                             "seed": plan["tuning_seed"], "epochs": plan["tuning_epochs"], "patience": 0})
    return jobs


def depth_jobs(plan, best_light):
    return [{"run_id": f"tune_lightgcn_depth{layer}", "purpose": "depth_selection", "settings": best_light | {"layers": layer},
             "seed": plan["tuning_seed"], "epochs": plan["tuning_epochs"], "patience": 0} for layer in [1, 2]]


def freeze_choices(plan, base_choices, selected_light, selection_records):
    choices = dict(base_choices); choices["lightgcn"] = selected_light | {"variant": "lightgcn"}
    for layer in [1, 2, 3]:
        if layer != selected_light["layers"]:
            name = f"lightgcn_l{layer}"
            choices[name] = selected_light | {"variant": name, "layers": layer}
    choices["xsimgcl_cl0"] = choices["xsimgcl"] | {"variant": "xsimgcl_cl0", "cl_weight": 0.0}
    return {"plan_hash": plan_hash(plan), "selection_data": "valid only", "choices": choices,
            "selection_records": selection_records, "seeds": plan["seeds"], "test_used_for_selection": False}


def final_jobs(plan, selection):
    return [{"run_id": f"final_{variant}_seed{seed}", "purpose": "final_main" if variant in {"bpr_mf","lightgcn","xsimgcl"} else "final_ablation",
             "settings": settings, "seed": seed, "epochs": plan["final_epochs"], "patience": plan["patience"]}
            for variant, settings in selection["choices"].items() for seed in plan["seeds"]]
