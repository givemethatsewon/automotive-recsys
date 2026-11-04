import argparse
import json
from pathlib import Path

from .audit import audit, write_json
from .popularity import run_popularity


def main():
    parser = argparse.ArgumentParser(description="Automotive 5-core experiment tools")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="Read-only data contract checks")
    audit_parser.add_argument("--output", type=Path, default=Path("results/audit.json"))
    pop_parser = sub.add_parser("popularity", help="Exact non-personalized reference")
    pop_parser.add_argument("--split", choices=["valid", "test"], default="valid")
    pop_parser.add_argument("--k", type=int, default=10)
    pop_parser.add_argument("--save-top-n", type=int, default=100)
    pop_parser.add_argument("--limit-users", type=int, help="Diagnostic prefix only, not a benchmark")
    pop_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.data_dir)
        write_json(args.output, result)
        print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]), "output": str(args.output)}, indent=2))
        if not result["passed"]:
            for check in result["checks"]:
                if not check["passed"]:
                    print(json.dumps(check, ensure_ascii=False))
            raise SystemExit(1)
    else:
        output = args.output or Path("results") / f"popularity_{args.split}"
        result = run_popularity(args.data_dir, output, args.split, args.k, args.save_top_n, args.limit_users)
        print(json.dumps({key: result[key] for key in ["model", "split", "evaluated_users", "metrics", "elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
