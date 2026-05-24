from __future__ import annotations

import argparse

import run_aggregate
import run_assign
import run_bioemu
import run_generate
import run_plots


STAGES = {
    "generate": run_generate.main,
    "bioemu": run_bioemu.main,
    "assign": run_assign.main,
    "aggregate": run_aggregate.main,
    "plots": run_plots.main,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", default="generate,bioemu,assign,aggregate,plots")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for stage in [x.strip() for x in args.stages.split(",") if x.strip()]:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {', '.join(STAGES)}")
        stage_args = ["--config", args.config]
        if args.overwrite:
            stage_args.append("--overwrite")
        print(f"=== Running stage: {stage} ===")
        status = STAGES[stage](stage_args)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

