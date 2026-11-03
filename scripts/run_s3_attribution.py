"""S3 支撑层归因实验入口；配置、冻结、续跑均复用统一调度器。"""
from pathlib import Path

from run_iemoeo_mechanisms import main


if __name__ == "__main__":
    raise SystemExit(main(default_plan=Path(__file__).resolve().parents[1] / "experiments" / "iemoeo_s3_attribution_v1.json",
                          default_suite="screen"))
