from pathlib import Path

from synpath.metrics.authenticity_backfill import run_backfill


if __name__ == "__main__":
    report = run_backfill(
        experiment_roots=[
            Path("/raid/dm236/synpath"),
            Path("/raid/dm236/synpath_balanced_global_clusters"),
        ],
        main_root=Path("/raid/dm236/synpath"),
        include_audits=True,
        gpu_ids=list(range(8)),
        batch_size=1024,
        report_path=Path(
            "/raid/dm236/synpath/reports/authenticity/"
            "authenticity_results.csv"
        ),
        resume=True,
    )
    print(report)
