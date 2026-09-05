"""Benchmark protocols: backbone-only comparison (WP5) and training overhead (WP6)."""
from .backbone_bench import (
    BackboneSpec,
    BenchmarkSpec,
    Protocol,
    RunSpec,
    build_runs,
    format_summary_table,
    read_matrix_csv,
    render_commands,
    summarize_results,
    total_gpu_hours,
    varying_keys,
    write_matrix_csv,
)
from .overhead_bench import HeadDropoutSchedule, OverheadResult, compare_configs, measure_step_overhead, write_overhead_csv

__all__ = [
    "BackboneSpec", "BenchmarkSpec", "Protocol", "RunSpec", "build_runs", "format_summary_table",
    "read_matrix_csv", "render_commands", "summarize_results", "total_gpu_hours", "varying_keys", "write_matrix_csv",
    "HeadDropoutSchedule", "OverheadResult", "compare_configs", "measure_step_overhead", "write_overhead_csv",
]
