import torch
from torch import nn

from starvla_lab.bench import HeadDropoutSchedule, compare_configs, measure_step_overhead, write_overhead_csv


def _make_step_fn(n_heads: int, d: int = 16, batch: int = 4):
    backbone = nn.Linear(d, d)
    heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_heads)])
    x = torch.randn(batch, d)

    def step(i: int) -> torch.Tensor:
        h = torch.relu(backbone(x))
        return sum(head(h).pow(2).mean() for head in heads)

    return step


def test_measure_returns_positive_timing():
    r = measure_step_overhead("one_head", _make_step_fn(1), batch_size=4, steps=3, warmup=1)
    assert r.steps == 3 and r.sec_per_step > 0 and r.samples_per_sec > 0
    assert r.peak_mem_mb == 0.0 and r.device == "cpu"


def test_compare_configs_and_csv(tmp_path):
    results = compare_configs({"h1": lambda: _make_step_fn(1), "h3": lambda: _make_step_fn(3)}, batch_size=4, steps=3, warmup=1)
    assert [r.name for r in results] == ["h1", "h3"]
    p = write_overhead_csv(results, tmp_path / "o.csv")
    text = p.read_text()
    assert text.splitlines()[0].startswith("name,steps,batch_size,sec_per_step")
    assert "h3" in text


def test_head_dropout_schedule_deterministic_and_bounded():
    a = HeadDropoutSchedule(["oft", "pi", "gr00t"], p_all=0.3, seed=1)
    b = HeadDropoutSchedule(["oft", "pi", "gr00t"], p_all=0.3, seed=1)
    seq_a = [a.active(i) for i in range(50)]
    seq_b = [b.active(i) for i in range(50)]
    assert seq_a == seq_b
    assert all(len(s) in (1, 3) for s in seq_a)
    assert any(len(s) == 1 for s in seq_a) and any(len(s) == 3 for s in seq_a)
    assert HeadDropoutSchedule(["oft"], p_all=1.0).active(0) == ("oft",)
    assert abs(a.expected_head_evals_per_step() - (0.3 * 3 + 0.7)) < 1e-9
