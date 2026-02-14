# CMD-Fork

[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)]()
[![Validation](https://img.shields.io/badge/validation-10M%20tests-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()

**Decision simulation engine that survived 10 million extreme stress tests.**  
Tests A/B/C strategies against RR, RB, DC, and irreversibility.  
No magic numbers — thresholds discovered empirically.

---

## What It Is

CMD-Fork simulates how decisions perform under layered uncertainty. It tests decision strategies against combinations of:

- **RR** (Responsibility Radius) — How far consequences reach
- **RB** (Resource Buffer) — Cushion against failure
- **DC** (Damage Containment) — Ability to limit cascades
- **Irreversibility** — Whether decisions can be undone

---

## Why It's Different

| Feature | CMD-Fork | Others |
|--------|----------|--------|
| **Empirical thresholds** | ✅ Discovered from data | ❌ Hard-coded "magic numbers" |
| **Stress-tested** | ✅ 10M extreme scenarios | ❌ Limited validation |
| **No hidden constants** | ✅ Everything explicit | ❌ Often opaque |
| **Personality diversity** | ✅ 1000+ agent types | ❌ Single "rational" actor |
| **Graceful degradation** | ✅ Survives 95% killing combos | ❌ Crashes or NaNs |

---

## Validation in 3 Numbers

10,000,000 ← Decisions simulated
0.8% ← Survival rate under extreme stress (expected)
0.614 ← Crisis threshold (discovered empirically, not coded)


The engine didn't crash. It didn't produce NaNs. It just got sad and kept working.

---

## Quick Start

```python
from cmd_fork_clean_v2 import CMDForkWorkflow

# Run 1000 decisions with default config
workflow = CMDForkWorkflow(mode="pure_random", n_runs=1000)
workflow.run()
stats = workflow.summary_stats()

print(f"A: {stats['A_pct']*100:.1f}%")
print(f"B: {stats['B_pct']*100:.1f}%")
print(f"C: {stats['C_pct']*100:.1f}%")
