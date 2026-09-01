# Refusal-Threshold Calibration

The threshold that decides *answer* versus *I don't know*, measured rather
than guessed. Produced with **zero LLM calls** -- it operates purely on
cross-encoder relevance scores, which is why the gate sits before generation.

## Confidence distributions

| Group | n | min | median | max |
|---|---|---|---|---|
| Answerable (should answer) | 51 | 0.0 | 0.9617 | 0.9999 |
| Unanswerable (should refuse) | 9 | 0.0 | 0.0 | 0.0104 |

A clean separation between the two medians is what makes a threshold
meaningful at all. Overlap between the distributions is the irreducible
error any single threshold must trade against.

## Recommended threshold: **0.02**

| Metric | Value |
|---|---|
| Answer correctness (answerable correctly answered) | 0.7843 |
| Refusal correctness (unanswerable correctly refused) | 1.0 |
| Balanced accuracy | 0.8922 |
| F1 (answer as positive class) | 0.8791 |
| False answers on out-of-corpus questions | 0 |
| Wrongly refused answerable questions | 11 |

Balanced accuracy is the selection criterion rather than raw F1: with
51 answerable against 9 unanswerable,
F1 is dominated by the majority class and would pick a threshold that
essentially never refuses.

## Trade-off curve

| τ | answer correctness | refusal correctness | balanced | F1 |
|---|---|---|---|---|
| 0.05 | 0.745 | 1.000 | 0.873 | 0.854 |
| 0.10 | 0.726 | 1.000 | 0.863 | 0.841 |
| 0.15 | 0.726 | 1.000 | 0.863 | 0.841 |
| 0.20 | 0.726 | 1.000 | 0.863 | 0.841 |
| 0.25 | 0.726 | 1.000 | 0.863 | 0.841 |
| 0.30 | 0.706 | 1.000 | 0.853 | 0.828 |
| 0.35 | 0.706 | 1.000 | 0.853 | 0.828 |
| 0.40 | 0.706 | 1.000 | 0.853 | 0.828 |
| 0.45 | 0.706 | 1.000 | 0.853 | 0.828 |
| 0.50 | 0.706 | 1.000 | 0.853 | 0.828 |
| 0.55 | 0.706 | 1.000 | 0.853 | 0.828 |
| 0.60 | 0.667 | 1.000 | 0.833 | 0.800 |
| 0.65 | 0.667 | 1.000 | 0.833 | 0.800 |
| 0.70 | 0.667 | 1.000 | 0.833 | 0.800 |
| 0.75 | 0.667 | 1.000 | 0.833 | 0.800 |
| 0.80 | 0.647 | 1.000 | 0.824 | 0.786 |
| 0.85 | 0.627 | 1.000 | 0.814 | 0.771 |

**How to read this.** Raising τ makes the system more cautious: refusal
correctness rises while answer correctness falls. The recommended value is
the point where the two are best balanced. A deployment with a lower
tolerance for wrong answers than for unhelpful ones should deliberately
choose a higher τ -- the curve makes that a documented decision rather than
an accident.
