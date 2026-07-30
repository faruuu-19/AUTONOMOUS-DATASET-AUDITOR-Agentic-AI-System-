# Benchmark Results

Each dataset audited twice: agent-selected strategy vs. a forced-all-tools
baseline. Ground truth is the defect deliberately injected by
`generate_test_datasets.py`.

| Dataset | Rows | Known defect | Expected detector fired | Verdict | Score | Tools run | Skipped | Runtime |
|---|---|---|---|---|---|---|---|---|
| Injected leakage | 1,000 | 'will_default' leaks the target | PASS 1/1 | NOT READY | 71 | 1/5 | 4 | 0.1s |
| Class imbalance | 2,000 | 97% / 3% class split | PASS 1/1 | READY | 97 | 2/5 | 3 | 0.1s |
| Multiple issues | 1,600 | imbalance + 'churn_probability' leak + duplicate rows | PARTIAL 2/3 | NOT READY | 65 | 2/5 | 3 | 0.1s |
| Complex / large | 12,000 | 8% positive class + derived 'future_flag' leak | PARTIAL 1/2 | READY | 94 | 2/5 | 3 | 2.3s |
| Clean control | 500 | none (control) | n/a (control, clean) | READY | 100 | 3/5 | 2 | 0.1s |

## Classification metrics

Unit of classification is one (dataset x detector) pair, 25 in total. Ground truth positive means the dataset carries a defect that detector is meant to catch; predicted positive means it raised a critical or warning finding.

| Metric | Agent mode | All-tools baseline |
|---|---|---|
| Precision | 1.000 | 0.538 |
| Recall | 0.714 | 1.000 |
| F1 | 0.833 | 0.700 |
| False positive rate | 0.000 | 0.333 |
| False negative rate | 0.286 | 0.000 |
| Accuracy | 0.920 | 0.760 |
| TP / FP / FN / TN | 5 / 0 / 2 / 18 | 7 / 6 / 0 / 12 |

The baseline column isolates detector quality (every tool always runs).
The agent column is end-to-end performance, so any gap between them is
the cost of the agent's decision to skip tools.

## Aggregate

```
Detection recall            5/7 expected detectors fired   (71.4%)
False positives (controls)  0 critical findings across 1 clean dataset(s)
Tool-selection skip rate    60.0% of checks skipped
Critical retention          9.7% vs forced-all-tools baseline
Median runtime              0.1s  (range 0.1s - 2.3s, up to 12,000 rows)
```

## How these were measured

- **Skip rate** = `tools_skipped / 5`, averaged across datasets.
- **Critical retention** = criticals found by the agent divided by criticals
  found when all five tools are forced to run. 100% means the agent's
  skipping cost nothing.
- **Detection recall** counts an expected detector as successful when it
  produced at least one finding in agent mode.
- **False positives** are critical findings on datasets generated with no
  injected defects.
