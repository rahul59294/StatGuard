## Dataset summary

| Column | Type | Missing | Outliers |
| --- | --- | ---: | ---: |
| region | categorical | 0 | 0 |
| preference | categorical | 0 | 0 |

**Duplicate rows:** 14

## Assumption checks

| Check | Statistic | p-value | Verdict |
| --- | ---: | ---: | --- |
| Contingency table | 2 x 2 (18 complete rows) | n/a | usable |
| Expected counts | minimum expected count = 1 | n/a | 2 expected count(s) < 5 |

## Chosen test and reason

- **Name:** Fisher
- **Reason:** The contingency table is 2x2 and at least one expected count is below 5, so Fisher's exact test is appropriate.

## Result

- **Statistic:** 1
- **p-value:** 1
- **Effect size:** Cramér's V = 0
- **Group sizes:** region=North=9; South=9; preference=Tea=16; Coffee=2

## Interpretation

The Fisher returned a p-value of 1. The result does not provide sufficient evidence of association between the categorical variables at the 0.05 level.

## Reproduce

```python
import pandas as pd
import scipy

frame = pd.read_csv('samples/demo_cat.csv')
clean = frame[['region', 'preference']].dropna()
contingency_table = pd.crosstab(clean['region'], clean['preference'])
result = scipy.stats.fisher_exact(contingency_table)
print(result.pvalue)
```

CLI command: `statguard analyze samples/demo_cat.csv --cat1 region --cat2 preference`
