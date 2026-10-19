## Dataset summary

| Column | Type | Missing | Outliers |
| --- | --- | ---: | ---: |
| group | categorical | 0 | 0 |
| score | numeric | 2 | 1 |
| category | categorical | 1 | 0 |

**Duplicate rows:** 1

## Assumption checks

| Check | Statistic | p-value | Verdict |
| --- | ---: | ---: | --- |
| Shapiro-Wilk (A) | 0.9711 | 0.6094 | consistent with normality |
| Shapiro-Wilk (B) | 0.3248 | 7.580e-11 | not consistent with normality |
| Levene (equal variance) | 1.375 | 0.2458 | equal variance |

## Chosen test and reason

- **Name:** Mann-Whitney
- **Reason:** At least one group does not meet the normality check, so the non-parametric Mann-Whitney test is appropriate.

## Result

- **Statistic:** 44
- **p-value:** 3.372e-09
- **Effect size:** rank-biserial correlation = -0.8986
- **Group sizes:** A=28; B=31

## Interpretation

The observed difference between the groups was a higher median in group B (84.2) than group A (73), with a p-value of 3.372e-09. The rank-biserial correlation was -0.8986, indicating a large effect, and a small p-value alone does not show practical importance.

## Reproduce

```python
import pandas as pd
import scipy

frame = pd.read_csv('samples/demo.csv')
clean = frame[['group', 'score']].dropna()
groups = [clean.loc[clean['group'] == label, 'score'].astype(float).to_numpy() for label in pd.unique(clean['group'])]
group_1, group_2 = groups
result = scipy.stats.mannwhitneyu(group_1, group_2, alternative='two-sided')
print(result.pvalue)
```

CLI command: `statguard analyze samples/demo.csv --group group --value score`
