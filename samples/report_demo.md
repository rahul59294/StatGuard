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
| Shapiro-Wilk (A) | 0.971064159347552 | 0.609397687670936 | consistent with normality |
| Shapiro-Wilk (B) | 0.324800667725172 | 7.5801870371779e-11 | not consistent with normality |
| Levene (equal variance) | 1.37517857421885 | 0.245801764522923 | equal variance |

## Chosen test and reason

- **Name:** Mann-Whitney
- **Reason:** At least one group does not meet the normality check, so the non-parametric Mann-Whitney test is appropriate.

## Result

- **Statistic:** 44
- **p-value:** 3.37192303356722e-09
- **Effect size:** rank-biserial correlation = -0.898617511520737
- **Group sizes:** A=28; B=31

## Interpretation

The Mann-Whitney returned a p-value of 3.37192303356722e-09. The result provides evidence of difference between the groups at the 0.05 level.

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
