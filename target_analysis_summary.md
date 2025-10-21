# Target Range Analysis for EEG Challenge 2025

## Challenge 1: Response Time (rt_from_stimulus)

### Dataset Information
- **Total samples**: 15,038
  - Train: 10,446
  - Validation: 2,299
  - Test: 2,293

### Statistical Summary (All data)
```
count    15038.000000
mean         1.595800
std          0.380494
min          0.100000
25%          1.370000
50%          1.610000
75%          1.858000
max          2.420000
```

### Target Range
- **Min**: 0.1000 seconds
- **Max**: 2.4200 seconds
- **Range**: 2.3200 seconds

### Split Statistics
| Split | Mean | Std Dev |
|-------|------|---------|
| Train | 1.6142 | 0.3746 |
| Val   | 1.5394 | 0.4080 |
| Test  | 1.5684 | 0.3711 |

### Observations
- The response times range from 0.1 to 2.42 seconds
- The distribution is fairly centered around 1.6 seconds (median: 1.61s)
- There is good consistency across splits (similar means and standard deviations)
- Standard deviation of ~0.38 seconds indicates moderate variability in response times

---

## Challenge 2: Externalizing Factor (p_factor)

*Analysis in progress... This requires loading subject-level metadata from the EEG Challenge Dataset.*

To run the Challenge 2 analysis:
```bash
python3 challenge2_target_analysis.py
```

---

## Notes

- Challenge 1 focuses on **regression**: predicting continuous response time values
- Challenge 2 focuses on **regression of behavioral factors**: predicting the externalizing factor score
- Both challenges use EEG data from the contrast change detection task
- The targets have very different scales and meanings:
  - Challenge 1: Time-based measurements (0.1-2.4 seconds)
  - Challenge 2: Psychological factor scores (scale TBD)
