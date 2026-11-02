# Re-run results

1,191 creatives, dates 2014-10-21 to 2014-10-21.

## Original features (includes item_ctr_rank)

### Regression (CTR), 239-creative test set

|                          |    MAE |   RMSE |      R2 |
|:-------------------------|-------:|-------:|--------:|
| Baseline (training mean) | 0.108  | 0.1466 | -0.0003 |
| Linear Regression        | 0.0456 | 0.0724 |  0.756  |
| Random Forest            | 0.0019 | 0.0107 |  0.9947 |
| XGBoost                  | 0.0052 | 0.0192 |  0.9829 |
| XGBoost (tuned)          | 0.0061 | 0.0189 |  0.9834 |

### Classification (CTR above median)

|                           |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:--------------------------|-----------:|------------:|---------:|-------:|----------:|
| Baseline (majority class) |     0.4854 |      0      |   0      | 0      |    0.5    |
| Logistic Regression       |     0.9791 |      0.9836 |   0.9756 | 0.9796 |    0.9992 |
| Random Forest             |     1      |      1      |   1      | 1      |    1      |
| XGBoost                   |     0.9958 |      1      |   0.9919 | 0.9959 |    0.9982 |
| XGBoost (tuned)           |     0.9958 |      1      |   0.9919 | 0.9959 |    0.9939 |

### 5-fold CV, tuned XGBoost

- Regression RMSE: 0.0313 +/- 0.0116
- Regression R2: 0.9499 +/- 0.0332
- Classification ROC-AUC: 0.9976 +/- 0.0029
- Classification F1: 0.9932 +/- 0.0033

### Random forest feature importance (regression)

|                        |      0 |
|:-----------------------|-------:|
| item_ctr_rank          | 0.9723 |
| log_creatives_per_item | 0.0141 |
| banner_pos             | 0.0073 |
| log_pv                 | 0.0059 |
| device_type            | 0.0002 |
| is_app                 | 0.0001 |
| conn_type              | 0.0001 |
| dow                    | 0      |
| is_weekend             | 0      |

## Without item_ctr_rank

### Regression (CTR), 239-creative test set

|                          |    MAE |   RMSE |      R2 |
|:-------------------------|-------:|-------:|--------:|
| Baseline (training mean) | 0.108  | 0.1466 | -0.0003 |
| Linear Regression        | 0.1032 | 0.145  |  0.021  |
| Random Forest            | 0.1054 | 0.1485 | -0.0259 |
| XGBoost                  | 0.1048 | 0.1472 | -0.008  |
| XGBoost (tuned)          | 0.1044 | 0.1453 |  0.0174 |

### Classification (CTR above median)

|                           |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:--------------------------|-----------:|------------:|---------:|-------:|----------:|
| Baseline (majority class) |     0.4854 |      0      |   0      | 0      |    0.5    |
| Logistic Regression       |     0.5732 |      0.5827 |   0.6016 | 0.592  |    0.6044 |
| Random Forest             |     0.5774 |      0.5859 |   0.6098 | 0.5976 |    0.6211 |
| XGBoost                   |     0.5774 |      0.5965 |   0.5528 | 0.5738 |    0.6169 |
| XGBoost (tuned)           |     0.6025 |      0.6321 |   0.5447 | 0.5852 |    0.6251 |

### 5-fold CV, tuned XGBoost

- Regression RMSE: 0.1448 +/- 0.0024
- Regression R2: 0.0394 +/- 0.0169
- Classification ROC-AUC: 0.6616 +/- 0.0397
- Classification F1: 0.5857 +/- 0.0406

### Random forest feature importance (regression)

|                        |      0 |
|:-----------------------|-------:|
| log_pv                 | 0.6098 |
| device_type            | 0.1005 |
| conn_type              | 0.096  |
| log_creatives_per_item | 0.0889 |
| is_app                 | 0.0729 |
| banner_pos             | 0.0319 |
| dow                    | 0      |
| is_weekend             | 0      |
