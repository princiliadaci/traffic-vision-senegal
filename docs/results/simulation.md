# Simulation study: results

1000 simulated studies per scenario; true effect +12 vehicles per 5 min; detector recall 82%, 1 false detection per interval; 8 intervals counted by hand. Produced by `scripts/simulate_evaluation.py`.

## 2-hour recordings, autocorrelated traffic (rho = 0.6)

| estimator | mean estimate | bias | RMSE | 95% CI coverage | mean CI width | CI excludes 0 |
|---|---|---|---|---|---|---|
| E1. Before/after, treated street only | 14.8 | +2.8 | 6.8 | 69.5% | 13.8 | 90% |
| E2. DiD, raw counts, ordinary bootstrap | 10.1 | -1.9 | 8.2 | 70.3% | 17.8 | 56% |
| E3. DiD, calibrated counts, ordinary bootstrap | 12.0 | +0.0 | 9.5 | 73.1% | 21.3 | 56% |
| E4. DiD, calibrated counts, block bootstrap | 12.0 | +0.0 | 9.5 | 81.4% | 25.5 | 48% |
| E5. DiD, calibrated counts, batch means (default) | 12.0 | +0.0 | 9.5 | 92.7% | 37.2 | 27% |

## 6-hour recordings, autocorrelated traffic (rho = 0.6)

| estimator | mean estimate | bias | RMSE | 95% CI coverage | mean CI width | CI excludes 0 |
|---|---|---|---|---|---|---|
| E1. Before/after, treated street only | 14.8 | +2.8 | 4.7 | 60.6% | 8.4 | 100% |
| E2. DiD, raw counts, ordinary bootstrap | 10.0 | -2.0 | 5.2 | 70.3% | 10.8 | 83% |
| E3. DiD, calibrated counts, ordinary bootstrap | 12.0 | +0.0 | 5.8 | 72.6% | 13.0 | 83% |
| E4. DiD, calibrated counts, block bootstrap | 12.0 | +0.0 | 5.8 | 85.2% | 17.1 | 74% |
| E5. DiD, calibrated counts, batch means (default) | 12.0 | +0.0 | 5.8 | 94.0% | 24.6 | 49% |

## 2-hour recordings, independent intervals (rho = 0)

| estimator | mean estimate | bias | RMSE | 95% CI coverage | mean CI width | CI excludes 0 |
|---|---|---|---|---|---|---|
| E1. Before/after, treated street only | 14.8 | +2.8 | 4.2 | 86.1% | 12.6 | 100% |
| E2. DiD, raw counts, ordinary bootstrap | 9.8 | -2.2 | 4.7 | 93.0% | 16.3 | 65% |
| E3. DiD, calibrated counts, ordinary bootstrap | 11.7 | -0.3 | 5.0 | 94.4% | 19.5 | 65% |
| E4. DiD, calibrated counts, block bootstrap | 11.7 | -0.3 | 5.0 | 92.9% | 18.7 | 68% |
| E5. DiD, calibrated counts, batch means (default) | 11.7 | -0.3 | 5.0 | 96.1% | 23.4 | 49% |

## 6-hour recordings, independent intervals (rho = 0)

| estimator | mean estimate | bias | RMSE | 95% CI coverage | mean CI width | CI excludes 0 |
|---|---|---|---|---|---|---|
| E1. Before/after, treated street only | 14.7 | +2.7 | 3.3 | 70.2% | 7.4 | 100% |
| E2. DiD, raw counts, ordinary bootstrap | 9.8 | -2.2 | 3.3 | 84.0% | 9.6 | 97% |
| E3. DiD, calibrated counts, ordinary bootstrap | 11.7 | -0.3 | 3.0 | 94.1% | 11.5 | 97% |
| E4. DiD, calibrated counts, block bootstrap | 11.7 | -0.3 | 3.0 | 93.3% | 11.2 | 98% |
| E5. DiD, calibrated counts, batch means (default) | 11.7 | -0.3 | 3.0 | 95.3% | 13.2 | 94% |

![Coverage of the 95% intervals](simulation_coverage.png)
