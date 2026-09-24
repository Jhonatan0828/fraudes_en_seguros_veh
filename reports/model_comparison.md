# Comparación de modelos candidatos

Métrica de decisión: **pr_auc** (PR-AUC, apropiada con 5.99% de clase positiva).

| Modelo              |   pr_auc |   roc_auc |     f1 |   recall |   precision |   accuracy |   decision_threshold |   opt_f1 |   opt_recall |   opt_precision |
|:--------------------|---------:|----------:|-------:|---------:|------------:|-----------:|---------------------:|---------:|-------------:|----------------:|
| XGBoost             |   0.2411 |    0.83   | 0.2247 |   0.9135 |      0.1281 |     0.6219 |               0.6497 |   0.2891 |       0.6541 |          0.1856 |
| Random Forest       |   0.2024 |    0.8151 | 0.2281 |   0.8432 |      0.1319 |     0.6576 |               0.5995 |   0.2632 |       0.6054 |          0.1682 |
| Logistic Regression |   0.1696 |    0.8038 | 0.2289 |   0.8811 |      0.1316 |     0.644  |               0.6756 |   0.2482 |       0.5514 |          0.1601 |
| Logistic + SMOTE    |   0.1665 |    0.8016 | 0.2285 |   0.8324 |      0.1324 |     0.6628 |               0.6412 |   0.2475 |       0.6595 |          0.1523 |

## Modelo candidato

**XGBoost** — pr_auc = 0.2411, umbral de decisión = 0.6497
