# Predictive Customer Lifetime Value: A Comparative Modeling Study

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-informational)](#synthetic-data-and-limitations)
[![Reproducible Study](https://img.shields.io/badge/Study-Reproducible-brightgreen)](#reproduce-the-experiment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible Python study comparing historical run-rate estimates, **BG/NBD**, **Pareto/NBD**, and machine learning for predicting future customer purchases and financial contribution. Two Google Colab notebooks generate synthetic customer transactions, evaluate forecasts on a future holdout period, and examine model assumptions, numerical stability, and performance across 20 additional simulated customer populations.

This repository supports the forthcoming *Marketing Data Science* article **"Predicting Customer Lifetime Value: What Will Your Customers Be Worth?"**, a follow-up to [**"Customer Lifetime Value: What Is a Customer Really Worth?"**](https://blog.marketingdatascience.ai/customer-lifetime-value-what-is-a-customer-really-worth-47cf0efc510c) The article link will be added after publication.

## Why this project?

A common Customer Lifetime Value (CLV) calculation multiplies annual customer contribution by expected lifespan. That is a helpful introduction, but it leaves a difficult question: **Which customers will return, how often will they purchase, and how much will they contribute?**

This study compares accessible baselines with probabilistic purchasing models and supervised machine learning. Rather than assuming a more sophisticated model will win, it evaluates forecasts at both the **individual-customer** and **whole-portfolio** levels and investigates *why* errors occur.

## Run in Google Colab

| Notebook | Purpose | Open |
|---|---|---|
| [01_generate_synthetic_clv_data.ipynb](01_generate_synthetic_clv_data.ipynb) | Simulate customers, purchases, contribution and latent dropout; validate and export the frozen dataset | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joedom99/predictive-clv-modeling/blob/main/01_generate_synthetic_clv_data.ipynb) |
| [02_predictive_clv_model_comparison.ipynb](02_predictive_clv_model_comparison.ipynb) | Fit models, compare 2025 forecasts, audit assumptions and numerical behavior, project future contribution, and run replications | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joedom99/predictive-clv-modeling/blob/main/02_predictive_clv_model_comparison.ipynb) |

## Data and experimental design

Notebook 1 uses a fixed primary seed (`20261012`) to generate **2,000 fictional customers** with different acquisition dates, purchasing rates, transaction sizes, contribution margins, and probabilities of becoming inactive. True customer dropout is tracked by the simulator but cannot be inferred perfectly from observed transactions.

| Window | Dates | Use |
|---|---|---|
| Calibration | January 2022 to December 2024 | Customer histories and model fitting |
| Holdout | January to December 2025 | Forecast evaluation, **not** model-training features |

The primary dataset contains **33,557 transactions**: 25,416 in calibration and 8,141 in the holdout. Those holdout transactions represent **7,952 customer-day purchase occasions** after same-day aggregation. No new customers are acquired during 2025: the forecast concerns **the existing customer base**, not a growing company's total sales.

| File | Contents | Role |
|---|---|---|
| `customer_profiles.csv` | Observable customer and acquisition attributes | Model input |
| `calibration_transactions.csv` | Purchases through December 2024 | Model input |
| `transactions.csv` | Full history, including 2025 | Holdout evaluation; exclude 2025 from model features |
| `simulation_ground_truth.csv` | Simulator-known latent behavior and dropout | **Evaluation/audit only** |
| `simulation_metadata.json` | Random seed, date boundaries and generation details | Reproducibility |

Notebook 2 also creates evaluation tables, numerical fit checks, diagnostic figures, and a manifest documenting the experiment settings. The outputs from the run reported in the article are saved in [`clv_model_results/`](clv_model_results/) (see [Saved results](#saved-results)).

## Forecasting approaches

| Model | Basic idea |
|---|---|
| **Run-rate (lifetime)** | Extrapolate observed purchasing over the customer's history |
| **Run-rate (last 365 days)** | Extrapolate more recent purchasing activity |
| **BG/NBD + Gamma-Gamma** | Estimate future purchase occasions with BG/NBD, then contribution per occasion |
| **Pareto/NBD + Gamma-Gamma** | Model purchases and independent customer dropout, then contribution per occasion |
| **Random forest** | Learn a future-contribution forecast from historical customer features |
| **Gradient boosting** | Learn a future-contribution forecast through boosted decision trees |

The probabilistic models use a common customer-day purchase-occasion definition. Gamma-Gamma is fitted to positive repeat-occasion monetary values. Notebook 2 uses unpenalized probabilistic fits and documents numerical safeguards and an independent check of the BG/NBD expected-purchases expression.

**Why include both BG/NBD and Pareto/NBD?** They make different assumptions about when customers can permanently stop buying. Our synthetic generator allows dropout independently of purchase events, which may favor Pareto/NBD. This alignment is part of the experimental design, not evidence that Pareto/NBD will outperform other methods on real businesses.

## What the experiment measures

The primary test forecasts **2025 contribution from the 2,000 existing customers**. The notebook evaluates:

- **Customer-level MAE and RMSE:** how far individual forecasts differ from realized contribution.
- **Signed portfolio forecast error:** whether total predicted contribution is too high or too low. A low portfolio error can conceal offsetting customer errors.
- **Purchase-occasion accuracy:** evaluated separately from raw transaction counts.
- **Segment and high-value-customer results:** how performance varies across customer groups.
- **Residuals and bootstrap intervals:** patterns of individual error and within-sample uncertainty.
- **P(alive) calibration:** an *evaluation-only* comparison with simulator-known customer survival; latent truth is not passed to the models.
- **Three-year discounted projections:** illustrative 2025 to 2027 forecasts for the existing portfolio. Only 2025 has observed outcomes in this experiment; 2026 and 2027 are **not independently validated forecasts**.

### Results from the primary holdout

| Model | Customer MAE | Signed portfolio error |
|---|---:|---:|
| Pareto/NBD + Gamma-Gamma | **$113.83** | **+1.03%** |
| BG/NBD + Gamma-Gamma | $119.14 | +11.73% |
| Random forest | $125.41 | +3.83% |
| Gradient boosting | $127.71 | +3.48% |
| Run-rate (last 365 days) | $153.76 | +40.71% |
| Run-rate (lifetime) | $227.11 | +90.41% |

*Results are from fictional data, not published industry benchmarks. Full precision and additional metrics are in [`clv_model_results/model_metrics_2025.csv`](clv_model_results/model_metrics_2025.csv).*

### Replication across 20 additional simulated customer populations

The model settings are held constant while the experiment is repeated with seeds **101 to 120**. The table below summarizes customer-level and portfolio errors across those simulated populations:

| Model | Median customer MAE | Median absolute portfolio error |
|---|---:|---:|
| Pareto/NBD + Gamma-Gamma | **$113.52** | 3.54% |
| Random forest | $119.22 | **2.86%** |
| BG/NBD + Gamma-Gamma | $120.20 | 14.66% |
| Gradient boosting | $125.48 | **2.86%** |
| Run-rate (last 365 days) | $152.47 | 41.82% |
| Run-rate (lifetime) | $224.20 | 91.90% |

Pareto/NBD + Gamma-Gamma has the lowest customer-level MAE in **19 of the 20** additional simulations. Random forest and gradient boosting are effectively tied for the lowest median portfolio error. That establishes consistency *within this simulator*, not superiority across real-world industries or different data-generating processes.

## Diagnostics and numerical safeguards

The notebooks document modeling decisions instead of hiding difficult cases:

- Validate customer acquisition dates, transaction chronology, and calibration/holdout separation.
- Distinguish observable purchases from simulator-known dropout.
- Check frequency, recency and observation age (`T`) for probabilistic inputs.
- Aggregate same-day transactions before fitting purchase-occasion models.
- Verify that fitted models yield finite, sensible forecasts; report numerical warnings and fit checks.
- Examine regularization sensitivity and Gamma-Gamma's monetary assumptions without choosing settings to minimize 2025 error.
- Use the simulated ground truth only in clearly labeled retrospective evaluation.
- Record package versions, fitting conventions, and replication seeds in `experiment_manifest.json`.

A model that fits is not necessarily a model that predicts well. Forecast validity, numerical stability, and predictive accuracy are related but separate questions.

## Reproduce the experiment

1. Download or clone the repository, or open **Notebook 1** directly in Google Colab.
2. Run all cells of Notebook 1 to recreate and export the primary synthetic datasets. You can also use the committed CSV/JSON files without regenerating them.
3. Open **Notebook 2** in Colab. When prompted, upload the Notebook 1 ZIP, or the four input files plus `simulation_ground_truth.csv` (needed for the P(alive) audit). The notebook installs the `lifetimes` package automatically if it is missing.
4. Select **Runtime → Run all** and review the model-fit diagnostics, holdout comparison, replications, and generated charts.
5. Keep the original calibration/holdout dates, seeds, and fixed modeling choices to reproduce the documented experiment.

Notebook 2 takes about 10 minutes on a standard Colab CPU runtime, most of it for the **20 replication runs**. To skip them, set `RUN_REPLICATION = False` in the settings cell.

## Saved results

The [`clv_model_results/`](clv_model_results/) folder contains the outputs from the run reported in the article, including model metrics, replication results, customer-level predictions, the P(alive) audit, the three-year projection, diagnostic figures, and `experiment_manifest.json`.

Rerunning Notebook 2 should reproduce the probabilistic model results exactly. Machine learning and replication results may differ slightly depending on the Python and library versions in your environment.

## Repository layout

```text
predictive-clv-modeling/
├── README.md
├── LICENSE
├── 01_generate_synthetic_clv_data.ipynb
├── 02_predictive_clv_model_comparison.ipynb
├── customer_profiles.csv
├── calibration_transactions.csv
├── transactions.csv
├── simulation_ground_truth.csv
├── simulation_metadata.json
└── clv_model_results/
    ├── model_metrics_2025.csv
    ├── replication_summary.csv
    ├── replication_metrics.csv
    ├── replication_fit_checks.csv
    ├── customer_predictions_2025.csv
    ├── p_alive_audit.csv
    ├── three_year_discounted_projection.csv
    ├── experiment_manifest.json
    ├── fig_*.png
    └── ... (additional diagnostic tables)
```

## Synthetic data and limitations

**This is an educational simulation, not a validation against a real company's customers.** The dataset was constructed to exhibit plausible heterogeneity and dropout, but it is not calibrated to any particular industry or business.

The experiment has important boundaries:

- **Model–simulator alignment:** Pareto/NBD shares a key dropout assumption with the generator, which can affect relative performance.
- **Limited external validity:** Twenty additional seeds repeat one data-generating design; they do not represent 20 independent industries or empirical datasets.
- **Development history:** The primary 2025 holdout was viewed during earlier notebook development. Its results should not be described as a pristine, never-examined test set. Later replications strengthen the within-simulator assessment but do not establish real-world generalizability.
- **Model assumptions:** Noncontractual churn is not directly observed; purchase frequency and monetary-value modeling can be misspecified or numerically sensitive.
- **Finance:** Contribution after variable costs is **not** net profit. Acquisition costs, fixed expenses, and the timing of future cash flows matter in business decisions.
- **Forecast horizons:** The three-year charts illustrate model-based projections; only the first year is evaluated against realized transactions.

The goal is not to declare one algorithm universally best. It is to show how **assumptions, data design, evaluation metrics, and the business question** determine whether a CLV forecast is useful.

## Related reading

- [**Customer Lifetime Value: What Is a Customer Really Worth?**](https://blog.marketingdatascience.ai/customer-lifetime-value-what-is-a-customer-really-worth-47cf0efc510c): introductory article in the *Marketing Data Science* series.
- **Predicting Customer Lifetime Value: What Will Your Customers Be Worth?**: companion article (link forthcoming).
- [Marketing Data Science](https://blog.marketingdatascience.ai/)

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Joe Domaleski.

## Author

**Joe Domaleski**  
[Marketing Data Science](https://blog.marketingdatascience.ai/) · [Country Fried Creative](https://countryfriedcreative.com/) · [GitHub](https://github.com/joedom99)

I write about practical applications of marketing, analytics, data science, machine learning, and artificial intelligence.
