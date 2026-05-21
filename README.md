[README.md](https://github.com/user-attachments/files/27404731/README.md)
# Economic Prosperity and Football Success: Turkey & England (1993–2024)

**DSA 210 – Introduction to Data Science | Spring 2026**
**Muhammed Mustafa Güneş – 34313**

** Website : ** https://muhammedgunes54.github.io/DSA-210-Project/

---

## Project Overview

This project investigates whether a country's macroeconomic performance correlates with its football clubs' UEFA results and its national team's FIFA ranking. The study was expanded beyond the original Turkey-only scope to include England's Premier League (Big 6) and the England national team, enabling a cross-country structural comparison.

**Hypothesis H₁:** Stronger economic conditions are positively associated with better international football performance (ρ > 0).

---

## Dataset & Resolution

| | Turkey | England |
|---|---|---|
| **Timeframe** | 1993 – 2024 (31 years) | 1993 – 2024 (31 years) |
| **Data points** | 64 half-year periods | 64 half-year periods |
| **Resolution** | Bi-annual (H1: Jan–Jun, H2: Jul–Dec) | Bi-annual (H1: Jan–Jun, H2: Jul–Dec) |
| **Exchange rate** | USD/TRY (TCMB EVDS) | GBP/USD (Bank of England) |
| **Economic indicators** | USD/TRY, GDP growth %, CPI inflation % | GBP/USD, GDP growth %, CPI inflation % |
| **Club coverage** | Galatasaray, Fenerbahçe, Beşiktaş, Trabzonspor | Man City, Man Utd, Arsenal, Liverpool, Tottenham, Chelsea |
| **National team** | Turkey FIFA ranking | England FIFA ranking |

**Combined ML dataset: 128 rows** (Turkey + England merged), which substantially increases statistical power for machine learning compared to the original 64-row dataset.

### Data Sources
- **USD/TRY**: Central Bank of Turkey (TCMB EVDS) – evds2.tcmb.gov.tr
- **GBP/USD**: Bank of England – bankofengland.co.uk
- **GDP growth & CPI**: World Bank – data.worldbank.org
- **UEFA coefficients**: UEFA.com / rsssf.org / kassiesa.net/uefa/coeff
- **FIFA rankings**: FIFA.com historical archive

### Data Files
| File | Description |
|---|---|
| `master_data_v2.xlsx` | Main dataset — 4 sheets: `turkey_data`, `england_data`, `combined`, `README` |
| `analysis_v3.py` | Full analysis script (Google Colab compatible) |

---

## Methodology

1. **Exploratory Data Analysis (EDA)** — Time-series visualisations of exchange rates, UEFA points per club, and FIFA success indices for both countries.
2. **Composite Economic Index** — Three indicators (exchange rate, GDP growth, CPI inflation) normalised to [0,1] and averaged into a single `econ_index` per period.
3. **Hypothesis Testing** — Pearson correlation coefficient (one-tailed, α = 0.05) with Spearman ρ as a non-parametric backup. 95% confidence intervals via Fisher z-transform.
4. **Machine Learning**
   - K-Means clustering (k=3, elbow method) to discover natural economic-football eras
   - PCA for 2-D cluster visualisation
   - Supervised classification (Logistic Regression, Random Forest, Gradient Boosting) with 5-fold stratified cross-validation
   - OLS regression to quantify economic predictors of club success

---

## Key Findings

### Hypothesis Testing
| Country | Economic Index × Club UEFA | Economic Index × National Team |
|---|---|---|
| **Turkey** | r = +0.225, p = 0.037 ✅ **H₀ rejected** | r = +0.166, p = 0.096 ❌ not significant |
| **England** | r = −0.398, p = 0.999 ❌ not significant | r = +0.240, p = 0.028 ✅ **H₀ rejected** |

**Turkey — Club Level:** A statistically significant positive correlation was found (p < 0.05). Economic strength associates with better UEFA performance, likely because a stronger TRY increases clubs' purchasing power for foreign players and reduces debt servicing costs.

**Turkey — National Team:** No statistically significant correlation. Performance appears driven by generational talent and domestic player pools rather than immediate economic conditions.

**England — Club Level:** A negative (though not significant) correlation was observed. Premier League clubs derive the majority of their revenue from global broadcast rights and foreign ownership, making their UEFA performance largely independent of — and sometimes inversely related to — the UK domestic economy. A strong GBP reduces pound-denominated value of foreign-currency revenues.

**England — National Team:** A small but significant positive correlation (r = +0.240). England's national team performance shows a modest link to economic conditions, possibly through youth academy investment.

### Machine Learning Results (5-fold stratified CV)
| Model | Turkey Accuracy | England Accuracy | Combined Accuracy |
|---|---|---|---|
| Logistic Regression | 0.674 | 0.806 | 0.851 |
| Random Forest | **0.688** | **0.860** | **0.907** |
| Gradient Boosting | 0.688 | 0.845 | 0.891 |
| **Country prediction (TR vs ENG)** | — | — | **1.000** |

The combined 128-row dataset yields 90.7% accuracy in predicting high vs. low club success from economic indicators alone, substantially outperforming both individual country models. Perfect country classification (100%) confirms that the two economies are structurally distinct in the feature space.

### OLS Regression
| Model | R² | Adj. R² | F-statistic | p(F) |
|---|---|---|---|---|
| Turkey — Club Success | 0.609 | 0.589 | 31.13 | < 0.001 |
| Turkey — National Team | 0.059 | 0.012 | 1.26 | 0.296 |
| England — Club Success | 0.160 | 0.118 | 3.82 | 0.014 |
| England — National Team | 0.081 | 0.035 | 1.77 | 0.163 |

Economic variables explain 60.9% of the variance in Turkish club UEFA performance, a strong result for macroeconomic predictors in sport.

---

## Limitations

1. **Sample size**: N = 64 per country is below the ideal for machine learning. The combined N = 128 partially addresses this (noted by course instructor).
2. **Annual granularity for GDP/CPI**: Both indicators are annual; the same value is assigned to both half-year periods within a year, reducing temporal resolution.
3. **Approximate UEFA points**: Per-club half-year UEFA points are estimates reconstructed from archival data; exact values should be verified against primary UEFA sources.
4. **Confounders not captured**: Manager changes, transfer policy, injuries, and ownership changes are not included in the model.
5. **England ownership effect**: Abu Dhabi's acquisition of Man City (2008) and similar foreign investments create a structural break not captured by UK economic indicators alone.

---

## How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/MuhammedGunes54/DSA-210-Project.git

# 2. Install dependencies
pip install pandas openpyxl scikit-learn matplotlib seaborn scipy

# 3. Open Google Colab and upload master_data_v2.xlsx
#    (or run locally — the script auto-detects the file)

# 4. Run the analysis
python analysis_v3.py
```

### Requirements
```
pandas
openpyxl
scikit-learn
matplotlib
seaborn
scipy
numpy
```

---

## AI Disclosure

This project used Claude (Anthropic) as an AI assistant for:
- Refactoring and extending the Python analysis code
- Preparing the `master_data_v2.xlsx` dataset structure
- Generating the composite economic index formula

All analytical decisions, hypothesis framing, and interpretation of results were made by the student. Raw outputs and prompts are documented in the commit history.

---

## Repository Structure

```
DSA-210-Project/
├── master_data_v2.xlsx          # Main dataset (Turkey + England, 4 sheets)
├── analysis_v3.py               # Full analysis script
├── Dsa210 - Term Project Proposal Muhammed M...pdf   # Original proposal
├── README.md                    # This file
└── figures/                     # Generated PNG figures (after running script)
    ├── fig_overview_turkey.png
    ├── fig_overview_england.png
    ├── fig_comparison.png
    ├── fig_heatmap.png
    ├── fig_elbow.png
    └── fig_pca_clusters.png
```
