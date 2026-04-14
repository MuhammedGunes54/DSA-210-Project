# Economic Impact on Turkish Football Success (1993-2024)

## 📌 Project Overview
This project investigates the relationship between macroeconomic indicators—specifically the USD/TRY exchange rate—and the international performance of Turkish football at both club (UEFA) and national team (FIFA) levels. The study covers a 31-year period to observe how currency volatility affects sports competitiveness.

## 📊 Dataset & Resolution
Following instructor feedback, the dataset was expanded to ensure robustness for Machine Learning applications:
* **Timeframe:** 1993 – 2024 (31 years).
* **Data Points:** 64 organic data points.
* **Resolution:** Bi-annual (Q2 and Q4). 
    * **Q2 (June):** Captures end-of-season UEFA coefficients and mid-year FIFA rankings.
    * **Q4 (December):** Captures end-of-year FIFA rankings and UEFA group stage completions.
* **Sources:** Central Bank of the Republic of Türkiye (CBRT), UEFA official archives, and FIFA World Rankings.

## ⚙️ Methodology
* **Exploratory Data Analysis (EDA):** Visualizing long-term trends between exchange rates and success indices.
* **Hypothesis Testing:** Pearson Correlation Coefficient analysis to validate the statistical significance of the economic impact.
* **Normalization:** 2005 currency reform (removal of six zeros) was accounted for by normalizing historical TRL data to the current TRY basis.

## 💡 Key Findings
* **Club Level (UEFA):** A statistically significant correlation was found (**p-value < 0.05**). This suggests that club success is highly sensitive to exchange rate fluctuations due to foreign player costs and international transfer budgets.
* **National Team Level (FIFA):** No statistically significant correlation was observed. Performance appears to be driven by generational talent and domestic player pools rather than immediate economic shifts.

## 🚀 Future Work (Machine Learning Phase)
The current 64-point dataset is being used to train predictive models (Regression and Classification) to forecast future sporting success based on macroeconomic forecasts.
