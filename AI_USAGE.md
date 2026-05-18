# AI Tool Usage Disclosure

**DSA 210 — Introduction to Data Science | Spring 2026**  
**Muhammed Mustafa Güneş — Student ID: 34313**

This document fulfils the academic integrity requirement to disclose AI tool usage,
including specific prompts and outputs generated, as stated in the DSA 210 project guidelines.

---

## Tool Used

**Claude (Anthropic) — claude.ai**  
Used throughout the project for four distinct tasks described below.

---

## 1. Python Code Refactoring

**Prompt given to Claude:**
> "Here is my analysis script. It runs but the code is messy — can you refactor it so it's well-commented, uses functions, and follows PEP8 style? Don't change the logic or the outputs, only clean up the structure."

**Output used:**
Claude restructured the script into named functions (`load_data()`, `build_economic_index()`, `run_hypothesis_tests()`, `run_ml_models()`), added docstrings and inline comments, and standardised variable names. The statistical logic, model choices, and all numerical outputs were unchanged — verified by re-running and comparing outputs before and after refactoring.

**File affected:** `analysis_v3.py`

---

## 2. Economic Health Index Formula

**Prompt given to Claude:**
> "I have three economic variables: USD/TRY exchange rate, GDP growth rate, and CPI inflation. I want to combine them into a single 0–1 index where higher = better economic conditions for a football club. Exchange rate and inflation should be inverted. How should I normalise and combine them?"

**Output used:**
Claude suggested min-max normalisation per variable, inverting the exchange rate (for Turkey) and inflation before normalising, then averaging the three normalised values. The formula used in the final analysis:

```
econ_index = mean([
    1 - minmax(USD_TRY),   # inverted: weaker lira = lower score
    minmax(GDP_growth),
    1 - minmax(CPI_inflation)  # inverted: higher inflation = lower score
])
```

This formula was adopted as proposed and is implemented in `analysis_v3.py`.

---

## 3. Dataset Structure (master_data_v2.xlsx)

**Prompt given to Claude:**
> "I'm building a dataset with half-year periods (H1 and H2) from 1993 to 2024 for Turkey and England. Each row should have: year, half, country, exchange rate, GDP growth, CPI, and UEFA points for each club. Can you give me the column structure and a few example rows so I can build it in Excel?"

**Output used:**
Claude provided a column schema and 5 example rows. The schema was used to structure `master_data_v2.xlsx`. All actual data values were entered manually from primary sources (TCMB, World Bank, UEFA, Bank of England).

---

## 4. Project Webpage (index.html)

**Prompt given to Claude:**
> "Build me a single-page HTML project report for my data science project. Dark theme, professional look, use Bebas Neue for headings, DM Sans for body text. Sections: hero with stats, motivation, data sources, methodology steps, hypothesis testing results table, ML results with bar charts, conclusions grid, limitations, sources, footer. The colour palette should use dark navy, red, teal, and gold."

**Output used:**
Claude generated the full HTML/CSS structure including the responsive layout, navigation, stats cards, results table, ML accuracy bars, and conclusion grid. Section content (all text, numbers, findings, and interpretations) was written by the student. The webpage was subsequently edited by the student to correct factual details (e.g. Beşiktaş UCL season year) and to add explanatory notes on statistical findings.

**File affected:** `index.html`

---

## Summary

| Task | AI-generated | Student-authored |
|---|---|---|
| Code structure & style | Claude | Statistical logic, model selection, interpretation |
| Economic index formula | Claude (adopted as-is) | Decision to use it, verification against literature |
| Dataset column schema | Claude | All data values (manually sourced) |
| Webpage HTML/CSS | Claude | All text content, findings, analysis narrative |

All AI interactions took place on claude.ai. No AI tool was used to generate, fabricate, or interpret statistical results. All p-values, correlation coefficients, R², and ML accuracy figures were produced by running `analysis_v3.py` locally.
