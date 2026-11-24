# SD6105 Data Visualisation — Tutorial Solutions

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-teal)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-orange)
![NTU](https://img.shields.io/badge/NTU-SD6105-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **Course:** SD6105 Data Visualisation, NTU Master of Science in Data Science
> **Tutorials:** 1 – 4 | **Language:** Python 3 · Pandas · Matplotlib · Seaborn

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Tutorial 1 — Data Attributes & AI-Assisted Plotting](#tutorial-1--data-attributes--ai-assisted-plotting)
- [Tutorial 2 — Visual Encoding](#tutorial-2--visual-encoding)
- [Tutorial 3 — Basic Plots & Charts](#tutorial-3--basic-plots--charts)
- [Tutorial 4 — Visual Perception](#tutorial-4--visual-perception)
- [Key Concepts Covered](#key-concepts-covered)
- [Dependencies](#dependencies)
- [Data Sources](#data-sources)

---

## Overview

This repository contains **complete, runnable Python solutions** for all four SD6105 tutorials, covering the full data visualisation pipeline — from data classification and wrangling, through chart selection and visual encoding, to perceptual and design principles.


Each tutorial folder contains:

| File | Description |
|---|---|
| `tutorial[N]_solutions.py` | Complete annotated Python script |
| `data/` | Datasets used by the script |
| `charts/` | All generated output charts (PNG) |

---

## Repository Structure

```
SD6105-Data-Visualisation-Tutorials/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore
│
├── Tutorial_1_Data_Attributes/
│   ├── tutorial1_solutions.py         ← NOIR classification + plotting
│   ├── data/
│   │   ├── hire2FTE.csv               ← Monthly transaction volumes
│   │   └── Titanic.csv                ← 891 Titanic passengers
│   └── charts/
│       ├── q2_cluster_bar_chart.png
│       ├── q4_line_chart_effective.png
│       └── q5_combined_line_bar.png
│
├── Tutorial_2_Visual_Encoding/
│   ├── tutorial2_solutions.py         ← Bertin + Mackinlay + critique
│   └── charts/
│       ├── bertin_visual_variables_demo.png
│       └── mackinlay_effectiveness_ranking.png
│
├── Tutorial_3_Basic_Plots/
│   ├── tutorial3_solutions.py         ← Smoothing, composition, relation, distribution
│   ├── data/
│   │   ├── owidSG.csv                 ← Singapore COVID-19 (Mar–Aug 2020)
│   │   ├── Real estate.csv            ← 414 property transactions (Taiwan)
│   │   ├── Students performance.csv   ← 1000 students' exam scores
│   │   └── dept profits.xlsx          ← Dept X/Y/Z profits 2015–2021
│   │   # ⚠ global_terrorism.csv (29 MB) → download from Kaggle (see below)
│   └── charts/                        ← 12 output charts
│       ├── q2_covid_raw_overlay.png
│       ├── q3_covid_smoothed.png
│       ├── q2_terrorism_stacked_bar.png
│       ├── q3_terrorism_pct_stacked_bar.png
│       ├── q4_terrorism_region_stacked_area.png
│       ├── q1_real_estate_scatter_plots.png
│       ├── q5_real_estate_pairgrid.png
│       ├── q6_dept_profits_actual_vs_smoothed.png
│       ├── q2_gender_violin_plots.png
│       ├── q3_subjects_boxplot_medians.png
│       ├── q4_test_prep_boxplots.png
│       ├── q5_ethnic_group_kde.png
│       ├── q6_ethnic_subject_boxplots.png
│       ├── q7_parental_education_boxplot.png
│       └── q8_parental_education_by_gender.png
│
└── Tutorial_4_Visual_Perception/
    ├── tutorial4_solutions.py         ← Weber's Law, Gestalt, colour, Likert
    ├── data/
    │   ├── SG Temp2.csv               ← Singapore monthly temp 2015–2016
    │   ├── Children Growth Rate.csv   ← Boys/girls height growth 0–20 yrs
    │   ├── FT pie and bar.csv         ← Financial Times quiz data
    │   ├── teaching feedback.csv      ← Likert-scale teaching survey
    │   └── Changing Profits - CSV.csv ← Dept A/B/C profits 2015–2021
    └── charts/                        ← 8 output charts
        ├── q1_weber_grey_squares.png
        ├── q1_temp_sidebyside_bar.png
        ├── q1_temp_improved_bar.png
        ├── q2_temp_difference_chart.png
        ├── q2_dept_c_presentation.png
        ├── q1_colour_palettes.png
        ├── q2_children_growth_rate.png
        └── q6_teaching_feedback_diverging.png
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/SD6105-Data-Visualisation-Tutorials.git
cd SD6105-Data-Visualisation-Tutorials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run any tutorial from its own folder
cd Tutorial_1_Data_Attributes
python tutorial1_solutions.py

cd ../Tutorial_3_Basic_Plots
# ⚠ Download global_terrorism.csv first (see Data Sources below)
python tutorial3_solutions.py
```

> **Note:** Each script must be run **from inside its own folder** so that `./data/` and `./charts/` relative paths resolve correctly.

---

## Tutorial 1 — Data Attributes & AI-Assisted Plotting

### Learning Objectives

- Classify data using the **NOIR scale** (Nominal, Ordinal, Interval, Ratio)
- Distinguish **discrete vs. continuous** data
- Create and critique **cluster bar charts** and **line charts**
- Use `fill_between` shading to reveal a growing shortfall

### Datasets

| File | Rows | Description |
|---|---|---|
| `hire2FTE.csv` | 3 | Monthly Received/Processed transaction volumes |
| `Titanic.csv` | 891 | Real Titanic passenger data |

### NOIR Classification Summary

| Data (Titanic) | Scale | Reasoning |
|---|---|---|
| PassengerId | Nominal | Labels only |
| Pclass | Ordinal | 1st > 2nd > 3rd, unequal gaps |
| Age | Ratio | Absolute zero (0 yrs = no age lived) |
| Fare | Ratio | Monetary value with absolute zero |
| Embarked | Nominal | C/S/Q are unordered labels |

### Output Charts

| Chart | Key Technique | Visualisation Concept |
|---|---|---|
| `q2_cluster_bar_chart.png` | Clustered bars + `ax.annotate` | Compare Received vs Processed (grouped bar) |
| `q4_line_chart_effective.png` | `fill_between` red shading | Weber's Law: shading reveals gap impossible to see in bar chart |
| `q5_combined_line_bar.png` | Twin Y-axis (`twinx`) | Dual encoding: volume (lines) + shortfall (bars) |

**Q4 — Why line chart beats clustered bar here:**
> Weber's Law (ΔI/I = k): the ~5–30 unit gap is imperceptible against ~150-unit bar heights. A line chart with `fill_between` makes the *difference* — not individual values — the visual focus.

---

## Tutorial 2 — Visual Encoding

### Learning Objectives

- Name Bertin's **7 visual variables** and the **3 item mark types**
- Apply **Mackinlay's effectiveness ranking** for different data scales
- Explain **redundant encoding** and when it is beneficial
- Critique real-world infographics using visual encoding principles

### Key Theory

**Bertin's 3 Item Marks:**
| Mark | Dimension | Example |
|---|---|---|
| **Point** | 0D — location only | Scatter plot dot |
| **Line** | 1D — path, trajectory | Line chart |
| **Area** | 2D — region, extent | Bar, map region |

**Mackinlay's Top Channel per Data Type:**
| Data Type | #1 Channel | #2 Channel | Worst |
|---|---|---|---|
| Quantitative | Position | Length | Shape |
| Ordinal | Position | Density/Value | Shape |
| Nominal | Position | Colour Hue | Volume |

**Redundant Encoding:** Using ≥2 visual channels for the same variable. Improves speed, accuracy, and accessibility (e.g., colour-blind users).

### Output Charts

| Chart | Content |
|---|---|
| `bertin_visual_variables_demo.png` | 7 visual variables applied to line marks |
| `mackinlay_effectiveness_ranking.png` | Ranked bar charts for Q/O/N data types |

---

## Tutorial 3 — Basic Plots & Charts

### Learning Objectives

- Apply **rolling window smoothing** to reveal hidden trends in noisy time series
- Build **stacked bar** and **stacked area charts** for composition analysis
- Apply **LOWESS regression** for non-parametric relationship fitting
- Compare distributions with **violin plots**, **box plots**, and **KDE overlays**

### Datasets

| File | Rows | Description |
|---|---|---|
| `owidSG.csv` | 184 | Singapore COVID-19 daily data (Mar–Aug 2020) |
| `global_terrorism.csv` ⚠ | 181,691 | Global terrorism incidents 1970–2017 (download from Kaggle) |
| `Real estate.csv` | 414 | Property transactions with MRT distance, age, stores |
| `Students performance.csv` | 1,000 | Exam scores by gender, ethnicity, parental education |
| `dept profits.xlsx` | 7 | Annual profits of Dept X/Y/Z 2015–2021 |

### Section 3.1 — Smoothing

**Key code pattern:**
```python
# 7-day centered rolling mean — reveals peaks in noisy COVID data
smooth = sg["new_cases"].rolling(window=7, center=True).mean()

# OWID "new_cases_smoothed" uses right-edge (no center):
smooth_owid = sg["new_cases"].rolling(window=7, min_periods=1).mean()
```

**Singapore COVID-19 Findings:**
- **Peak 1 (~23 Apr 2020):** Foreign worker dormitory clusters + Circuit Breaker
- **Peak 2 (~30 Jul 2020):** Phase 2 reopening social mixing

⚠ **Pitfall — Smoothing over-hides crisis years:**
Cubic-spline smoothing of Dept X/Y/Z profits hides the 2017 crisis dip and may imply false crossovers between departments that never happened in real data.

### Section 3.2 — Composition

**`pd.cut()` for 5-year period binning:**
```python
terror["period"] = pd.cut(
    terror["iyear"],
    bins=range(1970, 2020, 5),
    labels=[f"{y}–{y+4}" for y in range(1970, 2015, 5)],
    right=False
)
```
> Replaces a manual `assign_period()` function — cleaner, less error-prone.

**100% stacked chart trade-off:**
| Chart Type | Shows | Hides |
|---|---|---|
| Absolute stacked bar | Volume + composition | % changes hard to compare |
| 100% stacked bar | % composition only | Absolute attack counts |

### Section 3.3 — Relationships

**LOWESS vs Linear Regression:**
```python
from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(y, x, frac=0.3, return_sorted=True)  # frac: neighbourhood size
```
| Method | Assumptions | Use When |
|---|---|---|
| `sns.regplot(order=1)` | Linear | Simple monotonic trend |
| `sns.regplot(order=2/3)` | Polynomial | Obvious curve |
| **LOWESS** | **None (data-driven) | **Complex, changing trend |

**Real Estate Key Finding:**
> MRT distance is the **primary driver** of price. House age appears correlated with price only because older buildings happen to be near older (pre-2000) MRT lines — **confounding variable**

### Section 3.4 — Distributions

**Violin plot vs Box plot:**
```python
# Split violin: two sub-groups mirrored on left/right halves
sns.violinplot(data=df, x="gender", y="score",
               split=True, inner="box",
               palette={"female": "salmon", "male": "steelblue"})
```
> Datasets can have **identical quartiles but different distribution shapes** — violin plots reveal this; box plots do not.

**FacetGrid with `despine`:**
```python
g = sns.FacetGrid(df, col="gender", height=6, aspect=1.4)
g.map_dataframe(sns.boxplot, x="education", y="score", palette="Blues_d")
g.set_xticklabels(rotation=30)
g.despine(left=True)    # Remove left spine for cleaner look
```

### Output Charts

| Chart | Section | Key Insight |
|---|---|---|
| `q2_covid_raw_overlay.png` | 3.1 | Raw daily data is too noisy for trend analysis |
| `q3_covid_smoothed.png` | 3.1 | 7-day rolling mean reveals two distinct COVID peaks |
| `q6_dept_profits_actual_vs_smoothed.png` | 3.1 | Smoothing hides the 2017 crisis |
| `q2_terrorism_stacked_bar.png` | 3.2 | Bombing/Explosion dominates since 1990s |
| `q3_terrorism_pct_stacked_bar.png` | 3.2 | % view: Armed Assault share rises after 2000 |
| `q4_terrorism_region_stacked_area.png` | 3.2 | Middle East peaks ~2013–2014; W. Europe dominated mid-70s |
| `q1_real_estate_scatter_plots.png` | 3.3 | MRT distance has strongest non-linear negative effect |
| `q5_real_estate_pairgrid.png` | 3.3 | Pairwise relationships — MRT distance dominates |
| `q2_gender_violin_plots.png` | 3.4 | Girls lead Reading/Writing; Boys marginally lead Math |
| `q3_subjects_boxplot_medians.png` | 3.4 | Math has lowest median score → toughest subject |
| `q4_test_prep_boxplots.png` | 3.4 | Test prep improves all subjects; Writing gains most |
| `q5_ethnic_group_kde.png` | 3.4 | Group E best overall; Group A lowest |
| `q6_ethnic_subject_boxplots.png` | 3.4 | Group E leads all three subjects consistently |
| `q7_parental_education_boxplot.png` | 3.4 | Master's degree parents → highest student scores |
| `q8_parental_education_by_gender.png` | 3.4 | Education effect similar for both genders |

---

## Tutorial 4 — Visual Perception

### Learning Objectives

- Apply **Weber's Law** (ΔI/I = k) to explain discrimination difficulty
- Use **Gestalt principles** (Figure-Ground, Enclosure, Proximity, Similarity) to design presentations
- Select appropriate **colour palettes** (categorical, sequential, diverging)
- Redesign a **Likert-scale chart** using a diverging stacked bar
- Apply **Kosslyn's 8 psychological principles** to critique real infographics

### Section 4.1 — Weber's Law

| Background | Hardest Pair | Why |
|---|---|---|
| White | Square #7 (192) & #8 (255) | At high intensity, relative step (63/255) is tiny |
| Black | Square #1 (0) & #2 (32) | At low intensity, absolute step (32) is imperceptible |

**Solution:** Use **logarithmic/perceptual spacing** so each step spans the same JND (just noticeable difference).

### Section 4.2 — Overcoming Weber's Law

**Why side-by-side bars fail for April comparison:**
Bars start at ~0°C; the ~0.3–0.5°C April difference is invisible against a ~28°C bar height. Weber's Law: the relative difference ≈ 0.3/28 ≈ 1% — far below the JND.

**Fix 1 — Truncate Y-axis:** Start at 25.5°C instead of 0 to amplify the visual difference.
**Fix 2 — Difference chart:** Directly encode `2016 − 2015` as bar height — no mental subtraction needed.

```python
diff = temp_2016 - temp_2015
ax.bar(months, diff, color=["orange" if d > 0 else "steelblue" for d in diff])
```

### Section 4.3 — Gestalt Principles in Practice

| Gestalt Principle | Implementation in Dept C Chart |
|---|---|
| **Figure-Ground** | Dept C = red solid line; A & B = dimgrey dashed (background) |
| **Enclosure** | Transparent red `axvspan` + dashed box around 2018 crisis year |
| **Proximity** | Direct end-labels at right edge of each line (no legend needed) |
| **Similarity** | All Dept C annotations share firebrick red colour |

### Section 4.5 — Colour Palettes

```python
# Categorical — 9 distinct hues for nominal data
cat_hex = ["#9b59b6","#3498db","#95a5a6","#e74c3c","#34495e","#2ecc71",
           "#e67e22","#1abc9c","#f39c12"]
sns.set_palette(cat_hex)

# Sequential — varying lightness for ordered data
seq = sns.color_palette("Blues", 9)

# Diverging — two hues meeting at neutral for bipolar data
div = sns.color_palette("RdBu", 9)
```

| Type | When | Seaborn Name |
|---|---|---|
| **Categorical** | Nominal unordered categories | `tab10`, `Set1`, `Paired` |
| **Sequential** | Ordered one-direction data | `Blues`, `YlOrRd`, `viridis` |
| **Diverging** | Data with meaningful midpoint | `RdBu`, `coolwarm`, `RdYlGn` |

### Section 4.7 — Teaching Feedback Redesign

**Why clustered bar is wrong for Likert:**
1. Forces mental summation of positive/negative bars per professor
2. Arbitrary colour palette implies no ordinal order
3. No Gestalt figure-ground — all bars equally prominent

**Diverging stacked bar solution:**

```python
# Positive bars extend right from center; negative extend left
right_start = 0
for cat in ["Agree", "Strongly Agree"]:
    ax.barh(i, vals[cat], 0.6, left=right_start, color=DIV_COLORS[cat])
    right_start += vals[cat]

left_start = 0
for cat in ["Disagree", "Strongly Disagree"]:
    ax.barh(i, -vals[cat], 0.6, left=left_start, color=DIV_COLORS[cat])
    left_start -= vals[cat]
```

**Findings:**
| Professor | % Positive | % Negative | Interpretation |
|---|---|---|---|
| Prof C | 84% | 11% | Best feedback, decisive |
| Prof A | 84% | 8% | Good, more neutral responses |
| Prof B | 28% | 32% | Polarised — inconsistent quality |

### Output Charts

| Chart | Section | Key Concept |
|---|---|---|
| `q1_weber_grey_squares.png` | 4.1 | Weber's Law: discrimination on white vs. black background |
| `q1_temp_sidebyside_bar.png` | 4.2 | Weber's Law: April difference invisible in side-by-side bars |
| `q1_temp_improved_bar.png` | 4.2 | Fix: truncated Y-axis + direct annotation |
| `q2_temp_difference_chart.png` | 4.2 | Fix: difference chart makes April gap immediately visible |
| `q2_dept_c_presentation.png` | 4.3 | Gestalt: Figure-Ground, Enclosure, Proximity, Similarity |
| `q1_colour_palettes.png` | 4.5 | Three palette types: categorical / sequential / diverging |
| `q2_children_growth_rate.png` | 4.5 | Girls peak growth ~11–12 yrs; Boys ~13–14 yrs |
| `q6_teaching_feedback_diverging.png` | 4.7 | Diverging stacked bar for Likert data |

---

## Key Concepts Covered

| Concept | Tutorial | Description |
|---|---|---|
| **NOIR Scale** | T1 | Nominal, Ordinal, Interval, Ratio data classification |
| **Bertin's Visual Variables** | T2 | 7 channels: position, size, shape, value, hue, orientation, texture |
| **Mackinlay Ranking** | T2 | Position always #1; Shape worst for quantitative |
| **Redundant Encoding** | T2 | Multiple channels → faster, more accurate perception |
| **Rolling Smoothing** | T3 | `df.rolling(window=7, center=True).mean()` |
| **LOWESS Regression** | T3 | Non-parametric; frac controls neighbourhood size |
| **pd.cut() Binning** | T3 | Clean period labelling without custom functions |
| **Violin vs Box Plot** | T3 | Violin reveals shape; box hides bimodal distributions |
| **Weber's Law** | T4 | ΔI/I = k; JND proportional to base intensity |
| **Stevens's Power Law** | T4 | Length (n=1) accurate; Area (n≈0.7) underestimated |
| **Gestalt Principles** | T4 | Proximity, Similarity, Enclosure, Figure-Ground, Continuity |
| **Diverging Colour Palette** | T4 | Bipolar data (Likert, +/−) → two hues from neutral centre |
| **Kosslyn's 8 Principles** | T4 | Salience, Discriminability, Compatibility, Capacity Limitation |
| **Data-Ink Ratio** | T4 | Maximise: remove chartjunk; minimise non-data ink |
| **Lie Factor** | T4 | Effect in graphic / Effect in data = 1.0 for truthful charts |

---

## Dependencies

```
pandas>=2.0          # Data manipulation and wrangling
numpy>=1.24          # Numerical operations
matplotlib>=3.7      # Base plotting library
seaborn>=0.13        # Statistical visualisation (built on matplotlib)
scipy>=1.11          # Linear regression (stats.linregress), interpolation
statsmodels>=0.14    # LOWESS smoothing
openpyxl>=3.1        # Read .xlsx files (dept profits, etc.)
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Data Sources

| Dataset | Source | Notes |
|---|---|---|
| `Titanic.csv` | [Kaggle — Titanic ML](https://www.kaggle.com/c/titanic) | 891 passengers |
| `hire2FTE.csv` | Course material | Fictional transaction data |
| `owidSG.csv` | [Our World in Data](https://ourworldindata.org/covid-cases) | Singapore COVID-19 subset |
| `global_terrorism.csv` ⚠ | [Kaggle — GTD](https://www.kaggle.com/datasets/START-UMD/gtd/data) | **29 MB — download manually** |
| `Real estate.csv` | [Kaggle — Real Estate Prediction](https://www.kaggle.com/quantbruce/real-estate-price-prediction) | 414 transactions (Taiwan) |
| `Students performance.csv` | [Kaggle — Students Performance](https://www.kaggle.com/spscientist/students-performance-in-exams) | 1000 students |
| `dept profits.xlsx` | Course material | Fictional company data |
| `SG Temp2.csv` | Course material | Singapore 2015–2016 temperatures |
| `Children Growth Rate.csv` | [Disabled World](https://www.disabled-world.com/calculators-charts/height-weight-teens.php) (modified) | Average height growth 0–20 yrs |
| `teaching feedback.csv` | Course material | Fictional Likert survey data |

---

## Code Style & Optimisations

The scripts in this repository apply best practices from the course's Python cheat sheets:

| Pattern | Example | Benefit |
|---|---|---|
| `df.assign()` chaining | `.assign(shortfall=lambda d: d.received - d.processed)` | Readable transformation pipeline |
| `pd.cut()` binning | `pd.cut(year, bins=range(1970,2020,5), labels=...)` | Replaces verbose custom functions |
| `df.nlargest(n, col)` | `.nlargest(1, "smooth_7d").iloc[0]` | Cleaner peak detection |
| `sns.set_context("talk")` | Global style setting | Presentation-ready font sizes |
| `sns.despine()` | `sns.despine(fig=g.fig, left=True)` | Cleaner FacetGrid aesthetics |
| Named colours | `"steelblue"`, `"firebrick"`, `"darkorange"` | Self-documenting, from Matplotlib colour reference |
| `encoding="utf-8-sig"` | `pd.read_csv(..., encoding="utf-8-sig")` | Handles BOM characters in CSV |

---

*SD6105 Data Visualisation — NTU Master of Science in Data Science*
*Solutions compiled for exam preparation and revision.*
