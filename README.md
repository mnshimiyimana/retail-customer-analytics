# Retail Customer Analytics

**End-to-end data analytics project** — from raw transactional data to an interactive multi-page dashboard.

Built as a demonstration of core data analyst skills: cleaning, EDA, segmentation, cohort analysis, hypothesis testing, forecasting, and interactive visualisation.

---

## Dashboard Preview

> **Live dashboard:** run `python dashboard.py` then open [http://127.0.0.1:8050](http://127.0.0.1:8050)

| Page | Content |
|---|---|
| **Overview** | KPI cards · Revenue trend · Category sunburst · Channel performance |
| **Sales Trends** | Filterable revenue chart · MoM growth · Day × Month heatmap |
| **Customers** | RFM scatter · Segment treemap · Summary table |
| **Retention** | Cohort heatmap · Average retention decay curve |
| **Forecast** | 6-month projection · 95% prediction intervals · Model diagnostics |
| **Regional** | Revenue by UK region · Top 12 products · Channel breakdown |

---

## Project Structure

```
Customer_Analytics_Project/
│
├── dashboard.py                     # Plotly Dash interactive dashboard
├── Customer_Analytics_Project.ipynb # Full analysis notebook
├── Customer_Analytics_Project.html  # Static HTML export (no Python needed)
├── assets/
│   └── style.css                    # Custom dark-theme styles
└── build_notebook.py                # Script that generates the notebook
```

---

## Dataset

Synthetically generated to mirror a real-world e-commerce export (Shopify / WooCommerce / BigQuery structure).

| Field | Description |
|---|---|
| `order_id` | Unique order identifier |
| `customer_id` | Anonymised customer ID |
| `order_date` | Transaction date |
| `category` | Product category (6 categories) |
| `product` | Product name (30 SKUs) |
| `quantity` | Units purchased |
| `unit_price` | Price per unit (£) |
| `revenue` | Total order revenue (£) |
| `region` | UK region (8 regions) |
| `channel` | Acquisition channel (6 channels) |

**Scale:** ~18,000 transactions · 4,200 customers · Jan 2024 – Dec 2025  
**Realism:** power-law customer distribution (Zipf), seasonal demand curve, Q4 holiday uplift, injected data quality issues (duplicates, missing prices)

---

## Analysis Techniques

### 1. Data Cleaning
- Deduplication of order records
- Category-median imputation for missing unit prices
- Time feature extraction (year, month, quarter, day-of-week, week-of-year)

### 2. Exploratory Data Analysis
- Revenue distribution and outlier inspection
- Day-of-week and seasonal patterns
- Category and channel breakdowns

### 3. Sales Trend Analysis
- Monthly revenue with 3-month rolling average
- Month-on-month growth rate
- Year-on-year comparison (2024 vs 2025)

### 4. RFM Customer Segmentation
- Quintile scoring on Recency, Frequency, and Monetary value
- K-Means clustering with Elbow Method + Silhouette Score to select k=4
- Segments: **Champions · Loyal Customers · At-Risk · Lost/Inactive**

### 5. Cohort Retention Analysis
- Customers grouped by first-purchase month
- Retention tracked across 12 subsequent months
- Heatmap and average decay curve

### 6. Statistical Hypothesis Testing

| Test | Question | Method |
|---|---|---|
| 1 | Do weekends generate higher AOV? | Welch t-test |
| 2 | Do email customers have higher LTV than social? | Mann-Whitney U |
| 3 | Does order frequency correlate with lifetime value? | Pearson r |
| 4 | Is Q4 revenue significantly higher than other quarters? | Welch t-test |

### 7. Revenue Forecasting
- OLS linear regression with trend variable + 11 monthly seasonal dummies
- 6-month horizon with 95% prediction intervals
- In-sample R² reported

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.13** | Core language |
| **Pandas** | Data manipulation and aggregation |
| **NumPy** | Numerical operations and data generation |
| **Scikit-learn** | K-Means clustering, StandardScaler, LinearRegression |
| **SciPy** | Welch t-test, Mann-Whitney U, Pearson correlation |
| **Plotly** | Interactive charts |
| **Dash** | Multi-page web dashboard framework |
| **Dash Bootstrap Components** | Responsive grid layout |
| **Jupyter Notebook** | Narrative analysis document |

---

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn \
            plotly dash dash-bootstrap-components jupyter
```

### Run the interactive dashboard
```bash
python dashboard.py
# Open http://127.0.0.1:8050
```

### Open the notebook
```bash
jupyter notebook Customer_Analytics_Project.ipynb
```

### View without Python
Open `Customer_Analytics_Project.html` in any browser — no Python or Jupyter required.

---

## Key Findings

- **Q4 drives disproportionate revenue** — statistically significant uplift confirmed (p < 0.05)
- **Email channel delivers the highest customer LTV** — significantly higher than social media (Mann-Whitney U, p < 0.05)
- **Champions (top segment) account for ~42% of total revenue** despite being a small fraction of customers
- **Month-1 cohort churn is the steepest drop-off point** — largest retention loss happens in the first 30 days
- **Solar/renewable cost parity analogy:** order frequency and LTV show strong positive correlation (r > 0.85)
- **Weekend AOV is higher** than weekday — statistically significant, suggesting premium intent on weekends

---

## Recommendations

1. **Invest in email marketing** — highest LTV channel, confirmed statistically
2. **Champion loyalty programme** — protect the segment generating 42% of revenue
3. **Q4 prep from August** — inventory, staffing, and campaigns need a 3-month lead time
4. **Weekend bundle promotions** — capitalise on higher weekend spend intent
5. **Month-1 onboarding sequence** — reduce the steepest churn window with a 7/14/30-day email flow

---

*Built by Maureen T. N. · April 2026*
