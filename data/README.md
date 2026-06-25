# Dataset Information

## UCI Bank Marketing Dataset

This study uses the **UCI Bank Marketing Dataset**, originally collected and published by:

> Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001

---

## Download

The dataset is publicly available at the UCI Machine Learning Repository:

🔗 **https://archive.ics.uci.edu/dataset/222/bank+marketing**

Download the file: `bank-full.csv`  
Place it in this `/data` folder before running any scripts.

---

## Dataset Details

| Property | Value |
|---|---|
| Records | 45,211 |
| Features | 16 input + 1 target |
| File format | CSV (semicolon-separated) |
| Target variable | y (yes = subscribed, no = did not subscribe) |
| Class distribution | 88.3% No, 11.7% Yes |
| Time period | May 2008 – November 2010 |
| Source institution | Portuguese banking institution |

---

## Feature Description

| Feature | Type | Description |
|---|---|---|
| age | Numerical | Age of the customer |
| job | Categorical | Type of job |
| marital | Categorical | Marital status |
| education | Categorical | Education level |
| default | Binary | Has credit in default? |
| balance | Numerical | Average annual account balance (euros) |
| housing | Binary | Has housing loan? |
| loan | Binary | Has personal loan? |
| contact | Categorical | Contact communication type |
| day | Numerical | Last contact day of month |
| month | Categorical | Last contact month |
| duration | Numerical | Last contact duration (seconds) |
| campaign | Numerical | Number of contacts this campaign |
| pdays | Numerical | Days since last contact from previous campaign |
| previous | Numerical | Number of contacts before this campaign |
| poutcome | Categorical | Outcome of previous campaign |
| **y** | **Binary (Target)** | **Subscribed to term deposit? (yes/no)** |

---

## Why This Dataset?

- Real-world banking data from an actual Portuguese bank
- Widely used in published academic research — enabling direct benchmarking
- Well-documented with no missing values
- Contains a realistic class imbalance representative of marketing campaign data
