# Using Big Data Analytics to Improve Customer Experience and Targeting in the Banking Sector

**Author:** Samuel Ukoha  
**Program:** Master of Science in Data Science & Analytics  
**Institution:** Toronto Metropolitan University  
**Year:** 2026  

---

## 📋 Project Overview

This Major Research Project (MRP) investigates how big data analytics and machine learning can improve customer experience and targeting in the banking sector. Using the UCI Bank Marketing Dataset — a real-world dataset of 45,211 records collected from a Portuguese bank's direct telemarketing campaigns — this study conducts an extended comparative analysis of six machine learning classifiers for predicting customer subscription to term deposit products.

### Key Contributions
- Most comprehensive classifier comparison to date on the UCI Bank Marketing Dataset
- First benchmark of **XGBoost** on this dataset (not tested in any prior published study)
- First systematic evaluation of **SMOTE** for class imbalance handling on this dataset
- Direct benchmarking of results against Moro et al. (2014) and Tran et al. (2023)
- Actionable customer targeting recommendations for banking practitioners

---

## 🔬 Research Questions

1. Which machine learning classifier produces the highest predictive performance for customer subscription prediction on the UCI Bank Marketing Dataset?
2. How do the results of this extended benchmark compare with published findings from Moro et al. (2014) and Tran et al. (2023)?
3. What is the impact of applying SMOTE for class imbalance handling on model performance?
4. Which customer characteristics are the strongest predictors of term deposit subscription?

---

## 📊 Dataset

**UCI Bank Marketing Dataset**  
- **Source:** UCI Machine Learning Repository  
- **Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing  
- **Records:** 45,211  
- **Features:** 16 input features + 1 binary target variable  
- **Target:** Whether the customer subscribed to a term deposit (yes/no)  
- **Class Imbalance:** 88.3% No | 11.7% Yes  
- **Original Paper:** Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.

> ⚠️ The dataset is not included in this repository due to file size. Please download it directly from the UCI repository link above and place `bank-full.csv` in the `/data` folder before running the scripts.

---

## 🤖 Models Compared

| Model | Learning Paradigm | Prior Benchmark |
|---|---|---|
| Logistic Regression | Linear | Moro et al. (2014), Tran et al. (2023) |
| Decision Tree | Tree-Based | Moro et al. (2014), Tran et al. (2023) |
| Random Forest | Ensemble (Bagging) | Tran et al. (2023) |
| **XGBoost** | **Ensemble (Boosting)** | **New — not previously benchmarked** |
| SVM | Kernel-Based | Moro et al. (2014), Tran et al. (2023) |
| KNN | Instance-Based | Tran et al. (2023) |

---

## ⚗️ Experimental Design

### Experiment 1 — Baseline (No SMOTE)
All six models trained on the imbalanced dataset using default hyperparameters. Replicates the approach of Moro et al. (2014) and Tran et al. (2023) for direct comparison.

### Experiment 2 — With SMOTE
SMOTE applied to the training set only. All six models retrained and evaluated. Quantifies the impact of class imbalance handling on model performance.

### Experiment 3 — Hyperparameter Tuning
GridSearchCV with 10-fold stratified cross-validation applied to the top 3 models from Experiment 2. Produces optimised final results.

### Evaluation Metrics
- Accuracy
- Precision
- Recall *(most critical — identifies actual subscribers)*
- F1-Score
- AUC-ROC

Results reported as **Mean ± Standard Deviation** across 10-fold cross-validation.

---

## 📁 Repository Structure

```
MRP-Banking-Customer-Targeting/
│
├── README.md
├── .gitignore
│
├── data/
│   └── README.md           # Dataset info and download link
│
├── notebooks/
│   └── MRP_Analysis.ipynb  # Jupyter notebook version of full pipeline
│
├── src/
│   ├── chapter3_figures.py # Chapter 3 EDA visualisations
│   └── mrp_experiments.py  # Full experiment pipeline (Exp 1, 2, 3)
│
├── figures/
│   ├── figure1_class_distribution.png
│   ├── figure2_numeric_distributions.png
│   ├── figure3_subscription_by_job.png
│   ├── figure4_subscription_by_education.png
│   ├── figure5_correlation_heatmap.png
│   ├── figure6_exp1_vs_exp2_auc.png
│   ├── figure7_all_metrics_exp2.png
│   ├── figure8_smote_recall_impact.png
│   ├── figure9_roc_curves_exp2.png
│   ├── figure10_confusion_matrices.png
│   └── figure11_cv_auc_mean_std.png
│
├── results/
│   ├── exp1_mean_std_results.csv
│   ├── exp2_mean_std_results.csv
│   └── exp3_mean_std_results.csv
│
└── report/
    └── MRP_Chapter1.docx
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/MRP-Banking-Customer-Targeting.git
cd MRP-Banking-Customer-Targeting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download `bank-full.csv` from:  
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing  
Place it in the `/data` folder.

### 4. Generate Chapter 3 Figures
```bash
python src/chapter3_figures.py
```

### 5. Run Full Experiment Pipeline
```bash
python src/mrp_experiments.py
```

Results will be saved to the `/results` folder and figures to the `/figures` folder.

---

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
scipy>=1.9.0
```

---

## 📚 References

1. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001

2. Tran, H., Le, N., & Nguyen, V.-H. (2023). Customer churn prediction in the banking sector using machine learning-based classification models. *IJIKM*, 18, 87–105. https://doi.org/10.28945/5086

3. Wedel, M., & Kannan, P. K. (2016). Marketing analytics for data-rich environments. *Journal of Marketing*, 80(6), 97–121. https://doi.org/10.1509/jm.15.0413

4. De Caigny, A., Coussement, K., & De Bock, K. W. (2018). A new hybrid classification algorithm for customer churn prediction. *European Journal of Operational Research*, 269(2), 760–772. https://doi.org/10.1016/j.ejor.2018.02.009

5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.

6. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Samuel Ukoha**  
MSc Data Science & Analytics  
Toronto Metropolitan University  
Toronto, Ontario, Canada, 2026
