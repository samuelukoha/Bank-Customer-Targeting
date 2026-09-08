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
Bank-Customer-Targeting/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   └── README.md
│
├── src/
│   ├── data_exploratory_analysis.py
│   ├── methodology_and_experiments.py
│   └── feature_importance_analysis.py        
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
│   ├── figure11_cv_auc_mean_std.png
│   ├── figure12_methodology_diagram.png      
│   ├── figure13_feature_importance_rf_xgb.png 
│   ├── figure14_consensus_feature_importance.png 
│   └── figure15_lr_coefficients.png           
│
├── results/                                   
│   ├── README.md
│   ├── experiment1_results.csv
│   ├── experiment2_results.csv
│   ├── experiment3_results.csv
│   ├── exp1_mean_std_results.csv
│   ├── exp2_mean_std_results.csv
│   ├── exp3_mean_std_results.csv
│   ├── feature_importance_consensus.csv
│   ├── feature_importance_rf.csv
│   ├── feature_importance_xgb.csv
│   └── feature_importance_lr.csv
│
└── report/
    ├── MRP.docx
    ├── Samuel_Ukoha_Full_MRP_Report.docx      
    └── Samuel_Ukoha_MRP_Poster.pdf            
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

### 4. Generate Chapter 3 — Exploratory Data Analysis Figures
```bash
python src/data_exploratory_analysis.py
```

### 5. Run Chapter 4 — Methodology and Experiments Pipeline
```bash
python src/methodology_and_experiments.py
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

1. Chen, M., Mao, S., & Liu, Y. (2014). Big data: A survey. Mobile Networks and Applications, 19(2),
171–209. ResearchGate
2. Wedel, M., & Kannan, P. K. (2016). Marketing analytics for data-rich environments. Journal of Marketing,
80(6), 97–121. ResearchGate
3. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing.
Decision Support Systems, 62, 22–31. ResearchGate
4. Tran, H., Le, N., & Nguyen, V.-H. (2023). Customer churn prediction in the banking sector using machine
learning-based classification models. IJIKM, 18, 87–105. Open access
5. De Caigny, A., Coussement, K., & De Bock, K. W. (2018). A new hybrid classification algorithm for
customer churn prediction based on logistic regression and decision trees. European Journal of Operational
Research, 269(2), 760–772. ResearchGate
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd
ACM SIGKDD, 785–794. ArXiv
7. Ngai, E. W. T., Hu, Y., Wong, Y. H., Chen, Y., & Sun, X. (2011). The application of data mining
techniques in financial fraud detection. Decision Support Systems, 50(3), 559–569. ResearchGate
8. Kambatla, K., Kollias, G., Kumar, V., & Grama, A. (2014). Trends in big data analytics. Journal of
Parallel and Distributed Computing, 74(7), 2561–2573. ResearchGate
9. Provost, F., & Fawcett, T. (2013). Data science and its relationship to big data and data-driven decision
making. Big Data, 1(1), 51–59. ResearchGate
10. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. Semantic Scholar
11. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297. Semantic
Scholar
12. Zhang, Z. (2016). Introduction to machine learning: K-nearest neighbors. Annals of Translational
Medicine, 4(11), 218. PubMed open access
13. Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81–106. Semantic Scholar
14. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority
over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357. Open access
15. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and
Data Engineering, 21(9), 1263–1284. ResearchGate
16. Lemmens, A., & Croux, C. (2006). Bagging and boosting classification trees to predict churn. Journal
of Marketing Research, 43(2), 276–286. ResearchGate
17. Coussement, K., & Van den Poel, D. (2008). Churn prediction in subscription services: An application
of support vector machines. Expert Systems with Applications, 34(1), 313–327. ResearchGate
18. Larivi`ere, B., & Van den Poel, D. (2005). Predicting customer retention and profitability by using random
forests and regression forest techniques. Expert Systems with Applications, 29(2), 472–484. ResearchGate
19. Peng, C.-Y. J., Lee, K. L., & Ingersoll, G. M. (2002). An introduction to logistic regression analysis and
reporting. Journal of Educational Research, 96(1), 3–14. ResearchGate
20. Han, J., Kamber, M., & Pei, J. (2012). Data Mining: Concepts and Techniques (3rd ed.). Morgan
Kaufmann. Widely available
21. Ganganwar, V. (2012). An overview of classification algorithms for imbalanced datasets. International
Journal of Emerging Technology and Advanced Engineering, 2(4), 42–47. Open access
22. Malik, M., & Thomas, L. C. (2010). Modelling credit risk of portfolio of consumer loans. Journal of the
Operational Research Society, 61(3), 411–420. ResearchGate
23. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825–2830. Open access — https://jmlr.org/papers/v12/pedreg
osa11a.html

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Samuel Ukoha**  
MSc Data Science & Analytics  
samuel.n.ukoha@gmail.com
Toronto Metropolitan University  
Toronto, Ontario, Canada, 2026
