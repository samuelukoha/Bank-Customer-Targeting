"""
================================================================================
MRP — Using Big Data Analytics to Improve Customer Experience and
       Targeting in the Banking Sector
Samuel Ukoha | Toronto Metropolitan University | 2026
================================================================================
EXPERIMENT PIPELINE — Chapter 4 Implementation
    → Outcome Measure Validation
    → Data Preprocessing & Feature Engineering
    → Stratified 80/20 Train/Test Split
    → Experiment 1: Baseline (No SMOTE)
    → Experiment 2: With SMOTE
    → Experiment 3: Hyperparameter Tuning (Best 3 Models)
    → Cross-Validation (10-Fold Stratified)
    → Benchmarking vs Moro et al. (2014) & Tran et al. (2023)
    → Results Visualisation
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Global Settings 
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BLUE_DARK  = '#1F3864'
BLUE_MID   = '#2E75B6'
BLUE_LIGHT = '#BDD7EE'
TEAL       = '#00B0D7'
WHITE      = '#FFFFFF'
GRAY       = '#F5F5F5'

plt.rcParams.update({
    'font.family':      'Arial',
    'axes.titlesize':   12,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.titleweight': 'bold',
    'figure.dpi':       150,
})

print("=" * 70)
print("MRP EXPERIMENT PIPELINE — Samuel Ukoha")
print("Using Big Data Analytics to Improve Customer Experience")
print("and Targeting in the Banking Sector")
print("=" * 70)



# STEP 1 — LOAD DATA

print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('bank-full.csv', sep=';')
print(f"  Dataset shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Class distribution:")
print(df['y'].value_counts())
print(f"\n  Class percentages:")
print((df['y'].value_counts(normalize=True) * 100).round(2))



# STEP 2 — OUTCOME MEASURE VALIDATION

print("\n" + "=" * 70)
print("[STEP 2] OUTCOME MEASURE VALIDATION")
print("=" * 70)

# Encode target for statistical tests
df['y_bin'] = (df['y'] == 'yes').astype(int)
subscribers     = df[df['y_bin'] == 1]
non_subscribers = df[df['y_bin'] == 0]

print(f"\n  Subscribers    : {len(subscribers):,} ({len(subscribers)/len(df)*100:.1f}%)")
print(f"  Non-Subscribers: {len(non_subscribers):,} ({len(non_subscribers)/len(df)*100:.1f}%)")

# 2a. Mann-Whitney U Test for numerical features
print("\n  [2a] Mann-Whitney U Test — Numerical Features")
print("  " + "-" * 50)
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
mw_results = []
for col in num_cols:
    stat, p = mannwhitneyu(
        subscribers[col], non_subscribers[col], alternative='two-sided')
    sig = " Significant" if p < 0.05 else " Not significant"
    mw_results.append({'Feature': col, 'U-Statistic': round(stat, 2),
                        'p-value': round(p, 4), 'Result': sig})
    print(f"  {col:12s} | U={stat:,.0f} | p={p:.4f} | {sig}")

mw_df = pd.DataFrame(mw_results)

# 2b. Chi-Square Test for categorical features
print("\n  [2b] Chi-Square Test — Categorical Features")
print("  " + "-" * 50)
cat_cols = ['job', 'marital', 'education', 'default', 'housing',
            'loan', 'contact', 'poutcome']
chi_results = []
for col in cat_cols:
    ct = pd.crosstab(df[col], df['y'])
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = " Significant" if p < 0.05 else " Not significant"
    chi_results.append({'Feature': col, 'Chi2': round(chi2, 2),
                         'p-value': round(p, 4), 'Result': sig})
    print(f"  {col:12s} | Chi2={chi2:,.2f} | p={p:.4f} | {sig}")

chi_df = pd.DataFrame(chi_results)

print("\n   Outcome measure validation complete.")
print("  All features showing p < 0.05 confirm meaningful relationship")
print("  with the target variable — model building is statistically justified.")



# STEP 3 — DATA PREPROCESSING & FEATURE ENGINEERING

print("\n" + "=" * 70)
print("[STEP 3] DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

df_model = df.drop(columns=['y_bin']).copy()

# Check for duplicates
dupes = df_model.duplicated().sum()
print(f"\n  Duplicates found: {dupes}")
if dupes > 0:
    df_model = df_model.drop_duplicates()
    print(f"  Duplicates removed. New shape: {df_model.shape}")

# Encode target
df_model['y'] = (df_model['y'] == 'yes').astype(int)

# Ordinal encoding for education
df_model['education'] = df_model['education'].map(
    {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3})

# Ordinal encoding for poutcome
df_model['poutcome'] = df_model['poutcome'].map(
    {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3})

# Binary encoding
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df_model[col] = (df_model[col] == 'yes').astype(int)

# Binary encoding for contact
df_model['contact'] = df_model['contact'].map(
    {'unknown': 0, 'telephone': 1, 'cellular': 2})

# Month encoding (ordinal)
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
df_model['month'] = df_model['month'].map(month_map)

# One-hot encoding for job and marital
df_model = pd.get_dummies(df_model, columns=['job', 'marital'], drop_first=True)

print(f"\n  Final dataset shape after encoding: {df_model.shape}")
print(f"  Features: {df_model.shape[1] - 1}")
print(f"  Target variable: y (0=No, 1=Yes)")

# Separate features and target
X = df_model.drop(columns=['y'])
y = df_model['y']

print(f"\n  Feature matrix X: {X.shape}")
print(f"  Target vector  y: {y.shape}")



# STEP 4 — STRATIFIED 80/20 TRAIN/TEST SPLIT

print("\n" + "=" * 70)
print("[STEP 4] STRATIFIED 80/20 TRAIN/TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

print(f"\n  Training set  : {X_train.shape[0]:,} records ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set      : {X_test.shape[0]:,}  records ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\n  Class distribution in training set:")
print(y_train.value_counts(normalize=True).round(3) * 100)
print(f"\n  Class distribution in test set:")
print(y_test.value_counts(normalize=True).round(3) * 100)
print("\n   Stratification confirmed — class ratios preserved in both sets.")

# Feature Scaling (fit on train, transform both)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n   Min-Max scaling applied (fit on training set only).")



# STEP 5 — APPLY SMOTE TO TRAINING SET ONLY

print("\n" + "=" * 70)
print("[STEP 5] APPLYING SMOTE TO TRAINING SET")
print("=" * 70)

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\n  Before SMOTE — Training set class distribution:")
print(pd.Series(y_train).value_counts())
print(f"\n  After SMOTE  — Training set class distribution:")
print(pd.Series(y_train_smote).value_counts())
print(f"\n  Training set size after SMOTE: {len(X_train_smote):,}")
print("   SMOTE applied to training set only. Test set remains untouched.")



# HELPER FUNCTIONS

def evaluate_model(model, X_tr, y_tr, X_te, y_te, cv_folds=10):
    """Train model, run cross-validation, evaluate on test set."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                          random_state=RANDOM_STATE)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X_tr, y_tr,
                                cv=skf, scoring=scoring,
                                return_train_score=False)
    # Fit on full training set for test evaluation
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] \
             if hasattr(model, 'predict_proba') \
             else model.decision_function(X_te)

    results = {
        'CV_Accuracy_Mean':  cv_results['test_accuracy'].mean(),
        'CV_Accuracy_Std':   cv_results['test_accuracy'].std(),
        'CV_Precision_Mean': cv_results['test_precision'].mean(),
        'CV_Precision_Std':  cv_results['test_precision'].std(),
        'CV_Recall_Mean':    cv_results['test_recall'].mean(),
        'CV_Recall_Std':     cv_results['test_recall'].std(),
        'CV_F1_Mean':        cv_results['test_f1'].mean(),
        'CV_F1_Std':         cv_results['test_f1'].std(),
        'CV_AUC_Mean':       cv_results['test_roc_auc'].mean(),
        'CV_AUC_Std':        cv_results['test_roc_auc'].std(),
        'Test_Accuracy':     accuracy_score(y_te, y_pred),
        'Test_Precision':    precision_score(y_te, y_pred),
        'Test_Recall':       recall_score(y_te, y_pred),
        'Test_F1':           f1_score(y_te, y_pred),
        'Test_AUC':          roc_auc_score(y_te, y_prob),
    }
    return results, y_pred, y_prob


def print_results(name, results):
    """Print formatted results for a model."""
    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"  {'Metric':<20} {'CV Mean':>10} {'CV Std':>10} {'Test':>10}")
    print(f"  {'─'*55}")
    metrics = [('Accuracy',  'CV_Accuracy_Mean',  'CV_Accuracy_Std',  'Test_Accuracy'),
               ('Precision', 'CV_Precision_Mean', 'CV_Precision_Std', 'Test_Precision'),
               ('Recall',    'CV_Recall_Mean',    'CV_Recall_Std',    'Test_Recall'),
               ('F1-Score',  'CV_F1_Mean',        'CV_F1_Std',        'Test_F1'),
               ('AUC-ROC',   'CV_AUC_Mean',       'CV_AUC_Std',       'Test_AUC')]
    for label, cv_m, cv_s, te in metrics:
        print(f"  {label:<20} {results[cv_m]:>10.4f} "
              f"{results[cv_s]:>10.4f} {results[te]:>10.4f}")



# EXPERIMENT 1 — BASELINE: ALL 6 MODELS WITHOUT SMOTE

print("\n" + "=" * 70)
print("EXPERIMENT 1 — BASELINE: ALL 6 MODELS WITHOUT SMOTE")
print("(Replicating approach of Moro et al. 2014 & Tran et al. 2023)")
print("=" * 70)

models_exp1 = {
    'Logistic Regression': LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(
        random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost':             XGBClassifier(
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    'SVM':                 CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=2000), cv=3),
    'KNN':                 KNeighborsClassifier(),
}

results_exp1 = {}
preds_exp1   = {}
probs_exp1   = {}

for name, model in models_exp1.items():
    print(f"\n  Training {name}...")
    res, pred, prob = evaluate_model(
        model, X_train_scaled, y_train,
        X_test_scaled, y_test)
    results_exp1[name] = res
    preds_exp1[name]   = pred
    probs_exp1[name]   = prob
    print_results(name, res)

exp1_df = pd.DataFrame(results_exp1).T
print("\n\n  EXPERIMENT 1 SUMMARY TABLE")
print(exp1_df[['Test_Accuracy','Test_Precision',
               'Test_Recall','Test_F1','Test_AUC']].round(4).to_string())
exp1_df.to_csv('experiment1_results.csv')
print("\n   Experiment 1 results saved to experiment1_results.csv")

# ── Format and print Experiment 1 in Mean+/-Std 
def format_mean_std(results_dict):
    """Format CV results as Mean+/-Std for supervisor reporting."""
    rows = []
    for model_name, res in results_dict.items():
        row = {
            'Model':     model_name,
            'Accuracy':  f"{res['CV_Accuracy_Mean']:.4f}+/-{res['CV_Accuracy_Std']:.4f}",
            'Precision': f"{res['CV_Precision_Mean']:.4f}+/-{res['CV_Precision_Std']:.4f}",
            'Recall':    f"{res['CV_Recall_Mean']:.4f}+/-{res['CV_Recall_Std']:.4f}",
            'F1-Score':  f"{res['CV_F1_Mean']:.4f}+/-{res['CV_F1_Std']:.4f}",
            'AUC-ROC':   f"{res['CV_AUC_Mean']:.4f}+/-{res['CV_AUC_Std']:.4f}",
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')

print("\n\n  EXPERIMENT 1 — Cross-Validation Results (Mean+/-Std)")
print("  " + "─" * 90)
print(format_mean_std(results_exp1).to_string())
format_mean_std(results_exp1).to_csv('exp1_mean_std_results.csv')
print("\n   Experiment 1 Mean+/-Std saved to exp1_mean_std_results.csv")


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — IMPROVED: ALL 6 MODELS WITH SMOTE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT 2 — IMPROVED: ALL 6 MODELS WITH SMOTE")
print("(New contribution — first systematic SMOTE evaluation on this dataset)")
print("=" * 70)

models_exp2 = {
    'Logistic Regression': LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(
        random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost':             XGBClassifier(
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    'SVM':                 CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=2000), cv=3),
    'KNN':                 KNeighborsClassifier(),
}

results_exp2 = {}
preds_exp2   = {}
probs_exp2   = {}

for name, model in models_exp2.items():
    print(f"\n  Training {name} with SMOTE...")
    res, pred, prob = evaluate_model(
        model, X_train_smote, y_train_smote,
        X_test_scaled, y_test)
    results_exp2[name] = res
    preds_exp2[name]   = pred
    probs_exp2[name]   = prob
    print_results(name, res)

exp2_df = pd.DataFrame(results_exp2).T
print("\n\n  EXPERIMENT 2 SUMMARY TABLE")
print(exp2_df[['Test_Accuracy','Test_Precision',
               'Test_Recall','Test_F1','Test_AUC']].round(4).to_string())
exp2_df.to_csv('experiment2_results.csv')
print("\n   Experiment 2 results saved to experiment2_results.csv")

print("\n\n  EXPERIMENT 2 — Cross-Validation Results (Mean+/-Std)")
print("  " + "─" * 90)
print(format_mean_std(results_exp2).to_string())
format_mean_std(results_exp2).to_csv('exp2_mean_std_results.csv')
print("\n   Experiment 2 Mean+/-Std saved to exp2_mean_std_results.csv")

# Identify top 3 models from Experiment 2 by AUC
top3 = exp2_df['Test_AUC'].nlargest(3).index.tolist()
print(f"\n  🏆 Top 3 models for Experiment 3: {top3}")


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — OPTIMISED: HYPERPARAMETER TUNING ON TOP 3 MODELS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT 3 — OPTIMISED: HYPERPARAMETER TUNING ON TOP 3 MODELS")
print("(GridSearchCV with 10-Fold Stratified Cross-Validation)")
print("=" * 70)

param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'params': {'C': [0.01, 0.1, 1, 10, 100],
                   'solver': ['liblinear', 'lbfgs']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'params': {'max_depth': [5, 10, 15, 20, None],
                   'min_samples_split': [2, 5, 10],
                   'criterion': ['gini', 'entropy']}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {'n_estimators': [100, 200, 300],
                   'max_depth': [10, 20, None],
                   'max_features': ['sqrt', 'log2']}
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=RANDOM_STATE,
                               eval_metric='logloss', verbosity=0),
        'params': {'learning_rate': [0.01, 0.1, 0.2],
                   'n_estimators': [100, 200, 300],
                   'max_depth': [3, 5, 7]}
    },
    'SVM': {
        'model': CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE, max_iter=2000), cv=3),
        'params': {'C': [0.1, 1, 10, 100],
                   'gamma': ['scale', 'auto'],
                   'kernel': ['rbf', 'linear']}
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5, 7, 9, 11, 15],
                   'metric': ['euclidean', 'manhattan']}
    },
}

results_exp3 = {}
preds_exp3   = {}
probs_exp3   = {}
best_params  = {}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

for name in top3:
    print(f"\n  Tuning {name}...")
    grid_info = param_grids[name]
    gs = GridSearchCV(
        grid_info['model'],
        grid_info['params'],
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_train_smote, y_train_smote)
    best_params[name] = gs.best_params_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV AUC: {gs.best_score_:.4f}")

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1] \
             if hasattr(best_model, 'predict_proba') \
             else best_model.decision_function(X_test_scaled)

    # Run final cross-validation with best model
    res, _, _ = evaluate_model(
        best_model, X_train_smote, y_train_smote,
        X_test_scaled, y_test)
    results_exp3[name] = res
    preds_exp3[name]   = y_pred
    probs_exp3[name]   = y_prob
    print_results(name, res)

exp3_df = pd.DataFrame(results_exp3).T
print("\n\n  EXPERIMENT 3 SUMMARY TABLE")
print(exp3_df[['Test_Accuracy','Test_Precision',
               'Test_Recall','Test_F1','Test_AUC']].round(4).to_string())
exp3_df.to_csv('experiment3_results.csv')
print("\n   Experiment 3 results saved to experiment3_results.csv")

print("\n\n  EXPERIMENT 3 — Cross-Validation Results (Mean+/-Std)")
print("  " + "─" * 90)
print(format_mean_std(results_exp3).to_string())
format_mean_std(results_exp3).to_csv('exp3_mean_std_results.csv')
print("\n   Experiment 3 Mean+/-Std saved to exp3_mean_std_results.csv")


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARKING AGAINST PUBLISHED WORK
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BENCHMARKING AGAINST PUBLISHED WORK")
print("=" * 70)

# Published results
published = {
    'Logistic Regression': {'Moro_AUC': 0.75,  'Tran_Acc': 0.8727},
    'Decision Tree':        {'Moro_AUC': 0.73,  'Tran_Acc': 0.9250},
    'Random Forest':        {'Moro_AUC': None,  'Tran_Acc': 0.9725},
    'XGBoost':              {'Moro_AUC': None,  'Tran_Acc': None},
    'SVM':                  {'Moro_AUC': 0.78,  'Tran_Acc': 0.9210},
    'KNN':                  {'Moro_AUC': None,  'Tran_Acc': 0.9180},
}

print(f"\n  {'Model':<22} {'Moro AUC':>10} {'Tran Acc':>10} "
      f"{'Exp1 AUC':>10} {'Exp2 AUC':>10}")
print("  " + "─" * 65)
for name in results_exp1.keys():
    moro = published[name]['Moro_AUC']
    tran = published[name]['Tran_Acc']
    e1   = results_exp1[name]['Test_AUC']
    e2   = results_exp2[name]['Test_AUC']
    moro_s = f"{moro:.4f}" if moro else "N/A"
    tran_s = f"{tran:.4f}" if tran else "N/A"
    print(f"  {name:<22} {moro_s:>10} {tran_s:>10} {e1:>10.4f} {e2:>10.4f}")


# ════════════════════════════════════════════════════════════════════════════
# RESULTS VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING RESULTS VISUALISATIONS")
print("=" * 70)

model_names = list(results_exp1.keys())
short_names = ['LR', 'DT', 'RF', 'XGB', 'SVM', 'KNN']
metrics     = ['Test_Accuracy', 'Test_Precision',
               'Test_Recall', 'Test_F1', 'Test_AUC']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

# ── Figure 6: Grouped Bar Chart — Exp 1 vs Exp 2 (AUC) 
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

x     = np.arange(len(model_names))
width = 0.35

e1_auc = [results_exp1[n]['Test_AUC'] for n in model_names]
e2_auc = [results_exp2[n]['Test_AUC'] for n in model_names]

bars1 = ax.bar(x - width/2, e1_auc, width, label='Exp 1 — Without SMOTE',
               color=BLUE_LIGHT, edgecolor=WHITE, linewidth=1.2)
bars2 = ax.bar(x + width/2, e2_auc, width, label='Exp 2 — With SMOTE',
               color=BLUE_DARK, edgecolor=WHITE, linewidth=1.2)

for bar, val in zip(bars1, e1_auc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color=BLUE_DARK)
for bar, val in zip(bars2, e2_auc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color=WHITE if val > 0.85 else BLUE_DARK)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=11)
ax.set_ylabel('AUC-ROC Score', fontsize=11)
ax.set_title('AUC-ROC Comparison: Without SMOTE vs With SMOTE\n'
             'Experiment 1 vs Experiment 2', pad=12, color=BLUE_DARK)
ax.set_ylim(0.5, 1.05)
ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color(BLUE_MID)
plt.tight_layout()
plt.savefig('figure6_exp1_vs_exp2_auc.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("\n  Figure 6 saved — Exp 1 vs Exp 2 AUC comparison")

# ── Figure 7: All Metrics Comparison — Experiment 2 (With SMOTE) 
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.patch.set_facecolor(WHITE)
fig.suptitle('All Metrics Comparison Across 6 Models (With SMOTE)',
             fontsize=13, fontweight='bold', color=BLUE_DARK, y=1.02)

colors_bar = [BLUE_DARK, BLUE_MID, TEAL, '#E85D04', BLUE_LIGHT, '#6A994E']

for ax, metric, label in zip(axes, metrics, metric_labels):
    ax.set_facecolor(GRAY)
    vals = [results_exp2[n][metric] for n in model_names]
    bars = ax.bar(short_names, vals, color=colors_bar,
                  edgecolor=WHITE, linewidth=1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_title(label, color=BLUE_DARK)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, color='#DDDDDD', linewidth=0.6, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(BLUE_MID)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figure7_all_metrics_exp2.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("  Figure 7 saved — All metrics Experiment 2")

# ── Figure 8: Recall Comparison — SMOTE Impact 
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

e1_recall = [results_exp1[n]['Test_Recall'] for n in model_names]
e2_recall = [results_exp2[n]['Test_Recall'] for n in model_names]

x     = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, e1_recall, width, label='Without SMOTE',
               color=BLUE_LIGHT, edgecolor=WHITE, linewidth=1.2)
bars2 = ax.bar(x + width/2, e2_recall, width, label='With SMOTE',
               color=TEAL, edgecolor=WHITE, linewidth=1.2)

for bar, val in zip(bars1, e1_recall):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color=BLUE_DARK)
for bar, val in zip(bars2, e2_recall):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color=BLUE_DARK)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=11)
ax.set_ylabel('Recall Score', fontsize=11)
ax.set_title('Impact of SMOTE on Recall (Minority Class Detection)\n'
             'Higher Recall = More Subscribers Correctly Identified',
             pad=12, color=BLUE_DARK)
ax.set_ylim(0, 1.15)
ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color(BLUE_MID)
plt.tight_layout()
plt.savefig('figure8_smote_recall_impact.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("  Figure 8 saved — SMOTE impact on Recall")

# ── Figure 9: ROC Curves — Experiment 2 All Models 
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

roc_colors = [BLUE_DARK, BLUE_MID, TEAL, '#E85D04', BLUE_LIGHT, '#6A994E']

for (name, prob), color in zip(probs_exp2.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = results_exp2[name]['Test_AUC']
    short = short_names[list(model_names).index(name)]
    ax.plot(fpr, tpr, color=color, linewidth=2.0,
            label=f'{short} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Classifier')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Recall/Sensitivity)', fontsize=11)
ax.set_title('ROC Curves for All 6 Models (With SMOTE)',
             pad=12, color=BLUE_DARK, fontsize=12)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6, linestyle='--')
ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.6, linestyle='--')
ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color(BLUE_MID)
plt.tight_layout()
plt.savefig('figure9_roc_curves_exp2.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("  Figure 9 saved — ROC Curves Experiment 2")

# ── Figure 10: Confusion Matrices — Experiment 2 
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor(WHITE)
fig.suptitle('Confusion Matrices for All 6 Models (With SMOTE)',
             fontsize=13, fontweight='bold', color=BLUE_DARK)
axes = axes.flatten()

for ax, (name, pred) in zip(axes, preds_exp2.items()):
    cm = confusion_matrix(y_test, pred)
    short = short_names[list(model_names).index(name)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Sub.', 'Subscribed'])
    disp.plot(ax=ax, colorbar=False,
              cmap='Blues', values_format='d')
    ax.set_title(short, color=BLUE_DARK, fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)

plt.tight_layout()
plt.savefig('figure10_confusion_matrices.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("  Figure 10 saved — Confusion Matrices")

# ── Figure 11: Cross-Validation Mean ± Std — AUC 
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

cv_means = [results_exp2[n]['CV_AUC_Mean'] for n in model_names]
cv_stds  = [results_exp2[n]['CV_AUC_Std']  for n in model_names]

bars = ax.bar(short_names, cv_means, color=colors_bar,
              edgecolor=WHITE, linewidth=1.2, width=0.55)
ax.errorbar(short_names, cv_means, yerr=cv_stds,
            fmt='none', color=BLUE_DARK, capsize=6,
            linewidth=2.0, capthick=2.0)

for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            mean + std + 0.01,
            f'{mean:.3f}\n±{std:.3f}',
            ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color=BLUE_DARK)

ax.set_ylabel('AUC-ROC Score (Mean ± Std)', fontsize=11)
ax.set_title('Fold Cross-Validation AUC-ROC\n'
             'Mean ± Standard Deviation (With SMOTE)',
             pad=12, color=BLUE_DARK)
ax.set_ylim(0.5, 1.15)
ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color(BLUE_MID)
plt.tight_layout()
plt.savefig('figure11_cv_auc_mean_std.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("  Figure 11 saved — Cross-Validation AUC Mean ± Std")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE — FINAL SUMMARY")
print("=" * 70)

print("\n  EXPERIMENT 1 — Baseline (No SMOTE)")
print(exp1_df[['Test_Accuracy','Test_Precision',
               'Test_Recall','Test_F1','Test_AUC']].round(4).to_string())

print("\n  EXPERIMENT 2 — With SMOTE")
print(exp2_df[['Test_Accuracy','Test_Precision',
               'Test_Recall','Test_F1','Test_AUC']].round(4).to_string())

print(f"\n  EXPERIMENT 3 — Tuned Top 3 Models ({', '.join(top3)})")
print(exp3_df[['Test_Accuracy','Test_Precision',
               'Test_Recall','Test_F1','Test_AUC']].round(4).to_string())

print("\n  Files saved:")
print("    → experiment1_results.csv")
print("    → experiment2_results.csv")
print("    → experiment3_results.csv")
print("    → exp1_mean_std_results.csv  (Mean+/-Std format)")
print("    → exp2_mean_std_results.csv  (Mean+/-Std format)")
print("    → exp3_mean_std_results.csv  (Mean+/-Std format)")
print("    → figure6_exp1_vs_exp2_auc.png")
print("    → figure7_all_metrics_exp2.png")
print("    → figure8_smote_recall_impact.png")
print("    → figure9_roc_curves_exp2.png")
print("    → figure10_confusion_matrices.png")
print("    → figure11_cv_auc_mean_std.png")

print("\n" + "=" * 70)
print("✅ MRP Experiment Pipeline Complete")
print("=" * 70)
