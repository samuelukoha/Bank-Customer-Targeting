"""
================================================================================
MRP — Feature Importance Analysis (Research Question 4)
Samuel Ukoha | Toronto Metropolitan University | 2026
================================================================================
Which customer characteristics are the strongest predictors
of term deposit subscription and how can these inform bank marketing?
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BLUE_DARK  = '#1F3864'
BLUE_MID   = '#2E75B6'
BLUE_LIGHT = '#BDD7EE'
TEAL       = '#00B0D7'
ORANGE     = '#E85D04'
GREEN      = '#2D6A4F'
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

# ── Load and preprocess ──────────────────────────────────────────────────────
print("Loading and preprocessing dataset...")
df = pd.read_csv('bank-full.csv', sep=';')

df['y_bin'] = (df['y'] == 'yes').astype(int)
df['education'] = df['education'].map(
    {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3})
df['poutcome'] = df['poutcome'].map(
    {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3})
for c in ['default', 'housing', 'loan']:
    df[c] = (df[c] == 'yes').astype(int)
df['contact'] = df['contact'].map(
    {'unknown': 0, 'telephone': 1, 'cellular': 2})
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
df['month'] = df['month'].map(month_map)
df = pd.get_dummies(df, columns=['job', 'marital'], drop_first=True)

X = df.drop(columns=['y', 'y_bin'])
y = df['y_bin']

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Features: {len(feature_names)}")
print(f"Training set (after SMOTE): {len(X_train_smote)}")


# ════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — Random Forest
# ════════════════════════════════════════════════════════════════════════════
print("\nTraining Random Forest for feature importance...")
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)

rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest — Top 15 Features:")
print(rf_importance.head(15).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — XGBoost
# ════════════════════════════════════════════════════════════════════════════
print("\nTraining XGBoost for feature importance...")
xgb = XGBClassifier(n_estimators=200, random_state=RANDOM_STATE,
                     eval_metric='logloss', verbosity=0)
xgb.fit(X_train_smote, y_train_smote)

xgb_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nXGBoost — Top 15 Features:")
print(xgb_importance.head(15).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — Logistic Regression (Coefficients)
# ════════════════════════════════════════════════════════════════════════════
print("\nTraining Logistic Regression for feature coefficients...")
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr.fit(X_train_smote, y_train_smote)

lr_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr.coef_[0],
    'Abs_Coefficient': np.abs(lr.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nLogistic Regression — Top 15 Features (by absolute coefficient):")
print(lr_importance.head(15)[['Feature', 'Coefficient', 'Abs_Coefficient']].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# CONSENSUS RANKING — Average across models
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CONSENSUS FEATURE IMPORTANCE RANKING")
print("=" * 60)

# Normalize all importance scores to [0, 1]
rf_norm = rf_importance.copy()
rf_norm['Importance'] = rf_norm['Importance'] / rf_norm['Importance'].max()

xgb_norm = xgb_importance.copy()
xgb_norm['Importance'] = xgb_norm['Importance'] / xgb_norm['Importance'].max()

lr_norm = lr_importance.copy()
lr_norm['Importance'] = lr_norm['Abs_Coefficient'] / lr_norm['Abs_Coefficient'].max()

# Merge
consensus = rf_norm[['Feature', 'Importance']].rename(columns={'Importance': 'RF'})
consensus = consensus.merge(
    xgb_norm[['Feature', 'Importance']].rename(columns={'Importance': 'XGB'}),
    on='Feature')
consensus = consensus.merge(
    lr_norm[['Feature', 'Importance']].rename(columns={'Importance': 'LR'}),
    on='Feature')
consensus['Average'] = consensus[['RF', 'XGB', 'LR']].mean(axis=1)
consensus = consensus.sort_values('Average', ascending=False)

print("\nTop 15 Features — Consensus Ranking (Normalized Average):")
print(consensus.head(15).to_string(index=False))

# Save to CSV
consensus.to_csv('feature_importance_consensus.csv', index=False)
rf_importance.to_csv('feature_importance_rf.csv', index=False)
xgb_importance.to_csv('feature_importance_xgb.csv', index=False)
lr_importance.to_csv('feature_importance_lr.csv', index=False)
print("\nCSV files saved.")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Top 15 Feature Importance Comparison (RF vs XGBoost)
# ════════════════════════════════════════════════════════════════════════════
top15 = consensus.head(15)['Feature'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(WHITE)
fig.suptitle('Top 15 Feature Importance: Random Forest vs XGBoost',
             fontsize=14, fontweight='bold', color=BLUE_DARK, y=1.02)

# Random Forest
ax1 = axes[0]
ax1.set_facecolor(GRAY)
rf_top = rf_importance[rf_importance['Feature'].isin(top15)].sort_values('Importance')
colors_rf = [BLUE_DARK if f == rf_importance.iloc[0]['Feature']
             else BLUE_MID if i >= len(rf_top) - 3
             else BLUE_LIGHT
             for i, f in enumerate(rf_top['Feature'])]
ax1.barh(rf_top['Feature'], rf_top['Importance'],
         color=colors_rf, edgecolor=WHITE, linewidth=0.8)
for i, (val, feat) in enumerate(zip(rf_top['Importance'], rf_top['Feature'])):
    ax1.text(val + 0.002, i, f'{val:.4f}', va='center',
             fontsize=8, fontweight='bold', color=BLUE_DARK)
ax1.set_title('Random Forest', color=BLUE_DARK, fontsize=12)
ax1.set_xlabel('Feature Importance', color=BLUE_DARK)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color(BLUE_MID)
ax1.xaxis.grid(True, color='#CCCCCC', linewidth=0.6, linestyle='--')
ax1.set_axisbelow(True)

# XGBoost
ax2 = axes[1]
ax2.set_facecolor(GRAY)
xgb_top = xgb_importance[xgb_importance['Feature'].isin(top15)].sort_values('Importance')
colors_xgb = [ORANGE if f == xgb_importance.iloc[0]['Feature']
              else TEAL if i >= len(xgb_top) - 3
              else BLUE_LIGHT
              for i, f in enumerate(xgb_top['Feature'])]
ax2.barh(xgb_top['Feature'], xgb_top['Importance'],
         color=colors_xgb, edgecolor=WHITE, linewidth=0.8)
for i, (val, feat) in enumerate(zip(xgb_top['Importance'], xgb_top['Feature'])):
    ax2.text(val + 0.002, i, f'{val:.4f}', va='center',
             fontsize=8, fontweight='bold', color=BLUE_DARK)
ax2.set_title('XGBoost', color=BLUE_DARK, fontsize=12)
ax2.set_xlabel('Feature Importance', color=BLUE_DARK)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color(BLUE_MID)
ax2.xaxis.grid(True, color='#CCCCCC', linewidth=0.6, linestyle='--')
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figure12_feature_importance_rf_xgb.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 12 saved — RF vs XGBoost feature importance")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Consensus Feature Importance (Top 15)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

top15_data = consensus.head(15).sort_values('Average')

colors_con = [BLUE_DARK if i >= len(top15_data) - 1
              else TEAL if i >= len(top15_data) - 3
              else BLUE_MID if i >= len(top15_data) - 5
              else BLUE_LIGHT
              for i in range(len(top15_data))]

bars = ax.barh(top15_data['Feature'], top15_data['Average'],
               color=colors_con, edgecolor=WHITE, linewidth=1.0, height=0.65)
for i, (val, feat) in enumerate(zip(top15_data['Average'], top15_data['Feature'])):
    ax.text(val + 0.008, i, f'{val:.3f}', va='center',
            fontsize=9, fontweight='bold', color=BLUE_DARK)

ax.set_title('Consensus Feature Importance (Top 15)\n'
             'Normalized Average Across Random Forest, XGBoost, and Logistic Regression',
             pad=14, color=BLUE_DARK, fontsize=12)
ax.set_xlabel('Normalized Average Importance', fontsize=11, color=BLUE_DARK)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color(BLUE_MID)
ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figure13_consensus_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 13 saved — Consensus feature importance")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Logistic Regression Coefficients (Top 15)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

lr_top15 = lr_importance.head(15).sort_values('Coefficient')

colors_lr = [GREEN if c > 0 else '#CC3333' for c in lr_top15['Coefficient']]

bars = ax.barh(lr_top15['Feature'], lr_top15['Coefficient'],
               color=colors_lr, edgecolor=WHITE, linewidth=1.0, height=0.65)
for i, (val, feat) in enumerate(zip(lr_top15['Coefficient'], lr_top15['Feature'])):
    offset = 0.03 if val > 0 else -0.03
    ha = 'left' if val > 0 else 'right'
    ax.text(val + offset, i, f'{val:.3f}', va='center', ha=ha,
            fontsize=9, fontweight='bold', color=BLUE_DARK)

ax.axvline(0, color=BLUE_DARK, linewidth=1.2, linestyle='-')
ax.set_title('Logistic Regression Coefficients (Top 15 by Magnitude)\n'
             'Green = Increases Subscription Probability  |  '
             'Red = Decreases Subscription Probability',
             pad=14, color=BLUE_DARK, fontsize=11)
ax.set_xlabel('Logistic Regression Coefficient', fontsize=11, color=BLUE_DARK)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color(BLUE_MID)
ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)

# Legend
legend_patches = [
    mpatches.Patch(color=GREEN, label='Positive (increases subscription)'),
    mpatches.Patch(color='#CC3333', label='Negative (decreases subscription)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('figure14_lr_coefficients.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 14 saved — LR coefficients")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 60)
print("\nTop 5 most important features (consensus):")
for i, row in consensus.head(5).iterrows():
    print(f"  {row['Feature']:<20} RF={row['RF']:.3f}  XGB={row['XGB']:.3f}  LR={row['LR']:.3f}  Avg={row['Average']:.3f}")

print("\nFiles saved:")
print("  → feature_importance_consensus.csv")
print("  → feature_importance_rf.csv")
print("  → feature_importance_xgb.csv")
print("  → feature_importance_lr.csv")
print("  → figure13_feature_importance_rf_xgb.png")
print("  → figure14_consensus_feature_importance.png")
print("  → figure15_lr_coefficients.png")

print("\n Research Question 4 — ANSWERED")
print("=" * 60)
