import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
 
# Load Dataset
df = pd.read_csv('bank-full.csv', sep=';')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Class distribution:\n{df['y'].value_counts()}")
pct = df['y'].value_counts(normalize=True) * 100
print(f"Class percentages:\n{pct.round(1)}")
 
# Colour Palette
BLUE_DARK  = '#1F3864'
BLUE_MID   = '#2E75B6'
BLUE_LIGHT = '#BDD7EE'
TEAL       = '#00B0D7'
WHITE      = '#FFFFFF'
GRAY       = '#F5F5F5'
TEXT_COLOR = '#333333'
 
plt.rcParams.update({
    'font.family':     'Arial',
    'axes.titlesize':  13,
    'axes.labelsize':  11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.titleweight':'bold',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       150,
})
 
 
# FIGURE 1 — Class Distribution: Subscribed vs Not Subscribed

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)
 
counts     = df['y'].value_counts()
labels     = ['No (Did Not Subscribe)', 'Yes (Subscribed)']
values     = [counts['no'], counts['yes']]
colors     = [BLUE_LIGHT, BLUE_DARK]
pcts       = [v / sum(values) * 100 for v in values]
 
bars = ax.bar(labels, values, color=colors, edgecolor=WHITE,
              linewidth=1.5, width=0.5)
 
for bar, val, pct in zip(bars, values, pcts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 400,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=TEXT_COLOR)
 
ax.set_title('Class Distribution: Subscribed vs Not Subscribed',
             pad=14, color=BLUE_DARK)
ax.set_ylabel('Number of Customers', color=TEXT_COLOR)
ax.set_ylim(0, max(values) * 1.18)
ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(colors=TEXT_COLOR)
ax.spines['left'].set_color(BLUE_MID)
ax.spines['bottom'].set_color(BLUE_MID)
 
plt.tight_layout()
plt.savefig('figure1_class_distribution.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 1 saved.")


# FIGURE 2 — Distribution of Age, Balance and Call Duration

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(WHITE)
fig.suptitle('Distribution of Key Numerical Variables',
             fontsize=14, fontweight='bold', color=BLUE_DARK, y=1.02)
 
variables = [
    ('age',      'Age (Years)',              BLUE_DARK),
    ('balance',  'Account Balance (Euros)',  BLUE_MID),
    ('duration', 'Call Duration (Seconds)',  TEAL),
]
 
for ax, (col, xlabel, color) in zip(axes, variables):
    ax.set_facecolor(GRAY)
    data = df[col]
 
    # Cap balance outliers for better visualisation
    if col == 'balance':
        data = data[data.between(data.quantile(0.01),
                                 data.quantile(0.99))]
 
    ax.hist(data, bins=40, color=color, edgecolor=WHITE,
            linewidth=0.6, alpha=0.9)
 
    # Add mean line
    mean_val = data.mean()
    ax.axvline(mean_val, color=BLUE_DARK if color != BLUE_DARK else TEAL,
               linewidth=2, linestyle='--', label=f'Mean: {mean_val:,.0f}')
 
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel('Frequency', color=TEXT_COLOR)
    ax.set_title(col.capitalize(), color=BLUE_DARK, fontweight='bold')
    ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BLUE_MID)
    ax.spines['bottom'].set_color(BLUE_MID)
    ax.legend(fontsize=9, framealpha=0.8)
 
plt.tight_layout()
plt.savefig('figure2_numeric_distributions.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 2 saved.")


# FIGURE 3 — Subscription Rate by Job Type

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)
 
job_sub = (df.groupby('job')['y']
             .apply(lambda x: (x == 'yes').sum() / len(x) * 100)
             .sort_values(ascending=True)
             .reset_index())
job_sub.columns = ['job', 'subscription_rate']
 
# Colour bars — highlight top 2 performers
bar_colors = [BLUE_DARK if rate >= job_sub['subscription_rate'].nlargest(2).min()
              else BLUE_MID if rate >= job_sub['subscription_rate'].median()
              else BLUE_LIGHT
              for rate in job_sub['subscription_rate']]
 
bars = ax.barh(job_sub['job'], job_sub['subscription_rate'],
               color=bar_colors, edgecolor=WHITE, linewidth=1.2, height=0.65)
 
for bar, val in zip(bars, job_sub['subscription_rate']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=10,
            fontweight='bold', color=TEXT_COLOR)
 
ax.set_xlabel('Subscription Rate (%)', color=TEXT_COLOR)
ax.set_title('Subscription Rate by Customer Job Type',
             pad=14, color=BLUE_DARK)
ax.set_xlim(0, job_sub['subscription_rate'].max() * 1.2)
ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(colors=TEXT_COLOR)
ax.spines['left'].set_color(BLUE_MID)
ax.spines['bottom'].set_color(BLUE_MID)
 
# Legend
legend_patches = [
    mpatches.Patch(color=BLUE_DARK,  label='Top performers'),
    mpatches.Patch(color=BLUE_MID,   label='Above average'),
    mpatches.Patch(color=BLUE_LIGHT, label='Below average'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.8)
 
plt.tight_layout()
plt.savefig('figure3_subscription_by_job.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 3 saved.")


# FIGURE 4 — Subscription Rate by Education Level

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)
 
edu_order = ['primary', 'secondary', 'tertiary', 'unknown']
edu_sub = (df[df['education'].isin(edu_order)]
             .groupby('education')['y']
             .apply(lambda x: (x == 'yes').sum() / len(x) * 100)
             .reindex(edu_order)
             .reset_index())
edu_sub.columns = ['education', 'subscription_rate']
 
edu_colors = [BLUE_LIGHT, BLUE_MID, BLUE_DARK, '#AAAAAA']
edu_labels = ['Primary', 'Secondary', 'Tertiary', 'Unknown']
 
bars = ax.bar(edu_labels, edu_sub['subscription_rate'],
              color=edu_colors, edgecolor=WHITE, linewidth=1.5, width=0.55)
 
for bar, val in zip(bars, edu_sub['subscription_rate']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=TEXT_COLOR)
 
ax.set_ylabel('Subscription Rate (%)', color=TEXT_COLOR)
ax.set_xlabel('Education Level', color=TEXT_COLOR)
ax.set_title('Subscription Rate by Education Level',
             pad=14, color=BLUE_DARK)
ax.set_ylim(0, edu_sub['subscription_rate'].max() * 1.2)
ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(colors=TEXT_COLOR)
ax.spines['left'].set_color(BLUE_MID)
ax.spines['bottom'].set_color(BLUE_MID)
 
plt.tight_layout()
plt.savefig('figure4_subscription_by_education.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 4 saved.")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Correlation Heatmap of Numerical Features
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(WHITE)
 
# Include more variables - encode binary/ordinal categoricals
df_corr = df.copy()
df_corr['subscribed']   = (df['y'] == 'yes').astype(int)
df_corr['housing_bin']  = (df['housing'] == 'yes').astype(int)
df_corr['loan_bin']     = (df['loan'] == 'yes').astype(int)
df_corr['default_bin']  = (df['default'] == 'yes').astype(int)
df_corr['edu_ord']      = df['education'].map(
                              {'primary': 1, 'secondary': 2,
                               'tertiary': 3, 'unknown': 0})
df_corr['marital_bin']  = (df['marital'] == 'married').astype(int)
 
# Encode poutcome ordinally: unknown=0, failure=1, other=2, success=3
df_corr['poutcome_ord'] = df['poutcome'].map(
                              {'unknown': 0, 'failure': 1,
                               'other': 2,   'success': 3})
 
cols = ['age', 'balance', 'duration', 'campaign',
        'pdays', 'previous', 'poutcome_ord',
        'housing_bin', 'loan_bin', 'default_bin',
        'edu_ord', 'marital_bin', 'subscribed']
 
labels = ['Age', 'Balance', 'Duration', 'Campaign',
          'P-Days', 'Previous', 'Prev. Outcome',
          'Housing\nLoan', 'Personal\nLoan', 'Default',
          'Education', 'Married', 'Subscribed']
 
corr_matrix = df_corr[cols].corr()
 
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-0.5, vmax=0.5,
    linewidths=1.0,
    linecolor='white',
    annot_kws={'size': 8.5, 'weight': 'bold'},
    ax=ax,
    cbar_kws={
        'shrink': 0.75,
        'label': 'Correlation Coefficient',
        'ticks': [-0.5, -0.25, 0, 0.25, 0.5]
    }
)
 
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9.5)
ax.set_yticklabels(labels, rotation=0, fontsize=9.5)
 
ax.set_title(
    'Correlation Heatmap of Features\n(including Target Variable: Subscribed)',
    pad=16, color=BLUE_DARK, fontsize=13, fontweight='bold')
 
plt.tight_layout()
plt.savefig('figure5_correlation_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor=WHITE)
plt.close()
print("Figure 5 saved.")
 
print("\n All 5 figures generated successfully!")
print("Files saved:")
print("  → figure1_class_distribution.png")
print("  → figure2_numeric_distributions.png")
print("  → figure3_subscription_by_job.png")
print("  → figure4_subscription_by_education.png")
print("  → figure5_correlation_heatmap.png")
