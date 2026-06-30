"""
Generate figures for the Gryzzly dataset.
Run after metrics_empirical_triangle_clean_10000.csv has been produced.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})
BLUE   = '#2563EB'
ORANGE = '#EA580C'
GRAY   = '#6B7280'
BASE_DIR  = Path(__file__).resolve().parents[1]
FILES_DIR = BASE_DIR / "outputs"
OUT       = BASE_DIR / "figures" / "eng"
OUT.mkdir(parents=True, exist_ok=True)

# Load empirical-triangle results and merge in extra columns from the full run
df_new = pd.read_csv(FILES_DIR / "metrics_empirical_triangle_clean_10000.csv")
df_old = pd.read_csv(FILES_DIR / "metrics_results_full.csv")

extra_cols = ['project_id', 'n_employees', 'n_dependencies',
              'critical_path_tasks', 'avg_employee_efficiency']
df = df_new.merge(df_old[extra_cols], on='project_id', how='left')

if 'mc_success' in df.columns:
    df = df[df['mc_success'] == True].copy()
else:
    df = df.copy()
df = df[df['det_duration_days'] <= 500]
df = df[df['p50'] <= 600]
df = df.dropna(subset=['schedule_risk_ratio', 'p50', 'p90',
                        'det_duration_days', 'avg_employee_efficiency'])

print(f"Projects for analysis: {len(df)}")
print(df[['schedule_risk_ratio', 'det_duration_days', 'p50', 'p90',
          'avg_employee_efficiency', 'n_tasks']].describe().round(2))

# Figure A — Schedule risk ratio distribution
fig, ax = plt.subplots(figsize=(8, 4.5))

sns.histplot(df['schedule_risk_ratio'], bins=40, color=BLUE,
             edgecolor='white', linewidth=0.5, ax=ax)

med   = df['schedule_risk_ratio'].median()
mean_ = df['schedule_risk_ratio'].mean()
ax.axvline(med,   color=ORANGE, lw=2,   linestyle='--', label=f'Median = {med:.2f}')
ax.axvline(mean_, color=GRAY,   lw=1.5, linestyle=':',  label=f'Mean = {mean_:.2f}')
ax.set_xlim(left=0, right=0.5)
ax.set_xlabel('Schedule Reserve  (P90 − P50) / P50', labelpad=8)
ax.set_ylabel('Number of projects')
ax.set_title('Schedule Reserve Distribution\n(real Gryzzly projects, N = {:,})'.format(len(df)))
ax.legend()
fig.savefig(OUT / 'A_risk_distribution.png')
plt.close()
print("✓ A_risk_distribution.png")

# Figure B — Deterministic (CPM) estimate vs Monte Carlo P50
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(df['det_duration_days'], df['p50'],
           alpha=0.35, s=20, color=BLUE, edgecolors='none')
lim = max(df['det_duration_days'].max(), df['p50'].max()) * 1.05
ax.plot([0, lim], [0, lim], '--', color=GRAY, lw=1.5, label='CPM estimate = P50 (ideal)')

z  = np.polyfit(df['det_duration_days'], df['p50'], 1)
p  = np.poly1d(z)
xs = np.linspace(0, lim, 200)
ax.plot(xs, p(xs), color=ORANGE, lw=2, label=f'Trend: P50 ≈ {z[0]:.2f} × CPM')

ax.set_xlabel('Deterministic duration (CPM), days')
ax.set_ylabel('Monte Carlo P50, days')
ax.set_title('CPM vs Monte Carlo P50\nmedian delta = 0 days')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.legend()
fig.savefig(OUT / 'B_det_vs_p50.png')
plt.close()
print("✓ B_det_vs_p50.png")

# Figure C — P50 vs P90 uncertainty spread
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(df['p50'], df['p90'],
           alpha=0.35, s=20, color=BLUE, edgecolors='none')
lim = max(df['p50'].max(), df['p90'].max()) * 1.05
ax.plot([0, lim], [0, lim],        '--', color=GRAY,   lw=1.5, label='P50 = P90')
ax.plot([0, lim], [0, lim * 1.31], ':',  color=ORANGE, lw=1.5, label='+31% buffer')

ax.set_xlabel('Monte Carlo P50, days')
ax.set_ylabel('Monte Carlo P90, days')
ax.set_title('Uncertainty Spread: P50 to P90\nmedian P90/P50 = 1.31×')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.legend()
fig.savefig(OUT / 'C_p50_vs_p90.png')
plt.close()
print("✓ C_p50_vs_p90.png")

# Figure D — Schedule reserve vs project size
fig, ax = plt.subplots(figsize=(8, 4.5))

bins   = [2, 4, 7, 11, 16, 25, 160]
labels = ['2–3', '4–6', '7–10', '11–15', '16–24', '25+']
df['task_bin'] = pd.cut(df['n_tasks'], bins=bins, labels=labels)
binned = df.groupby('task_bin', observed=True)['schedule_risk_ratio'].agg(
    ['median', 'mean', 'count']).reset_index()
binned.columns = ['bin', 'median', 'mean', 'count']

ax.bar(binned['bin'], binned['median'], color=BLUE,
       edgecolor='white', linewidth=0.5, label='Median reserve')
for i, row in binned.iterrows():
    ax.text(i, row['median'] + 0.003, f'n={int(row["count"])}',
            ha='center', va='bottom', fontsize=9, color=GRAY)

ax.axhline(df['schedule_risk_ratio'].median(), color=ORANGE,
           lw=1.5, linestyle='--', label='Overall median')
ax.set_xlabel('Number of tasks in project')
ax.set_ylabel('Schedule reserve (median)')
ax.set_title('Schedule Reserve by Project Size')
ax.legend()
fig.savefig(OUT / 'D_risk_vs_size.png')
plt.close()
print("✓ D_risk_vs_size.png")

# Figure F — Spearman correlation matrix
corr_cols = ['n_tasks', 'n_employees', 'n_dependencies',
             'critical_path_tasks', 'avg_employee_efficiency',
             'det_duration_days', 'schedule_risk_ratio']
corr   = df[corr_cols].corr(method='spearman')
labels = ['Tasks', 'Employees', 'Dependencies',
          'Critical path', 'Efficiency', 'CPM duration', 'Reserve']

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 9}, ax=ax)
ax.set_title('Spearman Rank Correlation\n(real dataset)', pad=10)
plt.xticks(rotation=35, ha='right')
fig.savefig(OUT / 'F_spearman_correlation.png')
plt.close()
print("✓ F_spearman_correlation.png")

print(f"\nAll figures saved to: {OUT.resolve()}")
print("\n=== KEY NUMBERS ===")
print(f"N projects:              {len(df)}")
print(f"Median reserve:          {df['schedule_risk_ratio'].median():.3f}  (~{df['schedule_risk_ratio'].median()*100:.0f}%)")
print(f"Mean reserve:            {df['schedule_risk_ratio'].mean():.3f}")
print(f"P90/P50 ratio (median):  {(df['p90']/df['p50']).median():.2f}x")
print(f"Det vs P50 delta median: {df['det_vs_p50_delta'].median():.1f} days")
print(f"Efficiency median:       {df['avg_employee_efficiency'].median():.2f}")
corr_risk_tasks = df[['n_tasks', 'schedule_risk_ratio']].corr(method='spearman').iloc[0, 1]
corr_risk_eff   = df[['avg_employee_efficiency', 'schedule_risk_ratio']].corr(method='spearman').iloc[0, 1]
print(f"ρ(n_tasks, risk):        {corr_risk_tasks:.3f}")
print(f"ρ(efficiency, risk):     {corr_risk_eff:.3f}")
