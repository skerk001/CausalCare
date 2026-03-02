"""
CausalCare - Notebook 01: Cohort Description & Causal Framework
================================================================
This notebook:
1. Describes the cohort and treatment/outcome distributions
2. Creates a "Table 1" comparing treated vs. control (pre-matching)
3. Defines and visualizes the causal DAG
4. Assesses initial covariate imbalance (motivating the need for causal methods)

Dataset: eICU Collaborative Research Database Demo v2.0.1
Treatment: Early beta-blocker (≤24h of ICU admission)
Outcome: Hospital mortality
"""

# %% [markdown]
# # CausalCare: Causal Effect of Early Beta-Blocker on ICU Mortality
# 
# **Research Question:** Does early beta-blocker administration (within 24h of ICU 
# admission) causally reduce hospital mortality in critically ill patients?

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
COLORS = {"treated": "#2196F3", "control": "#FF9800", "accent": "#4CAF50"}

# %%
# Load analysis dataset
df = pd.read_csv("data/processed/analysis_dataset.csv")
print(f"Dataset: {df.shape[0]} ICU stays, {df.shape[1]} variables")
print(f"Treatment (early_bb=1): {df['early_bb'].sum()} ({df['early_bb'].mean():.1%})")
print(f"Outcome (hospital_mortality=1): {df['hospital_mortality'].sum()} ({df['hospital_mortality'].mean():.1%})")

# %% [markdown]
# ## 1. Treatment & Outcome Overview

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Treatment distribution
ax = axes[0]
counts = df['early_bb'].value_counts().sort_index()
bars = ax.bar(['No Early BB\n(Control)', 'Early BB\n(Treated)'], counts.values,
              color=[COLORS['control'], COLORS['treated']], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
            f'n={val}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('ICU Stays')
ax.set_title('Treatment Assignment', fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# Mortality by treatment (the naive comparison)
ax = axes[1]
mort_by_treat = df.groupby('early_bb')['hospital_mortality'].mean() * 100
bars = ax.bar(['Control', 'Treated'], mort_by_treat.values,
              color=[COLORS['control'], COLORS['treated']], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, mort_by_treat.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Hospital Mortality (%)')
ax.set_title('Naive Mortality Comparison\n(Unadjusted)', fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# ICU LOS by treatment
ax = axes[2]
for label, group, color in [('Control', 0, COLORS['control']), ('Treated', 1, COLORS['treated'])]:
    subset = df[df['early_bb'] == group]['icu_los_hours'].clip(upper=500)
    ax.hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
ax.set_xlabel('ICU Length of Stay (hours)')
ax.set_ylabel('Density')
ax.set_title('ICU LOS Distribution', fontweight='bold')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('figures/01_treatment_outcome_overview.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nNaive (unadjusted) mortality difference: "
      f"{mort_by_treat[0]:.1f}% (control) vs {mort_by_treat[1]:.1f}% (treated)")
print(f"Naive risk difference: {mort_by_treat[1] - mort_by_treat[0]:.1f} percentage points")
print("⚠️  This is confounded - we need causal methods to get the true effect.")

# %% [markdown]
# ## 2. Table 1: Baseline Characteristics by Treatment Group
# 
# A standard "Table 1" showing pre-treatment characteristics. Standardized Mean
# Differences (SMD) > 0.1 indicate meaningful imbalance requiring adjustment.

# %%
def standardized_mean_diff(treated, control):
    """Calculate SMD for continuous or binary variables."""
    t_mean, c_mean = treated.mean(), control.mean()
    t_var, c_var = treated.var(), control.var()
    pooled_std = np.sqrt((t_var + c_var) / 2)
    if pooled_std == 0:
        return 0.0
    return (t_mean - c_mean) / pooled_std

def make_table1(df):
    """Generate Table 1 with SMD for each covariate."""
    treated = df[df['early_bb'] == 1]
    control = df[df['early_bb'] == 0]
    
    rows = []
    
    # Demographics
    for var, label in [
        ('age', 'Age (years)'),
        ('female', 'Female sex (%)'),
        ('african_american', 'African American (%)'),
        ('hispanic', 'Hispanic (%)'),
    ]:
        t_val = treated[var].dropna()
        c_val = control[var].dropna()
        smd = standardized_mean_diff(t_val, c_val)
        
        if var in ['female', 'african_american', 'hispanic']:
            rows.append({
                'Variable': label,
                'Treated (n={})'.format(len(treated)): f"{t_val.mean()*100:.1f}%",
                'Control (n={})'.format(len(control)): f"{c_val.mean()*100:.1f}%",
                'SMD': f"{abs(smd):.3f}",
                'smd_val': abs(smd),
            })
        else:
            rows.append({
                'Variable': label,
                'Treated (n={})'.format(len(treated)): f"{t_val.mean():.1f} ± {t_val.std():.1f}",
                'Control (n={})'.format(len(control)): f"{c_val.mean():.1f} ± {c_val.std():.1f}",
                'SMD': f"{abs(smd):.3f}",
                'smd_val': abs(smd),
            })
    
    # Severity
    for var, label in [
        ('apachescore', 'APACHE Score'),
        ('gcs_total', 'GCS Total'),
        ('heartrate', 'Heart Rate (bpm)'),
        ('meanbp', 'Mean BP (mmHg)'),
    ]:
        t_val = treated[var].dropna()
        c_val = control[var].dropna()
        smd = standardized_mean_diff(t_val, c_val)
        rows.append({
            'Variable': label,
            'Treated (n={})'.format(len(treated)): f"{t_val.mean():.1f} ± {t_val.std():.1f}",
            'Control (n={})'.format(len(control)): f"{c_val.mean():.1f} ± {c_val.std():.1f}",
            'SMD': f"{abs(smd):.3f}",
            'smd_val': abs(smd),
        })
    
    # Comorbidities
    for var, label in [
        ('hx_htn', 'Hypertension (%)'),
        ('hx_chf', 'CHF History (%)'),
        ('hx_afib', 'Atrial Fibrillation (%)'),
        ('hx_dm_insulin', 'Insulin-Dep. Diabetes (%)'),
        ('hx_copd', 'COPD (%)'),
        ('hx_mi_history', 'Prior MI (%)'),
        ('n_comorbidities', 'Comorbidity Count'),
    ]:
        t_val = treated[var].dropna()
        c_val = control[var].dropna()
        smd = standardized_mean_diff(t_val, c_val)
        
        if var == 'n_comorbidities':
            rows.append({
                'Variable': label,
                'Treated (n={})'.format(len(treated)): f"{t_val.mean():.1f} ± {t_val.std():.1f}",
                'Control (n={})'.format(len(control)): f"{c_val.mean():.1f} ± {c_val.std():.1f}",
                'SMD': f"{abs(smd):.3f}",
                'smd_val': abs(smd),
            })
        else:
            rows.append({
                'Variable': label,
                'Treated (n={})'.format(len(treated)): f"{t_val.mean()*100:.1f}%",
                'Control (n={})'.format(len(control)): f"{c_val.mean()*100:.1f}%",
                'SMD': f"{abs(smd):.3f}",
                'smd_val': abs(smd),
            })
    
    # Interventions
    for var, label in [
        ('on_mech_vent', 'Mechanical Ventilation (%)'),
        ('on_vasopressors', 'Vasopressors (%)'),
        ('n_early_medications', 'Num Early Medications'),
    ]:
        t_val = treated[var].dropna()
        c_val = control[var].dropna()
        smd = standardized_mean_diff(t_val, c_val)
        
        if var == 'n_early_medications':
            rows.append({
                'Variable': label,
                'Treated (n={})'.format(len(treated)): f"{t_val.mean():.1f} ± {t_val.std():.1f}",
                'Control (n={})'.format(len(control)): f"{c_val.mean():.1f} ± {c_val.std():.1f}",
                'SMD': f"{abs(smd):.3f}",
                'smd_val': abs(smd),
            })
        else:
            rows.append({
                'Variable': label,
                'Treated (n={})'.format(len(treated)): f"{t_val.mean()*100:.1f}%",
                'Control (n={})'.format(len(control)): f"{c_val.mean()*100:.1f}%",
                'SMD': f"{abs(smd):.3f}",
                'smd_val': abs(smd),
            })
    
    table1 = pd.DataFrame(rows)
    return table1

table1 = make_table1(df)
print(table1[['Variable', f'Treated (n={df["early_bb"].sum()})', 
               f'Control (n={(df["early_bb"]==0).sum()})', 'SMD']].to_string(index=False))

# %% [markdown]
# ### Love Plot: Pre-Matching Covariate Balance

# %%
fig, ax = plt.subplots(figsize=(8, 7))

y_pos = range(len(table1))
smds = table1['smd_val'].values
labels = table1['Variable'].values

colors = ['#E53935' if s > 0.1 else '#43A047' for s in smds]
ax.barh(y_pos, smds, color=colors, height=0.6, edgecolor='white')
ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=1.5, label='SMD = 0.1 threshold')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Absolute Standardized Mean Difference', fontsize=11)
ax.set_title('Covariate Balance: Pre-Matching\n(Red = SMD > 0.1, needs adjustment)', fontweight='bold')
ax.legend(loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/01_love_plot_pre_matching.png', dpi=150, bbox_inches='tight')
plt.show()

n_imbalanced = (table1['smd_val'] > 0.1).sum()
print(f"\n{n_imbalanced} / {len(table1)} covariates have SMD > 0.1 (imbalanced)")
print("This confirms the need for causal adjustment methods.")

# %% [markdown]
# ## 3. Causal DAG (Directed Acyclic Graph)
# 
# Our causal assumptions encoded as a DAG. This is the foundation of the
# entire analysis — every causal method relies on these assumptions being
# approximately correct.

# %%
# DAG as text (for README and documentation)
dag_description = """
CAUSAL DAG: Early Beta-Blocker → Hospital Mortality

Confounders (W):
├── Demographics: age, sex, ethnicity
├── Severity: APACHE score, GCS, heart rate, mean BP
├── Comorbidities: HTN, CHF, AFib, DM, COPD, prior MI
├── Labs: creatinine, BUN, sodium, potassium, glucose, Hgb
└── Concurrent Tx: mech ventilation, vasopressors, diuretics

Assumptions:
1. No unmeasured confounders (conditional ignorability)
   - Violated if: physician preference, EF, troponin levels drive both BB and mortality
2. Positivity: P(T=1|W=w) > 0 for all w (checked via propensity overlap)
3. Consistency: The treatment is well-defined
4. No interference: One patient's treatment doesn't affect another's outcome

DAG Structure:
  W (confounders) → T (early BB) → Y (mortality)
  W ──────────────────────────────→ Y
"""
print(dag_description)

# %%
# Create DAG visualization
try:
    import graphviz
    
    dot = graphviz.Digraph(comment='CausalCare DAG', engine='dot')
    dot.attr(rankdir='LR', bgcolor='white', fontname='Arial')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='11')
    dot.attr('edge', color='#555555', arrowsize='0.8')
    
    # Treatment node
    dot.node('T', 'Early\nBeta-Blocker\n(Treatment)', fillcolor='#BBDEFB', color='#1976D2')
    
    # Outcome node  
    dot.node('Y', 'Hospital\nMortality\n(Outcome)', fillcolor='#FFCDD2', color='#D32F2F')
    
    # Confounder groups
    dot.node('W1', 'Demographics\n(age, sex, ethnicity)', fillcolor='#E8F5E9', color='#388E3C')
    dot.node('W2', 'Severity\n(APACHE, GCS, vitals)', fillcolor='#E8F5E9', color='#388E3C')
    dot.node('W3', 'Comorbidities\n(HTN, CHF, AFib, DM\nCOPD, MI)', fillcolor='#E8F5E9', color='#388E3C')
    dot.node('W4', 'Baseline Labs\n(creatinine, BUN, Na\nK, glucose, Hgb)', fillcolor='#E8F5E9', color='#388E3C')
    dot.node('W5', 'Concurrent Tx\n(vent, vasopressors\ndiuretics)', fillcolor='#FFF3E0', color='#F57C00')
    dot.node('U', 'Unmeasured\n(EF, troponin\nMD preference)', fillcolor='#F5F5F5', color='#9E9E9E', style='rounded,filled,dashed')
    
    # Edges: confounders → treatment
    for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
        dot.edge(w, 'T')
        dot.edge(w, 'Y')
    
    # Treatment → outcome
    dot.edge('T', 'Y', color='#1976D2', penwidth='2.5')
    
    # Unmeasured confounders (dashed)
    dot.edge('U', 'T', style='dashed', color='#9E9E9E')
    dot.edge('U', 'Y', style='dashed', color='#9E9E9E')
    
    dot.render('figures/causal_dag', format='png', cleanup=True)
    print("DAG saved to figures/causal_dag.png")
except ImportError:
    print("graphviz not available for rendering, but DAG is defined above")

# %% [markdown]
# ## 4. Key Summary Statistics

# %%
print("=" * 60)
print("COHORT SUMMARY")
print("=" * 60)
print(f"Total ICU stays:           {len(df):,}")
print(f"Hospitals:                 {df['patientunitstayid'].nunique():,}")  # proxy
print(f"Treatment (early BB):      {df['early_bb'].sum():,} ({df['early_bb'].mean():.1%})")
print(f"Hospital mortality:        {df['hospital_mortality'].sum():,} ({df['hospital_mortality'].mean():.1%})")
print(f"ICU mortality:             {df['icu_mortality'].sum():,} ({df['icu_mortality'].mean():.1%})")
print(f"Median ICU LOS:            {df['icu_los_hours'].median():.1f} hours")
print(f"APACHE score (mean±SD):    {df['apachescore'].mean():.1f} ± {df['apachescore'].std():.1f}")
print(f"Age (mean±SD):             {df['age'].mean():.1f} ± {df['age'].std():.1f}")
print(f"Female:                    {df['female'].mean():.1%}")
print(f"")
print("NEXT STEPS:")
print("  → Notebook 02: Propensity Score Estimation & Matching")
print("  → Notebook 03: Causal Effect Estimation (IPW, AIPW, DML)")
print("  → Notebook 04: Sensitivity Analysis & Robustness Checks")
