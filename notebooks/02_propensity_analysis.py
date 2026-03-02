"""
CausalCare - Notebook 02: Propensity Score Analysis
=====================================================
1. Estimate propensity scores (Logistic Regression + GBM)
2. Assess positivity (overlap) via PS distribution plots
3. 1:1 nearest-neighbor matching with caliper
4. Stabilized IPW with weight trimming
5. Covariate balance: Love plot (before/after)
6. PS-based ATE estimates with bootstrap CIs
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import warnings; warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COL = {"treated": "#2196F3", "control": "#FF9800", "matched": "#4CAF50"}

df = pd.read_csv("data/processed/analysis_dataset.csv")
print(f"Dataset: {len(df)} stays | Treated: {df['early_bb'].sum()} | Deaths: {df['hospital_mortality'].sum()}")

# --- Confounder set (pre-treatment variables satisfying backdoor criterion) ---
CONFOUNDERS = [
    'age','female','african_american','hispanic','asian','other_ethnicity','night_admission',
    'apachescore','gcs_total','heartrate','meanbp','temperature','respiratoryrate',
    'lab_creatinine','lab_bun','lab_sodium','lab_potassium','lab_glucose',
    'lab_hgb','lab_wbc_x_1000','lab_platelets_x_1000','lab_bicarbonate',
    'hx_htn','hx_chf','hx_afib','hx_dm_insulin','hx_dm_noninsulin',
    'hx_copd','hx_mi_history','hx_stroke','hx_pvd','hx_renal_failure','n_comorbidities',
    'on_mech_vent','on_vasopressors','on_diuretic','on_statin',
]
available = [c for c in CONFOUNDERS if c in df.columns]

# Prepare modelling dataframe with missingness indicators + median imputation
df_m = df[['patientunitstayid','early_bb','hospital_mortality','icu_los_hours','log_icu_los']+available].copy()
for col in available:
    if df_m[col].isna().mean() > 0.05:
        df_m[f'{col}_miss'] = df_m[col].isna().astype(int)
    if df_m[col].isna().any():
        df_m[col] = df_m[col].fillna(df_m[col].median())

feat = [c for c in df_m.columns if c not in ['patientunitstayid','early_bb','hospital_mortality','icu_los_hours','log_icu_los']]
X = df_m[feat].values; T = df_m['early_bb'].values; Y = df_m['hospital_mortality'].values
Xs = StandardScaler().fit_transform(X)

# ==========================================================================
# 1. PROPENSITY SCORE ESTIMATION (cross-validated to prevent overfitting)
# ==========================================================================
ps_lr = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0, random_state=42),
                          Xs, T, cv=5, method='predict_proba')[:,1]
ps_gbm = cross_val_predict(GradientBoostingClassifier(n_estimators=200, max_depth=4,
                           learning_rate=0.05, subsample=0.8, random_state=42),
                           Xs, T, cv=5, method='predict_proba')[:,1]
df_m['ps_lr'] = ps_lr; df_m['ps_gbm'] = ps_gbm

for label, ps in [("Logistic Regression", ps_lr), ("GBM", ps_gbm)]:
    print(f"\n{label} PS — Treated: {ps[T==1].mean():.3f}±{ps[T==1].std():.3f}  "
          f"Control: {ps[T==0].mean():.3f}±{ps[T==0].std():.3f}")

# ==========================================================================
# 2. POSITIVITY: PS Overlap Plots
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ps, ttl in [(axes[0], ps_lr, 'Logistic Regression'), (axes[1], ps_gbm, 'Gradient Boosting')]:
    ax.hist(ps[T==0], bins=40, alpha=.6, color=COL['control'], label='Control', density=True, edgecolor='white')
    ax.hist(ps[T==1], bins=40, alpha=.6, color=COL['treated'], label='Treated', density=True, edgecolor='white')
    ax.set_xlabel('Propensity Score'); ax.set_ylabel('Density')
    ax.set_title(f'PS Overlap: {ttl}', fontweight='bold'); ax.legend()
    ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/02_ps_overlap.png', dpi=150, bbox_inches='tight'); plt.show()

# ==========================================================================
# 3. PS MATCHING (1:1 nearest-neighbor, caliper = 0.2*SD of logit PS)
# ==========================================================================
def ps_match(ps, treatment, caliper_sd=0.2, seed=42):
    rng = np.random.default_rng(seed)
    logit = np.log(ps/(1-ps+1e-10)); cal = caliper_sd * logit.std()
    t_idx = np.where(treatment==1)[0]; c_idx = np.where(treatment==0)[0]
    rng.shuffle(t_idx)
    mt, mc, used = [], [], set()
    for ti in t_idx:
        d = np.abs(logit[c_idx] - logit[ti])
        avail = np.array([c not in used for c in c_idx]); d[~avail] = np.inf
        best = np.argmin(d)
        if d[best] <= cal:
            ci = c_idx[best]; mt.append(ti); mc.append(ci); used.add(ci)
    return np.array(mt), np.array(mc)

mt, mc = ps_match(ps_lr, T)
print(f"\nMatched {len(mt)} pairs / {T.sum()} treated ({len(mt)/T.sum():.1%})")
matched_idx = np.concatenate([mt, mc])
df_matched = df_m.iloc[matched_idx].copy()

# ==========================================================================
# 4. IPW (Stabilized + Trimmed at 1st/99th percentile)
# ==========================================================================
def stabilized_ipw(ps, treatment, trim=(1,99)):
    ps_c = np.clip(ps, *np.percentile(ps, trim))
    p = treatment.mean()
    return np.where(treatment==1, p/ps_c, (1-p)/(1-ps_c))

ipw_w = stabilized_ipw(ps_lr, T)
df_m['ipw_w'] = ipw_w
ess_t = ipw_w[T==1].sum()**2 / (ipw_w[T==1]**2).sum()
ess_c = ipw_w[T==0].sum()**2 / (ipw_w[T==0]**2).sum()
print(f"IPW — Treated ESS: {ess_t:.0f}/{T.sum()}  Control ESS: {ess_c:.0f}/{(1-T).sum()}")

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(ipw_w[T==0], bins=50, alpha=.6, color=COL['control'], label='Control', edgecolor='white')
ax.hist(ipw_w[T==1], bins=50, alpha=.6, color=COL['treated'], label='Treated', edgecolor='white')
ax.set_xlabel('IPW Weight'); ax.set_title('Stabilized IPW Weights', fontweight='bold')
ax.legend(); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/02_ipw_weights.png', dpi=150, bbox_inches='tight'); plt.show()

# ==========================================================================
# 5. COVARIATE BALANCE: Love Plot (before / matched / IPW)
# ==========================================================================
def smd(df_sub, var, tcol='early_bb', w=None):
    t = df_sub[df_sub[tcol]==1][var].dropna(); c = df_sub[df_sub[tcol]==0][var].dropna()
    if w is not None:
        tw = w[df_sub[tcol]==1][:len(t)]; cw = w[df_sub[tcol]==0][:len(c)]
        tm = np.average(t, weights=tw); cm = np.average(c, weights=cw)
        tv = np.average((t-tm)**2, weights=tw); cv = np.average((c-cm)**2, weights=cw)
    else:
        tm, cm, tv, cv = t.mean(), c.mean(), t.var(), c.var()
    ps = np.sqrt((tv+cv)/2)
    return abs((tm-cm)/ps) if ps>0 else 0

bal_vars = [('age','Age'),('female','Female'),('apachescore','APACHE'),('meanbp','Mean BP'),
            ('heartrate','Heart Rate'),('gcs_total','GCS'),('hx_htn','HTN'),('hx_chf','CHF'),
            ('hx_afib','AFib'),('hx_mi_history','Prior MI'),('n_comorbidities','Comorbidities'),
            ('on_mech_vent','Mech Vent'),('on_vasopressors','Vasopressors'),
            ('lab_creatinine','Creatinine'),('lab_sodium','Sodium'),('lab_glucose','Glucose')]

smd_df = pd.DataFrame([{
    'Variable': lab,
    'Unadjusted': smd(df_m, var),
    'PS Matched': smd(df_matched, var),
    'IPW': smd(df_m, var, w=ipw_w),
} for var, lab in bal_vars])

fig, ax = plt.subplots(figsize=(9,8))
y = np.arange(len(smd_df))
ax.scatter(smd_df['Unadjusted'], y, color='#E53935', s=80, zorder=3, label='Unadjusted', marker='o')
ax.scatter(smd_df['PS Matched'], y, color=COL['matched'], s=80, zorder=3, label='PS Matched', marker='s')
ax.scatter(smd_df['IPW'], y, color=COL['treated'], s=80, zorder=3, label='IPW', marker='^')
for i in range(len(smd_df)):
    ax.plot([smd_df['Unadjusted'].iloc[i], min(smd_df['PS Matched'].iloc[i], smd_df['IPW'].iloc[i])],
            [y[i], y[i]], color='gray', alpha=.3, linewidth=1)
ax.axvline(x=0.1, color='gray', ls='--', lw=1.5, alpha=.7, label='SMD = 0.1')
ax.set_yticks(y); ax.set_yticklabels(smd_df['Variable'], fontsize=10)
ax.set_xlabel('|SMD|', fontsize=11)
ax.set_title('Love Plot: Covariate Balance Before & After Adjustment', fontweight='bold', fontsize=13)
ax.legend(loc='lower right'); ax.spines[['top','right']].set_visible(False); ax.invert_yaxis()
plt.tight_layout(); plt.savefig('figures/02_love_plot_comparison.png', dpi=150, bbox_inches='tight'); plt.show()

print(f"Imbalanced (SMD>0.1): Unadjusted={int((smd_df['Unadjusted']>0.1).sum())}  "
      f"Matched={int((smd_df['PS Matched']>0.1).sum())}  IPW={int((smd_df['IPW']>0.1).sum())}")

# ==========================================================================
# 6. ATE ESTIMATES (PS-based) with bootstrap CIs
# ==========================================================================
naive_ate = Y[T==1].mean() - Y[T==0].mean()

# PS Matching ATE
yt = df_matched[df_matched['early_bb']==1]['hospital_mortality'].values
yc = df_matched[df_matched['early_bb']==0]['hospital_mortality'].values
psm_ate = yt.mean() - yc.mean()
boot_psm = [np.random.default_rng(s).choice(yt,len(yt),True).mean() -
            np.random.default_rng(s+10000).choice(yc,len(yc),True).mean() for s in range(2000)]
psm_ci = np.percentile(boot_psm, [2.5, 97.5])

# IPW ATE
ipw_y1 = np.sum(T*Y*ipw_w)/np.sum(T*ipw_w); ipw_y0 = np.sum((1-T)*Y*ipw_w)/np.sum((1-T)*ipw_w)
ipw_ate = ipw_y1 - ipw_y0
n = len(T)
boot_ipw = []
for s in range(2000):
    idx = np.random.default_rng(s).choice(n, n, True)
    w,t,y = ipw_w[idx], T[idx], Y[idx]
    boot_ipw.append(np.sum(t*y*w)/np.sum(t*w) - np.sum((1-t)*y*w)/np.sum((1-t)*w))
ipw_ci = np.percentile(boot_ipw, [2.5, 97.5])

print("\n" + "="*65)
print("AVERAGE TREATMENT EFFECT (ATE): Early BB → Hospital Mortality")
print("="*65)
print(f"{'Method':<22} {'ATE':>8} {'95% CI':>22}  {'p<0.05':>6}")
print("-"*65)
print(f"{'Naive (unadjusted)':<22} {naive_ate:>+8.4f} {'--':>22}  {'--':>6}")
print(f"{'PS Matching (1:1)':<22} {psm_ate:>+8.4f}   [{psm_ci[0]:>+.4f}, {psm_ci[1]:>+.4f}]  "
      f"{'*' if (psm_ci[0]>0 or psm_ci[1]<0) else 'ns':>6}")
print(f"{'IPW (stabilized)':<22} {ipw_ate:>+8.4f}   [{ipw_ci[0]:>+.4f}, {ipw_ci[1]:>+.4f}]  "
      f"{'*' if (ipw_ci[0]>0 or ipw_ci[1]<0) else 'ns':>6}")
print("-"*65)
print("Negative ATE → early BB associated with lower mortality")

# Save for downstream notebooks
df_m['matched'] = 0; df_m.iloc[matched_idx, df_m.columns.get_loc('matched')] = 1
df_m.to_csv('data/processed/analysis_with_ps.csv', index=False)

results = pd.DataFrame([
    {'method':'Naive','ate':naive_ate,'ci_lo':np.nan,'ci_hi':np.nan},
    {'method':'PS Matching','ate':psm_ate,'ci_lo':psm_ci[0],'ci_hi':psm_ci[1]},
    {'method':'IPW','ate':ipw_ate,'ci_lo':ipw_ci[0],'ci_hi':ipw_ci[1]},
])
results.to_csv('data/processed/ate_results_ps.csv', index=False)
print("\nSaved analysis_with_ps.csv and ate_results_ps.csv")
