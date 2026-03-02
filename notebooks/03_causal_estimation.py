"""
CausalCare - Notebook 03: Causal Effect Estimation
====================================================
1. AIPW (Doubly Robust) - manual cross-fit implementation
2. DoWhy end-to-end pipeline
3. Double Machine Learning (EconML LinearDML)
4. Causal Forest heterogeneous effects (CATE)
5. Forest plot comparing all methods
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
COL = {"treated":"#2196F3","control":"#FF9800","matched":"#4CAF50","dr":"#9C27B0","dml":"#E91E63"}

df = pd.read_csv("data/processed/analysis_with_ps.csv")
print(f"Loaded: {len(df)} | Treated: {df['early_bb'].sum()} | Deaths: {df['hospital_mortality'].sum()}")

CONF = ['age','female','african_american','hispanic','asian','other_ethnicity','night_admission',
        'apachescore','gcs_total','heartrate','meanbp','temperature','respiratoryrate',
        'lab_creatinine','lab_bun','lab_sodium','lab_potassium','lab_glucose',
        'lab_hgb','lab_wbc_x_1000','lab_platelets_x_1000','lab_bicarbonate',
        'hx_htn','hx_chf','hx_afib','hx_dm_insulin','hx_dm_noninsulin',
        'hx_copd','hx_mi_history','hx_stroke','hx_pvd','hx_renal_failure','n_comorbidities',
        'on_mech_vent','on_vasopressors','on_diuretic','on_statin']
avail = [c for c in CONF if c in df.columns]
miss_cols = [c for c in df.columns if c.endswith('_miss')]
feat = avail + miss_cols
for c in feat:
    if df[c].isna().any(): df[c] = df[c].fillna(df[c].median())
X = df[feat].values; T = df['early_bb'].values; Y = df['hospital_mortality'].values
Xs = StandardScaler().fit_transform(X)

# 1. AIPW ===================================================================
print("\n" + "="*65 + "\n1. AUGMENTED IPW (Doubly Robust)\n" + "="*65)
def aipw(X, T, Y, ns=5, seed=42):
    kf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
    mu1, mu0, ps = np.zeros(len(Y)), np.zeros(len(Y)), np.zeros(len(Y))
    for tr, te in kf.split(X, T):
        ps_m = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        ps_m.fit(X[tr], T[tr]); ps[te] = ps_m.predict_proba(X[te])[:,1]
        tm, cm = T[tr]==1, T[tr]==0
        o1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        o0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        if tm.sum()>10: o1.fit(X[tr][tm], Y[tr][tm]); mu1[te] = o1.predict(X[te])
        if cm.sum()>10: o0.fit(X[tr][cm], Y[tr][cm]); mu0[te] = o0.predict(X[te])
    ps = np.clip(ps, 0.01, 0.99)
    a1 = mu1 + T*(Y-mu1)/ps; a0 = mu0 + (1-T)*(Y-mu0)/(1-ps)
    tau = a1 - a0; return tau.mean(), tau.std()/np.sqrt(len(tau)), tau

aipw_ate, aipw_se, aipw_tau = aipw(Xs, T, Y)
aipw_ci = (aipw_ate - 1.96*aipw_se, aipw_ate + 1.96*aipw_se)
print(f"AIPW ATE: {aipw_ate:+.4f}  95% CI: [{aipw_ci[0]:+.4f}, {aipw_ci[1]:+.4f}]")

# 2. DoWhy ==================================================================
print("\n" + "="*65 + "\n2. DoWhy PIPELINE\n" + "="*65)
dowhy_ate = np.nan
try:
    from dowhy import CausalModel
    df_dw = df[['early_bb','hospital_mortality']+avail].copy()
    for c in df_dw.columns:
        if df_dw[c].isna().any(): df_dw[c] = df_dw[c].fillna(df_dw[c].median())
    m = CausalModel(data=df_dw, treatment='early_bb', outcome='hospital_mortality', common_causes=avail)
    est = m.identify_effect(proceed_when_unidentifiable=True)
    e_ipw = m.estimate_effect(est, method_name="backdoor.propensity_score_weighting",
                               method_params={"weighting_scheme":"ips_stabilized_weight"})
    e_lr = m.estimate_effect(est, method_name="backdoor.linear_regression")
    print(f"DoWhy IPW: {e_ipw.value:+.4f}")
    print(f"DoWhy LinReg: {e_lr.value:+.4f}")
    dowhy_ate = e_ipw.value
except Exception as e:
    print(f"DoWhy error: {e}")

# 3. Double ML ===============================================================
print("\n" + "="*65 + "\n3. DOUBLE MACHINE LEARNING (EconML)\n" + "="*65)
dml_ate, dml_ci = np.nan, (np.nan, np.nan)
try:
    from econml.dml import LinearDML
    # Note: EconML requires regressors for both model_y and model_t,
    # even when T is binary. It residualizes T as continuous.
    dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        cv=5, random_state=42)
    dml.fit(Y, T, X=Xs)
    dml_ate = float(dml.ate(X=Xs)); dml_ci = tuple(float(x) for x in dml.ate_interval(X=Xs, alpha=0.05))
    print(f"LinearDML ATE: {dml_ate:+.4f}  95% CI: [{dml_ci[0]:+.4f}, {dml_ci[1]:+.4f}]")
except Exception as e:
    print(f"EconML error: {e}")

# 4. Causal Forest (CATE) ===================================================
print("\n" + "="*65 + "\n4. CAUSAL FOREST (Heterogeneous Effects)\n" + "="*65)
cf_ate, cf_ci, cate = np.nan, (np.nan, np.nan), None
try:
    from econml.dml import CausalForestDML
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        n_estimators=200, cv=5, random_state=42)
    cf.fit(Y, T, X=Xs)
    cate = cf.effect(Xs).flatten()
    cf_ate = float(cate.mean())
    try:
        cf_ci = tuple(float(x) for x in cf.ate_interval(X=Xs, alpha=0.05))
    except:
        # Fallback: bootstrap CI from CATE distribution
        boot_cf = [np.random.default_rng(s).choice(cate, len(cate), True).mean() for s in range(2000)]
        cf_ci = tuple(np.percentile(boot_cf, [2.5, 97.5]))
    print(f"Causal Forest ATE: {cf_ate:+.4f}  95% CI: [{cf_ci[0]:+.4f}, {cf_ci[1]:+.4f}]")
    print(f"CATE: mean={cate.mean():+.4f}, std={cate.std():.4f}, range=[{cate.min():+.4f}, {cate.max():+.4f}]")

    # Subgroup analysis
    for name, mask in [("Age≥65", df['age'].values>=65), ("CHF hx", df['hx_chf'].values==1),
                       ("HTN", df['hx_htn'].values==1), ("Vasopressors", df['on_vasopressors'].values==1)]:
        print(f"  {name:15s}: CATE={cate[mask].mean():+.4f} (n={mask.sum()})")

    # CATE plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(cate, bins=50, color=COL['dr'], alpha=.7, edgecolor='white')
    ax.axvline(0, color='black', ls='--', lw=1.5)
    ax.axvline(cate.mean(), color='red', lw=2, label=f'Mean={cate.mean():+.4f}')
    ax.set_xlabel('CATE'); ax.set_title('Individual Treatment Effect Distribution', fontweight='bold')
    ax.legend(); ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    apache = df['apachescore'].values; valid = ~np.isnan(apache)
    if valid.sum() > 100:
        q = pd.qcut(apache[valid], 5, labels=['Q1\nLeast\nSick','Q2','Q3','Q4','Q5\nMost\nSick'], duplicates='drop')
        cdf = pd.DataFrame({'q': q, 'cate': cate[valid]}).groupby('q')['cate']
        ax.bar(range(5), cdf.mean(), yerr=1.96*cdf.std()/np.sqrt(cdf.count()),
               capsize=5, color=COL['treated'], alpha=.7, edgecolor='white')
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(cdf.mean().index)
        ax.set_ylabel('Mean CATE')
        ax.set_title('Treatment Effect by APACHE Quintile', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.savefig('figures/03_cate_analysis.png', dpi=150, bbox_inches='tight'); plt.show()
except Exception as e:
    print(f"Causal Forest error: {e}")

# 5. FOREST PLOT =============================================================
print("\n" + "="*65 + "\n5. FOREST PLOT: ALL METHODS\n" + "="*65)
ps_res = pd.read_csv('data/processed/ate_results_ps.csv')
all_res = []
for _, r in ps_res.iterrows():
    all_res.append({'method':r['method'],'ate':r['ate'],'ci_lo':r.get('ci_lo',np.nan),'ci_hi':r.get('ci_hi',np.nan)})
all_res.append({'method':'AIPW (Doubly Robust)','ate':aipw_ate,'ci_lo':aipw_ci[0],'ci_hi':aipw_ci[1]})
if not np.isnan(dml_ate):
    all_res.append({'method':'Double ML','ate':dml_ate,'ci_lo':dml_ci[0],'ci_hi':dml_ci[1]})
if not np.isnan(cf_ate):
    all_res.append({'method':'Causal Forest','ate':cf_ate,'ci_lo':cf_ci[0],'ci_hi':cf_ci[1]})
res = pd.DataFrame(all_res)

print(f"\n{'Method':<25} {'ATE':>8} {'95% CI':>22}")
print("-"*60)
for _, r in res.iterrows():
    ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]" if not np.isnan(r['ci_lo']) else '--'
    print(f"{r['method']:<25} {r['ate']:>+8.4f}   {ci}")

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(res))
colors = ['#9E9E9E',COL['matched'],COL['treated'],COL['dr'],COL['dml'],'#FF5722'][:len(res)]
for i, (_, r) in enumerate(res.iterrows()):
    lo = r['ci_lo'] if not np.isnan(r['ci_lo']) else r['ate']
    hi = r['ci_hi'] if not np.isnan(r['ci_hi']) else r['ate']
    ax.plot([lo,hi],[y[i]]*2, color=colors[i], lw=2.5, solid_capstyle='round')
    ax.scatter(r['ate'], y[i], color=colors[i], s=120, zorder=5, edgecolors='white', lw=1.5)
ax.axvline(0, color='black', ls='--', lw=1.5, alpha=.5)
ax.set_yticks(y); ax.set_yticklabels(res['method'], fontsize=11)
ax.set_xlabel('ATE (Risk Difference)', fontsize=12)
ax.set_title('Forest Plot: Effect of Early Beta-Blocker on Hospital Mortality', fontweight='bold', fontsize=13)
ax.spines[['top','right']].set_visible(False); ax.invert_yaxis()
ax.annotate('← Favors BB', xy=(ax.get_xlim()[0]*.7, len(res)-.3), fontsize=9, color=COL['treated'])
ax.annotate('Favors Control →', xy=(max(0.01, ax.get_xlim()[1]*.4), len(res)-.3), fontsize=9, color=COL['control'])
plt.tight_layout(); plt.savefig('figures/03_forest_plot.png', dpi=150, bbox_inches='tight'); plt.show()

res.to_csv('data/processed/ate_results_all.csv', index=False)
if cate is not None:
    pd.DataFrame({'patientunitstayid':df['patientunitstayid'],'cate':cate}).to_csv('data/processed/cate_scores.csv', index=False)
print("All results saved.")
