"""
CausalCare: Cohort Construction & Feature Engineering
======================================================
Builds the analysis-ready dataset from eICU Collaborative Research Database Demo.

Research Question: Does early beta-blocker administration (within 24h of ICU admission)
causally reduce hospital mortality in critically ill patients?

Key Design Decisions:
- Treatment: Early beta-blocker (≤24h of ICU admit) vs. no beta-blocker
- Primary Outcome: Hospital mortality
- Secondary Outcome: ICU length of stay (hours)
- Confounders: Demographics, APACHE severity, labs, comorbidities, interventions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARLY_WINDOW_MINUTES = 1440  # 24 hours

BB_PATTERN = (
    r"metoprolol|carvedilol|atenolol|propranolol|bisoprolol|"
    r"lopressor|labetalol|nadolol|sotalol|esmolol|toprol"
)

VASOPRESSOR_PATTERN = (
    r"norepinephrine|vasopressin|phenylephrine|epinephrine|"
    r"dopamine|levophed|neosynephrine|dobutamine"
)

DIURETIC_PATTERN = r"furosemide|lasix|bumetanide|torsemide"

STATIN_PATTERN = r"statin|atorvastatin|simvastatin|rosuvastatin|pravastatin|lovastatin"

# Labs to extract as baseline confounders (first value within 24h)
BASELINE_LABS = [
    "creatinine", "BUN", "sodium", "potassium", "glucose",
    "bicarbonate", "Hgb", "Hct", "WBC x 1000", "platelets x 1000",
    "calcium", "magnesium", "albumin", "bilirubin", "lactate",
]

# Comorbidity flags to extract from pastHistory
COMORBIDITY_MAP = {
    "htn": "Hypertension",
    "dm_insulin": "Insulin Dependent Diabetes",
    "dm_noninsulin": "Non-Insulin Dependent Diabetes",
    "chf": "Congestive Heart Failure",
    "afib": "atrial fibrillation",
    "copd": "COPD",
    "asthma": "Asthma",
    "mi_history": "Myocardial Infarction",
    "stroke": "Strokes",
    "pvd": "Peripheral Vascular Disease",
    "renal_failure": "Renal Failure",
    "cancer": "Cancer",
    "hypothyroid": "hypothyroidism",
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_tables(data_dir: str = "data/raw") -> dict:
    """Load all required eICU tables into a dictionary of DataFrames."""
    data_path = Path(data_dir)
    tables = {}
    
    filenames = [
        "patient", "medication", "lab", "diagnosis", "treatment",
        "apachePatientResult", "apacheApsVar", "pastHistory", "admissionDx",
    ]
    
    for fname in filenames:
        fpath = data_path / f"{fname}.csv"
        if fpath.exists():
            tables[fname] = pd.read_csv(fpath, low_memory=False)
            print(f"  Loaded {fname}: {len(tables[fname]):,} rows")
        else:
            print(f"  WARNING: {fname}.csv not found")
            
    return tables


# ---------------------------------------------------------------------------
# Treatment Assignment
# ---------------------------------------------------------------------------
def assign_treatment(
    patient: pd.DataFrame,
    medication: pd.DataFrame,
    window_minutes: int = EARLY_WINDOW_MINUTES,
) -> pd.DataFrame:
    """
    Assign binary treatment: early beta-blocker within `window_minutes` of ICU admission.
    
    CRITICAL for causal validity: Treatment must be defined based on a fixed
    time window, not conditional on patient trajectory. This avoids immortal
    time bias.
    
    Returns DataFrame with columns: patientunitstayid, early_bb, bb_start_offset
    """
    # Identify beta-blocker orders
    bb_meds = medication[
        medication["drugname"].str.contains(BB_PATTERN, case=False, na=False)
    ].copy()
    
    # Early BB: first BB order within the treatment window
    early_bb = (
        bb_meds[bb_meds["drugstartoffset"] <= window_minutes]
        .groupby("patientunitstayid")["drugstartoffset"]
        .min()
        .reset_index()
        .rename(columns={"drugstartoffset": "bb_start_offset"})
    )
    early_bb["early_bb"] = 1
    
    # Merge with all patients
    treat_df = patient[["patientunitstayid"]].merge(
        early_bb, on="patientunitstayid", how="left"
    )
    treat_df["early_bb"] = treat_df["early_bb"].fillna(0).astype(int)
    treat_df["bb_start_offset"] = treat_df["bb_start_offset"].fillna(np.nan)
    
    return treat_df


# ---------------------------------------------------------------------------
# Outcome Variables
# ---------------------------------------------------------------------------
def extract_outcomes(patient: pd.DataFrame) -> pd.DataFrame:
    """
    Extract outcome variables from the patient table.
    
    Primary: hospital_mortality (binary)
    Secondary: icu_los_hours (continuous)
    """
    outcomes = patient[["patientunitstayid"]].copy()
    
    # Hospital mortality
    outcomes["hospital_mortality"] = (
        patient["hospitaldischargestatus"] == "Expired"
    ).astype(int)
    
    # ICU mortality
    outcomes["icu_mortality"] = (
        patient["unitdischargestatus"] == "Expired"
    ).astype(int)
    
    # ICU LOS in hours
    outcomes["icu_los_hours"] = patient["unitdischargeoffset"] / 60.0
    
    # Log-transformed LOS (for regression)
    outcomes["log_icu_los"] = np.log1p(outcomes["icu_los_hours"].clip(lower=0))
    
    return outcomes


# ---------------------------------------------------------------------------
# Confounder Extraction: Demographics
# ---------------------------------------------------------------------------
def extract_demographics(patient: pd.DataFrame) -> pd.DataFrame:
    """Extract demographic confounders."""
    demo = patient[["patientunitstayid"]].copy()
    
    # Age (handle "> 89" as 90 for numeric processing)
    demo["age"] = patient["age"].replace("> 89", "90").astype(float)
    
    # Sex
    demo["female"] = (patient["gender"] == "Female").astype(int)
    
    # Ethnicity (one-hot, with Caucasian as reference)
    demo["african_american"] = (patient["ethnicity"] == "African American").astype(int)
    demo["hispanic"] = (patient["ethnicity"] == "Hispanic").astype(int)
    demo["asian"] = (patient["ethnicity"] == "Asian").astype(int)
    demo["other_ethnicity"] = patient["ethnicity"].isin(
        ["Other/Unknown", "Native American"]
    ).astype(int)
    
    # Admission context
    demo["night_admission"] = patient["unitadmittime24"].apply(
        lambda x: 1 if isinstance(x, str) and (int(x.split(":")[0]) >= 19 or int(x.split(":")[0]) < 7) else 0
    )
    
    # Unit type
    demo["unit_type"] = patient["unittype"].fillna("Unknown")
    
    return demo


# ---------------------------------------------------------------------------
# Confounder Extraction: Severity Scores
# ---------------------------------------------------------------------------
def extract_severity(
    apacheResult: pd.DataFrame,
    apacheAps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract severity-of-illness confounders from APACHE tables.
    
    These are measured at admission (before treatment), satisfying the
    temporal ordering requirement for valid confounders.
    """
    # APACHE score and predicted mortality
    severity = apacheResult[
        ["patientunitstayid", "apachescore", "predictedhospitalmortality"]
    ].copy()
    
    # Take the first APACHE record per stay
    severity = severity.sort_values("apachescore").groupby(
        "patientunitstayid"
    ).first().reset_index()
    
    # APACHE APS physiological variables (admission values)
    aps_cols = [
        "patientunitstayid", "intubated", "vent", "dialysis",
        "heartrate", "meanbp", "temperature", "respiratoryrate",
        "creatinine", "sodium", "hematocrit", "wbc",
        "bun", "glucose", "bilirubin", "albumin", "pao2", "pco2", "fio2",
        "eyes", "motor", "verbal",
    ]
    aps_available = [c for c in aps_cols if c in apacheAps.columns]
    aps = apacheAps[aps_available].copy()
    
    # GCS total from components
    if all(c in aps.columns for c in ["eyes", "motor", "verbal"]):
        aps["gcs_total"] = aps[["eyes", "motor", "verbal"]].sum(axis=1)
        aps.loc[aps["gcs_total"] == 0, "gcs_total"] = np.nan
    
    # Take first record per stay
    aps = aps.groupby("patientunitstayid").first().reset_index()
    
    # Merge
    severity = severity.merge(aps, on="patientunitstayid", how="outer")
    
    return severity


# ---------------------------------------------------------------------------
# Confounder Extraction: Baseline Labs
# ---------------------------------------------------------------------------
def extract_baseline_labs(
    lab: pd.DataFrame,
    window_minutes: int = EARLY_WINDOW_MINUTES,
) -> pd.DataFrame:
    """
    Extract first lab value within 24h of ICU admission for each patient.
    
    Only uses labs measured BEFORE or concurrent with treatment window
    to maintain temporal ordering.
    """
    # Filter to early labs
    early_labs = lab[
        (lab["labresultoffset"] >= -360) &  # up to 6h before ICU admit
        (lab["labresultoffset"] <= window_minutes)
    ].copy()
    
    # Convert to numeric
    early_labs["labresult"] = pd.to_numeric(early_labs["labresult"], errors="coerce")
    
    results = []
    for lab_name in BASELINE_LABS:
        lab_subset = early_labs[
            early_labs["labname"].str.contains(lab_name, case=False, na=False)
        ]
        # Take first (earliest) value per patient
        first_val = (
            lab_subset.sort_values("labresultoffset")
            .groupby("patientunitstayid")["labresult"]
            .first()
            .reset_index()
            .rename(columns={"labresult": f"lab_{lab_name.lower().replace(' ', '_')}"})
        )
        results.append(first_val)
    
    # Merge all labs
    if not results:
        return pd.DataFrame(columns=["patientunitstayid"])
    
    lab_df = results[0]
    for r in results[1:]:
        lab_df = lab_df.merge(r, on="patientunitstayid", how="outer")
    
    return lab_df


# ---------------------------------------------------------------------------
# Confounder Extraction: Comorbidities
# ---------------------------------------------------------------------------
def extract_comorbidities(pastHistory: pd.DataFrame) -> pd.DataFrame:
    """
    Extract binary comorbidity flags from the pastHistory table.
    
    These represent pre-existing conditions (before ICU admission),
    making them valid confounders.
    """
    comorbidities = (
        pastHistory.groupby("patientunitstayid")["pasthistorypath"]
        .apply(lambda x: " | ".join(x))
        .reset_index()
        .rename(columns={"pasthistorypath": "history_text"})
    )
    
    for flag_name, pattern in COMORBIDITY_MAP.items():
        comorbidities[f"hx_{flag_name}"] = (
            comorbidities["history_text"]
            .str.contains(pattern, case=False, na=False)
            .astype(int)
        )
    
    # Comorbidity count
    hx_cols = [c for c in comorbidities.columns if c.startswith("hx_")]
    comorbidities["n_comorbidities"] = comorbidities[hx_cols].sum(axis=1)
    
    # Drop the raw text
    comorbidities = comorbidities.drop(columns=["history_text"])
    
    return comorbidities


# ---------------------------------------------------------------------------
# Confounder Extraction: Concurrent Treatments
# ---------------------------------------------------------------------------
def extract_interventions(
    medication: pd.DataFrame,
    treatment: pd.DataFrame,
    window_minutes: int = EARLY_WINDOW_MINUTES,
) -> pd.DataFrame:
    """
    Extract concurrent interventions that may confound the treatment-outcome
    relationship (i.e., sicker patients receive more interventions).
    """
    stays = set(medication["patientunitstayid"].unique()) | set(
        treatment["patientunitstayid"].unique()
    )
    interv = pd.DataFrame({"patientunitstayid": list(stays)})
    
    # Mechanical ventilation
    mech_vent = treatment[
        treatment["treatmentstring"].str.contains(
            "mechanical ventilation", case=False, na=False
        )
    ]
    interv["on_mech_vent"] = interv["patientunitstayid"].isin(
        mech_vent["patientunitstayid"].unique()
    ).astype(int)
    
    # Vasopressors (within 24h)
    early_vaso = medication[
        (medication["drugname"].str.contains(VASOPRESSOR_PATTERN, case=False, na=False))
        & (medication["drugstartoffset"] <= window_minutes)
    ]
    interv["on_vasopressors"] = interv["patientunitstayid"].isin(
        early_vaso["patientunitstayid"].unique()
    ).astype(int)
    
    # Early diuretics
    early_diuretic = medication[
        (medication["drugname"].str.contains(DIURETIC_PATTERN, case=False, na=False))
        & (medication["drugstartoffset"] <= window_minutes)
    ]
    interv["on_diuretic"] = interv["patientunitstayid"].isin(
        early_diuretic["patientunitstayid"].unique()
    ).astype(int)
    
    # Early statins
    early_statin = medication[
        (medication["drugname"].str.contains(STATIN_PATTERN, case=False, na=False))
        & (medication["drugstartoffset"] <= window_minutes)
    ]
    interv["on_statin"] = interv["patientunitstayid"].isin(
        early_statin["patientunitstayid"].unique()
    ).astype(int)
    
    # Total early medication count (proxy for illness complexity)
    early_meds = medication[medication["drugstartoffset"] <= window_minutes]
    med_counts = (
        early_meds.groupby("patientunitstayid")["drugname"]
        .nunique()
        .reset_index()
        .rename(columns={"drugname": "n_early_medications"})
    )
    interv = interv.merge(med_counts, on="patientunitstayid", how="left")
    interv["n_early_medications"] = interv["n_early_medications"].fillna(0)
    
    return interv


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------
def build_analysis_dataset(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Build the complete analysis-ready dataset by joining all components.
    
    Returns a single DataFrame with:
    - Patient identifier
    - Treatment assignment
    - Outcome variables
    - All measured confounders
    """
    print("Loading tables...")
    tables = load_tables(data_dir)
    
    print("\nAssigning treatment...")
    treat = assign_treatment(tables["patient"], tables["medication"])
    print(f"  Treated (early BB): {treat['early_bb'].sum()}")
    print(f"  Control (no early BB): {(treat['early_bb']==0).sum()}")
    
    print("\nExtracting outcomes...")
    outcomes = extract_outcomes(tables["patient"])
    
    print("\nExtracting demographics...")
    demo = extract_demographics(tables["patient"])
    
    print("\nExtracting severity scores...")
    severity = extract_severity(tables["apachePatientResult"], tables["apacheApsVar"])
    
    print("\nExtracting baseline labs...")
    labs = extract_baseline_labs(tables["lab"])
    
    print("\nExtracting comorbidities...")
    comorbidities = extract_comorbidities(tables["pastHistory"])
    
    print("\nExtracting concurrent interventions...")
    interventions = extract_interventions(tables["medication"], tables["treatment"])
    
    # Join everything
    print("\nJoining all features...")
    df = treat.merge(outcomes, on="patientunitstayid", how="left")
    df = df.merge(demo, on="patientunitstayid", how="left")
    df = df.merge(severity, on="patientunitstayid", how="left")
    df = df.merge(labs, on="patientunitstayid", how="left")
    df = df.merge(comorbidities, on="patientunitstayid", how="left")
    df = df.merge(interventions, on="patientunitstayid", how="left")
    
    # Fill comorbidity NAs with 0 (absence = no history recorded)
    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    df[hx_cols] = df[hx_cols].fillna(0)
    df["n_comorbidities"] = df["n_comorbidities"].fillna(0)
    
    # Drop unit_type (categorical, will encode separately if needed)
    # For now keep it for EDA
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS DATASET BUILT")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Treatment (early_bb=1): {df['early_bb'].sum()} ({df['early_bb'].mean():.1%})")
    print(f"  Outcome (mortality=1): {df['hospital_mortality'].sum()} ({df['hospital_mortality'].mean():.1%})")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False).head(15)}")
    
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_analysis_dataset()
    
    # Save to processed
    out_path = Path("data/processed")
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "analysis_dataset.csv", index=False)
    print(f"\nSaved to {out_path / 'analysis_dataset.csv'}")
