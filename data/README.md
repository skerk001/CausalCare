# Data Directory

## eICU Collaborative Research Database Demo v2.0.1

This project uses the **eICU-CRD Demo** — an openly available subset of the eICU Collaborative Research Database. No credentialing or data use agreement is required.

### Download Instructions

1. Visit: https://physionet.org/content/eicu-crd-demo/2.0.1/
2. Download the CSV files
3. Place them in `data/raw/`

### Required Tables

| File | Description |
|------|-------------|
| `patient.csv` | Demographics, admission/discharge info |
| `medication.csv` | Medication orders and timing |
| `lab.csv` | Laboratory results |
| `diagnosis.csv` | ICD diagnoses |
| `treatment.csv` | Interventions (ventilation, etc.) |
| `apachePatientResult.csv` | APACHE severity scores |
| `apacheApsVar.csv` | APACHE physiological variables |
| `pastHistory.csv` | Comorbidity history |
| `admissionDx.csv` | Admission diagnoses |

### Building the Analysis Dataset

After placing raw CSVs in `data/raw/`:

```bash
cd CausalCare
python src/cohort.py
```

This generates `data/processed/analysis_dataset.csv`, which is used by all notebooks.

### Data Files (Not Tracked)

All CSV files are excluded from version control via `.gitignore`. The analysis pipeline is fully reproducible from the raw eICU demo data.
