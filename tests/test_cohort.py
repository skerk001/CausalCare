"""
CausalCare: Unit Tests for Cohort Construction
================================================
Validates data integrity, treatment assignment logic, and confounder extraction.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def analysis_dataset():
    """Load the processed analysis dataset if available."""
    path = Path("data/processed/analysis_dataset.csv")
    if not path.exists():
        pytest.skip("analysis_dataset.csv not found — run src/cohort.py first")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Data Integrity
# ---------------------------------------------------------------------------
class TestDataIntegrity:
    """Ensure the analysis dataset meets basic quality standards."""

    def test_no_duplicate_stays(self, analysis_dataset):
        """Each ICU stay should appear exactly once."""
        assert analysis_dataset["patientunitstayid"].is_unique

    def test_treatment_is_binary(self, analysis_dataset):
        """Treatment column must be 0 or 1."""
        assert set(analysis_dataset["early_bb"].unique()).issubset({0, 1})

    def test_outcome_is_binary(self, analysis_dataset):
        """Outcome column must be 0 or 1."""
        assert set(analysis_dataset["hospital_mortality"].unique()).issubset({0, 1})

    def test_positive_icu_los(self, analysis_dataset):
        """ICU length of stay should be non-negative."""
        assert (analysis_dataset["icu_los_hours"] >= 0).all()

    def test_age_range(self, analysis_dataset):
        """Age should be within plausible range (18–100)."""
        ages = analysis_dataset["age"].dropna()
        assert ages.min() >= 18
        assert ages.max() <= 100

    def test_minimum_sample_size(self, analysis_dataset):
        """Dataset should have at least 1000 observations."""
        assert len(analysis_dataset) >= 1000

    def test_treatment_prevalence(self, analysis_dataset):
        """Treatment prevalence should be between 5% and 50% for positivity."""
        prev = analysis_dataset["early_bb"].mean()
        assert 0.05 <= prev <= 0.50, f"Treatment prevalence {prev:.1%} outside expected range"


# ---------------------------------------------------------------------------
# Treatment Assignment
# ---------------------------------------------------------------------------
class TestTreatmentAssignment:
    """Validate treatment assignment logic."""

    def test_both_groups_present(self, analysis_dataset):
        """Both treated and control groups must be present."""
        assert analysis_dataset["early_bb"].sum() > 0
        assert (analysis_dataset["early_bb"] == 0).sum() > 0

    def test_outcome_exists_in_both_groups(self, analysis_dataset):
        """Mortality events should exist in both treatment groups."""
        for group in [0, 1]:
            subset = analysis_dataset[analysis_dataset["early_bb"] == group]
            assert subset["hospital_mortality"].sum() > 0


# ---------------------------------------------------------------------------
# Confounder Completeness
# ---------------------------------------------------------------------------
class TestConfounders:
    """Ensure key confounders are present and reasonably complete."""

    REQUIRED_CONFOUNDERS = [
        "age", "female", "apachescore", "hx_htn", "hx_chf",
        "on_mech_vent", "on_vasopressors", "n_comorbidities",
    ]

    def test_required_confounders_exist(self, analysis_dataset):
        """All required confounders must be present as columns."""
        missing = [c for c in self.REQUIRED_CONFOUNDERS if c not in analysis_dataset.columns]
        assert not missing, f"Missing confounders: {missing}"

    def test_confounder_missingness_below_threshold(self, analysis_dataset):
        """No required confounder should have > 30% missing values."""
        for col in self.REQUIRED_CONFOUNDERS:
            if col in analysis_dataset.columns:
                miss_rate = analysis_dataset[col].isna().mean()
                assert miss_rate <= 0.30, f"{col} has {miss_rate:.1%} missing"
