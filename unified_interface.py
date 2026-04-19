# -*- coding: utf-8 -*-
"""
Unified XGBoost Prediction Interface

Maps physical input parameters →
    1. Failure mode (classification)
    2. Ultimate load (regression)

Author: [Your Name]
"""

import pandas as pd
import joblib


class CatBoostUnifiedInterface:
    """
    Unified inference interface for XGBoost models.

    Mathematical representation:
        f : R^n → (Y_class, Y_reg)

    where:
        Y_class → Failure mode (categorical)
        Y_reg   → Ultimate load (continuous)
    """

    def __init__(self, clf_model_path: str, reg_model_path: str,
                 feature_names: list, label_names: list):
        """
        Parameters
        ----------
        clf_model_path : str
            Path to trained classification model (.pkl)
        reg_model_path : str
            Path to trained regression model (.pkl)
        feature_names : list
            Ordered feature list (MUST match training)
        label_names : list
            Mapping from class index → failure mode label
        """

        self.clf_model = joblib.load(clf_model_path)
        self.reg_model = joblib.load(reg_model_path)

        self.feature_names = feature_names
        self.label_names = label_names

    # =========================
    # Internal utilities
    # =========================
    def _merge_input(self, input_data: dict) -> dict:
        """
        Merge structured input (geometry/material/loading)
        into a flat dictionary.
        """
        flat_input = {}
        for section_name, section_values in input_data.items():
            if not isinstance(section_values, dict):
                raise ValueError(f"{section_name} must be a dictionary.")
            flat_input.update(section_values)
        return flat_input

    def _validate_input(self, flat_input: dict):
        """
        Ensure all required features exist.
        """
        missing = [f for f in self.feature_names if f not in flat_input]
        if missing:
            raise ValueError(f"Missing features: {missing}")

    # =========================
    # Main prediction
    # =========================
    def predict(self, input_data: dict) -> dict:
        """
        Perform unified prediction.

        Parameters
        ----------
        input_data : dict
            {
              "geometry": {...},
              "limite condition": {...},
              "material": {...}
              
            }

        Returns
        -------
        dict
            {
              'failure_mode': str,
              'ultimate_load': float
            }
        """

        # Merge inputs
        flat_input = self._merge_input(input_data)

        # Validate
        self._validate_input(flat_input)

        # Build DataFrame
        X = pd.DataFrame([flat_input])[self.feature_names]

        # --- Classification ---
        class_index = int(self.clf_model.predict(X)[0])
        failure_mode = self.label_names[class_index]

        # --- Regression ---
        ultimate_load = float(self.reg_model.predict(X)[0])

        return {
            "failure_mode": failure_mode,
            "ultimate_load": round(ultimate_load, 2)
        }

    # =========================
    # Optional: with confidence
    # =========================
    def predict_with_confidence(self, input_data: dict) -> dict:
        """
        Same as predict(), but includes classification confidence.
        """

        flat_input = self._merge_input(input_data)
        self._validate_input(flat_input)

        X = pd.DataFrame([flat_input])[self.feature_names]

        class_index = int(self.clf_model.predict(X)[0])
        probabilities = self.clf_model.predict_proba(X)[0]

        failure_mode = self.label_names[class_index]
        confidence = float(max(probabilities))

        ultimate_load = float(self.reg_model.predict(X)[0])

        return {
            "failure_mode": failure_mode,
            "confidence": round(confidence, 3),
            "ultimate_load": round(ultimate_load, 2)
        }


# =========================
# Feature & Label Space
# =========================

feature_names = [
    "L","H1","Bf1",
    "Bl1","t1","R1",
    "e","a","k",
    "p","d","LC","Fy"
]

label_names = ['L', 'D', 'G', 'L+D', 'L+G', 'FT', 'L+FT']


# =========================
# Instantiate Interface
# =========================

interface = CatBoostUnifiedInterface(
    clf_model_path=r"D:\data AI\SAMSOUMA\best_catboost_classification.pkl",
    reg_model_path=r"D:\data AI\SAMSOUMA\best_catboost_regression.pkl",
    
    feature_names=feature_names,
    label_names=label_names
)


# =========================
# Example Input (Structured)
# =========================

input_data = {

    "geometry": {
        
        "L": 400,
        "H1": 50,
        "Bf1": 20,
        "Bl1": 10,
        "t1":1.2,
        "R1":2.3,
        "e":25,
        "a":13,
        "k":21,
        "p":12,
        "d":11,
        
        
    },
     
    "limite condition": {
        "LC": 1,
    },


    "material": {
        "Fy": 350,
        
    },
}


# =========================
# Run Prediction
# =========================

result = interface.predict_with_confidence(input_data)

print("Failure mode :", result["failure_mode"])
print("Confidence   :", result["confidence"])
print("Ultimate load:", result["ultimate_load"])