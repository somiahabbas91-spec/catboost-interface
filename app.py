import streamlit as st
import pandas as pd
import joblib

# =========================
# INTERFACE CLASS (FULL)
# =========================
class CatBoostUnifiedInterface:

    def __init__(self, clf_model_path, reg_model_path,
                 feature_names, label_names):

        self.clf_model = joblib.load(clf_model_path)
        self.reg_model = joblib.load(reg_model_path)

        self.feature_names = feature_names
        self.label_names = label_names

    def _merge_input(self, input_data):
        flat_input = {}
        for section in input_data.values():
            flat_input.update(section)
        return flat_input

    def predict_with_confidence(self, input_data):

        flat_input = self._merge_input(input_data)

        X = pd.DataFrame([flat_input])[self.feature_names]

        # Classification
        class_pred = self.clf_model.predict(X)
        class_index = int(class_pred.flatten()[0])

        probs = self.clf_model.predict_proba(X)[0]
        confidence = float(max(probs))

        failure_mode = self.label_names[class_index]

        # Regression
        ultimate_load = float(self.reg_model.predict(X)[0])

        return {
            "failure_mode": failure_mode,
            "confidence": round(confidence, 3),
            "ultimate_load": round(ultimate_load, 2)
        }


# =========================
# FEATURE LIST (CLEANED)
# =========================
feature_names=[
        "L","H1","Bf1",
        "Bl1","t1","R1",
        "e","a","k",
        "p","d","LC","Fy"
    ]


label_names = ['L', 'D', 'G', 'L+D', 'L+G', 'FT', 'L+FT']

# =========================
# LOAD MODELS
# =========================
interface = CatBoostUnifiedInterface(
    clf_model_path="best_catboost_classification.pkl",
    reg_model_path="best_catboost_regression.pkl",
    feature_names=feature_names,
    label_names=label_names
)

# =========================
# STREAMLIT UI
# =========================
st.title("Structural Failure Prediction System")

st.header("Input Parameters")

L = st.number_input("L", value=400.0)
H1 = st.number_input("H1", value=50.0)
Bf1 = st.number_input("Bf1", value=20.0)
Bl1 = st.number_input("Bl1", value=10.0)
t1 = st.number_input("t1", value=1.2)
R1 = st.number_input("R1", value=2.3)
e = st.number_input("e", value=25.0)
a = st.number_input("a", value=13.0)
k = st.number_input("k", value=21.0)
p = st.number_input("p", value=12.0)
d = st.number_input("d", value=11.0)
LC = st.number_input("LC", value=1.0)
Fy = st.number_input("Fy", value=350.0)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict"):

   "geometry": {
            "L": L,
            "H1": H1,
            "Bf1": Bf1,
            "Bl1": Bl1,
            "t1": t1,
            "R1": R1,
            "e": e,
            "a": a,
            "k": k,
            "p": p,
            "d": d,
        },
        "limite condition": {
            "LC": LC,
        },
        "material": {
            "Fy": Fy,
        },
    }

    result = interface.predict_with_confidence(input_data)

    st.success("Prediction completed")

    st.write("Failure mode:", result["failure_mode"])
    st.write("Confidence:", result["confidence"])
    st.write("Ultimate load:", result["ultimate_load"])
