
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def build_dashboard(df, model=None):
    """Display a basic educational dashboard."""

    st.header("Project Dashboard")

    col1, col2 = st.columns(2)
    col1.metric("Total records", len(df))
    col2.metric("Total columns", len(df.columns))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if "Reaction_Score" in df.columns:
        st.subheader("Reaction Score Distribution")
        counts = df["Reaction_Score"].value_counts().sort_index()
        st.bar_chart(counts)

    if model is not None and "Reaction_Score" in df.columns:
        try:
            feature_columns = [
                "Age",
                "Biomarker_Level",
                "Pre_Existing_Conditions",
                "Vaccine_Dose_mcg",
                "Prior_Reaction_History",
            ]

            X = df[feature_columns]
            y = df["Reaction_Score"]

            if y.nunique() == 2:
                probabilities = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, probabilities)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label="ROC curve")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve (training-data demonstration)")
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

        except Exception as exc:
            st.info(f"ROC curve unavailable: {exc}")
