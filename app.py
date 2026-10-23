from pathlib import Path
import streamlit as st
import pandas as pd
from src.data_pipeline import load_and_clean
from src.risk_model import train_model, predict_one
from src.attention import encode_sequence, self_attention, plot_attention
from src.rag_index import query_guidelines
from src.report_generator import generate_demo_report, generate_llm_report, save_report


st.set_page_config(page_title="mRNA-VIP Educational Demo", page_icon="🧬", layout="wide")
st.title("🧬 mRNA-VIP: Educational Safety Analysis Pipeline")
st.error("STUDENT DEMO ONLY — NOT CLINICAL GUIDANCE. Do not use this app for real vaccine eligibility, dosing, or patient care.")
st.caption("This app demonstrates ML, toy self-attention, document retrieval (RAG), structured reports, and visualization.")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "patient_records (1).csv"
GUIDE_PATH = BASE_DIR / "data" / "mrna_safety_guidelines.txt"

@st.cache_data
def load_data():
    return load_and_clean(DATA_PATH)

@st.cache_resource
def get_model():
    df = load_data()
    return train_model(df)

with st.spinner("Loading data and training the educational model..."):
    df = load_data()
    model, metrics = get_model()

tab1, tab2, tab3, tab4 = st.tabs(["Overview & ML", "Toy Attention", "Guideline Search (RAG)", "Structured Report"])

with tab1:
    st.subheader("Dataset")
    st.write(f"Rows: **{len(df)}** · Columns: **{len(df.columns)}**")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Model evaluation")
    st.metric("ROC AUC", f"{metrics['auc']:.3f}")
    st.write("Classification report")
    st.dataframe(pd.DataFrame(metrics["classification_report"]).T, use_container_width=True)
    st.write("Confusion matrix")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0","Actual 1"],
                              columns=["Predicted 0","Predicted 1"]))
    st.caption("These metrics describe only this supplied sample dataset, not clinical validity.")
    st.subheader("Try a sample record")
    with st.form("predict_form"):
        c1,c2,c3 = st.columns(3)
        age = c1.number_input("Age", 18, 85, 45)
        biomarker = c2.number_input("Biomarker level", 0.0, 100.0, 4.5)
        conditions = c3.number_input("Pre-existing conditions", 0, 4, 1)
        c4,c5 = st.columns(2)
        dose = c4.number_input("Dataset dose field (mcg)", 25.0, 150.0, 80.0)
        history = c5.selectbox("Prior reaction history", ["None","Mild","Moderate"])
        submitted = st.form_submit_button("Predict dataset label")
    if submitted:
        pred, prob = predict_one(model, age, biomarker, conditions, dose, history)
        st.write(f"Predicted `Reaction_Score`: **{pred}**")
        st.write(f"Model probability for label 1: **{prob:.3f}**")
        st.warning("This is a dataset label prediction only, not a medical risk assessment.")

with tab2:
    st.subheader("Self-attention demonstration")
    sequence = st.text_input("Enter a short sequence using A, C, G, U", "ACGUACGUACGGCUA").upper()
    if st.button("Calculate attention"):
        try:
            X = encode_sequence(sequence)
            weights, output = self_attention(X)
            fig = plot_attention(weights, sequence)
            st.pyplot(fig)
            st.write("Attention matrix shape:", weights.shape)
            st.write("Each row sums to approximately 1:", bool(abs(weights.sum(axis=1)-1).max() < 1e-6))
            st.caption("Toy random embeddings/projections only; not a trained biological model.")
        except ValueError as exc:
            st.error(str(exc))

with tab3:
    st.subheader("Search the local text file")
    st.info("The supplied guideline file is a placeholder, not an official regulatory or clinical source.")
    question = st.text_input("Question", "What does the placeholder say about dosing?")
    if st.button("Retrieve relevant text"):
        try:
            with st.spinner("Creating embeddings and searching the local document..."):
                chunks, latency = query_guidelines(question, k=3)
            st.write(f"Retrieval latency: {latency:.3f} seconds")
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Retrieved chunk {i}**")
                st.write(chunk)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.caption("Check requirements installation and internet access for the first embedding-model download.")

with tab4:
    st.subheader("Create a structured educational report")
    patient_summary = st.text_area("Patient summary (use synthetic/non-identifying details)", "Synthetic example record; no real patient data.")
    report_mode = st.radio("Report mode", ["Safe demo (no API key needed)", "Optional LLM formatting (uses paid API)"])
    if st.button("Generate report"):
        try:
            with st.spinner("Retrieving relevant text..."):
                chunks, latency = query_guidelines("safety limits eligibility and dose", k=3)
            pred, prob = predict_one(model, 45, 4.5, 1, 80.0, "None")
            label = "High" if pred == 1 else "Low"
            if report_mode.startswith("Optional"):
                report = generate_llm_report(label, prob, chunks, patient_summary)
            else:
                report = generate_demo_report(label, prob, chunks, patient_summary)
            json_path, md_path = save_report(report, stem="latest_report")
            st.json(report.model_dump())
            st.success(f"Saved report files: {json_path} and {md_path}")
            st.download_button("Download JSON report", data=json_path.read_text(encoding="utf-8"),
                               file_name="educational_safety_report.json", mime="application/json")
            st.download_button("Download Markdown report", data=md_path.read_text(encoding="utf-8"),
                               file_name="educational_safety_report.md", mime="text/markdown")
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption("Engineering portfolio project only. A Pydantic-valid response is not proof of clinical correctness.")
