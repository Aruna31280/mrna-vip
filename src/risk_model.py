
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
]

CATEGORICAL_FEATURES = ["Prior_Reaction_History"]
TARGET = "Reaction_Score"


def train_model(df: pd.DataFrame):
    """Train a demonstration classifier and return its evaluation metrics."""

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    if y.nunique() < 2:
        raise ValueError("Reaction_Score must contain at least two classes.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    positive_probabilities = probabilities[:, positive_index]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=classes
        ).tolist(),
        "auc": (
            roc_auc_score(y_test, positive_probabilities)
            if y_test.nunique() == 2
            else 0.0
        ),
    }

    return model, metrics


def predict_one(
    model,
    age,
    biomarker,
    conditions,
    dose,
    history,
):
    """Predict a dataset label for one synthetic example."""

    sample = pd.DataFrame([{
        "Age": age,
        "Biomarker_Level": biomarker,
        "Pre_Existing_Conditions": conditions,
        "Vaccine_Dose_mcg": dose,
        "Prior_Reaction_History": history,
    }])

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    classes = list(model.classes_)

    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    probability = float(probabilities[positive_index])

    return prediction, probability
