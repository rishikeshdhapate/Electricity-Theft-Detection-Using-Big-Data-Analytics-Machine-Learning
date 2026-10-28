"""
Electricity Theft Detection Dashboard
Simple workflow:
1. Upload the historical consumer CSV.
2. Apache Spark reads the CSV and extracts the four training features.
3. The saved Spark Weighted Random Forest predicts theft-risk scores.
4. The dashboard lists consumers flagged for investigation.

Expected model folder: models/weighted_random_forest
Optional override: SPARK_MODEL_PATH=/absolute/path/to/saved/model

The model was trained on summary features from the full historical reading
window (~1,034 daily readings). Use the same historical format for predictions.
A flagged consumer is a lead for investigation, not proof of theft.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
)

THRESHOLD = 0.55
FEATURE_COLS = [
    "valid_readings",
    "missing_ratio",
    "mean_consumption",
    "total_consumption",
]
ID_CANDIDATES = [
    "CONS_NO", "cons_no", "consumer_id", "Consumer_ID", "CUSTOMER_ID"
]


@st.cache_resource(show_spinner="Starting Apache Spark and loading the model...")
def load_spark_and_model(model_path: str):
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel

    spark = (
        SparkSession.builder
        .appName("ElectricityTheftDetectionDashboard")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(model_path)
    return spark, model


def find_model_path():
    env_path = os.environ.get("SPARK_MODEL_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        Path("models/weighted_random_forest"),
        Path("../models/weighted_random_forest"),
        Path("../../models/weighted_random_forest"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def find_id_column(columns):
    by_lower = {str(c).lower(): c for c in columns}
    for candidate in ID_CANDIDATES:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return columns[0] if columns else "consumer_id"


def extract_features_spark(spark, raw_path: str):
    """
    Read the uploaded historical CSV and calculate the same four features
    used by training. Higher-order array operations avoid constructing very
    deep chained Spark sum expressions across ~1,034 reading columns.
    """
    from pyspark.sql import functions as F

    raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(raw_path)
    )

    if len(raw.columns) < 3:
        raise ValueError(
            "The CSV needs a consumer ID column and multiple daily-reading columns."
        )

    id_col = find_id_column(raw.columns)
    label_col = next((c for c in raw.columns if str(c).upper() == "FLAG"), None)
    reading_cols = [c for c in raw.columns if c not in {id_col, label_col}]

    if len(reading_cols) < 2:
        raise ValueError("Could not identify enough daily-reading columns.")

    # Cast readings once, then put them in one array per consumer.
    numeric_readings = [
        F.col(c).cast("double").alias(f"__reading_{i}")
        for i, c in enumerate(reading_cols)
    ]
    numeric = raw.select(
        F.col(id_col).cast("string").alias("CONS_NO"),
        *numeric_readings,
        *([F.col(label_col).cast("int").alias("FLAG")] if label_col else []),
    )

    renamed = [f"__reading_{i}" for i in range(len(reading_cols))]
    with_array = numeric.withColumn(
        "__readings",
        F.array(*[F.col(c) for c in renamed]),
    )

    valid_count = F.size(
        F.filter(F.col("__readings"), lambda x: x.isNotNull())
    )
    total = F.aggregate(
        F.col("__readings"),
        F.lit(0.0),
        lambda acc, x: acc + F.coalesce(x, F.lit(0.0)),
    )

    features = with_array.select(
        "CONS_NO",
        *([F.col("FLAG")] if label_col else []),
        valid_count.cast("double").alias("valid_readings"),
        (
            F.lit(1.0)
            - valid_count.cast("double") / F.lit(float(len(reading_cols)))
        ).alias("missing_ratio"),
        F.when(
            valid_count > 0,
            total / valid_count.cast("double"),
        ).otherwise(F.lit(None).cast("double")).alias("mean_consumption"),
        total.alias("total_consumption"),
    )
    return features, len(reading_cols), id_col, label_col is not None


def main():
    st.title("⚡ Electricity Theft Detection")
    st.write(
        "Upload a historical consumer-reading CSV. Spark extracts features, "
        "the trained model scores each consumer, and the dashboard lists those "
        "flagged for investigation."
    )
    st.caption(
        "Positive predictions are screening alerts—not confirmation of electricity theft."
    )

    model_path = find_model_path()
    if not model_path:
        st.error(
            "Saved Spark model not found. Place it at "
            "`models/weighted_random_forest` relative to this app, or set "
            "`SPARK_MODEL_PATH` to the model folder's absolute path."
        )
        st.code(
            "export SPARK_MODEL_PATH='/absolute/path/to/weighted_random_forest'\n"
            "streamlit run app.py"
        )
        st.stop()

    uploaded = st.file_uploader(
        "Upload historical consumer readings CSV",
        type=["csv"],
    )
    if uploaded is None:
        st.info(
            "Expected format: one consumer per row, a consumer ID column, "
            "and the full historical daily-reading columns. An optional `FLAG` "
            "column is excluded from model inputs."
        )
        return

    st.caption(f"Selected file: {uploaded.name} · {uploaded.size:,} bytes")
    if not st.button("Run prediction", type="primary"):
        return

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(uploaded.getvalue())
            temp_path = temp_file.name

        spark, model = load_spark_and_model(model_path)

        with st.spinner("Spark is loading the CSV and extracting features..."):
            features, reading_count, id_col, has_labels = extract_features_spark(
                spark, temp_path
            )
            row_count = features.count()

        if reading_count <= 60:
            st.error(
                f"Found only {reading_count} daily-reading columns. This model "
                "was trained on the full historical reading window, so use the "
                "matching historical format rather than a 30-day-only CSV."
            )
            st.stop()

        with st.spinner(f"Scoring {row_count:,} consumers with the trained model..."):
            from pyspark.ml.functions import vector_to_array
            from pyspark.sql import functions as F

            predictions = (
                model.transform(features)
                .withColumn(
                    "risk_score",
                    vector_to_array(F.col("probability"))[1],
                )
                .withColumn(
                    "predicted_label",
                    (F.col("risk_score") >= F.lit(THRESHOLD)).cast("int"),
                )
            )

            flagged = (
                predictions
                .filter(F.col("predicted_label") == 1)
                .select(
                    "CONS_NO",
                    "risk_score",
                    "valid_readings",
                    "total_consumption",
                    "mean_consumption",
                    "missing_ratio",
                    *([F.col("FLAG")] if has_labels else []),
                )
                .orderBy(F.col("risk_score").desc())
            )

            flagged_pdf = flagged.toPandas()
            all_count = predictions.count()
            flagged_count = len(flagged_pdf)

        if not flagged_pdf.empty:
            flagged_pdf["risk_score"] = (flagged_pdf["risk_score"] * 100).round(2)
            flagged_pdf = flagged_pdf.rename(
                columns={"risk_score": "risk_score_percent"}
            )

        st.success("Prediction completed.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Consumers processed", f"{all_count:,}")
        col2.metric("Flagged for investigation", f"{flagged_count:,}")
        col3.metric("Decision threshold", f"{THRESHOLD:.2f}")

        st.subheader("Consumers suspected of theft")
        if flagged_pdf.empty:
            st.success(
                "No consumers crossed the current model threshold in this upload."
            )
        else:
            st.dataframe(
                flagged_pdf,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download flagged consumers (CSV)",
                data=flagged_pdf.to_csv(index=False).encode("utf-8"),
                file_name="suspected_electricity_theft_consumers.csv",
                mime="text/csv",
            )

        st.caption(
            "The dashboard displays model-flagged consumers only. Investigate alerts "
            "using independent evidence before drawing conclusions."
        )

        if has_labels:
            st.info(
                "The uploaded file includes `FLAG`, which was not used as a model input. "
                "This dashboard does not calculate evaluation metrics automatically; "
                "evaluation is only valid when these labels represent the same period "
                "and the data was not used to train the model."
            )

    except Exception as exc:
        st.error("The dashboard could not complete the batch prediction.")
        st.exception(exc)
        st.caption(
            "If the error persists, share the complete traceback and the training "
            "code that saved the Spark PipelineModel."
        )
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
