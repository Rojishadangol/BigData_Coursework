import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Bus Service Analytics",
    page_icon="🚌",
    layout="wide"
)

# =====================================================
# LOAD EXTERNAL CSS (falls back to inline defaults if missing)
# =====================================================

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("""
    <style>
    .main{ background-color:#f5f8fc; }
    .block-container{ padding-top:2rem; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown(
    """
    <div class='title-wrap'>
        <div class='eyebrow'>LIVE OPERATIONAL ANALYTICS</div><br>
        <div class='page-title'>🚌 Smart City Bus Service Analytics</div>
        <div class='page-subtitle'>Intelligent Bus Service Risk Classification and
        Operational Analytics Using PySpark Big Data</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# =====================================================
# PATHS
# =====================================================

from pathlib import Path

BASE_DIR = Path(__file__).parent

PROCESSED_PATH = BASE_DIR / "Data" / "processed" / "integrated_journeys_csv"
RESULTS_PATH = BASE_DIR / "Data" / "results"
VEHICLE_MAP_PATH = BASE_DIR / "Data" / "processed" / "vehicle_map.html"
VEHICLE_CSV_PATH = BASE_DIR / "Data" / "processed" / "vehicle_locations.csv"




import glob

@st.cache_data
def load_journeys(folder):

    csv_files = glob.glob(str(folder / "part-*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}"
        )

    return pd.read_csv(csv_files[0])


try:
    eda_df = load_journeys(PROCESSED_PATH)
    st.success(
        "Integrated Bus Service Dataset Loaded Successfully (Timetable + Disruptions + Location)"
    )
except Exception as e:
    st.error(f"Could not load journey dataset: {e}")
    st.stop()
    
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def safe_load_csv(path, label):
    try:
        return load_csv(path)
    except FileNotFoundError:
        st.warning(f"{label} not found at {path} — skipping this section.")
        return None


def model_slug(name):
    """'Decision Tree' -> 'decision_tree' — used to find per-model export files."""
    return name.strip().lower().replace(" ", "_")


def load_model_specific_csv(base_name, model_name, label):
    """Try RESULTS_PATH/<base_name>_<model_slug>.csv first (e.g. confusion_matrix_decision_tree.csv),
    then fall back to the older generic RESULTS_PATH/<base_name>.csv so this still works with
    exports from before the multi-model comparison was added."""
    specific_path = os.path.join(RESULTS_PATH, f"{base_name}_{model_slug(model_name)}.csv")
    generic_path = os.path.join(RESULTS_PATH, f"{base_name}.csv")

    if os.path.exists(specific_path):
        return load_csv(specific_path), model_name
    if os.path.exists(generic_path):
        st.caption(
            f"Note: showing `{base_name}.csv` — this was exported without a model-specific "
            f"filename, so it may not be the {model_name} result. Export "
            f"`{base_name}_{model_slug(model_name)}.csv` from 06_ML_Model.ipynb for an exact match."
        )
        return load_csv(generic_path), None
    st.warning(f"{label} not found for {model_name} — skipping this section.")
    return None, None


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Analytics",
        "Machine Learning",
        "Vehicle Map",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Dataset: BODS (3 catalogues)

Records:
227,010 Journeys
1,530 Disruptions
27,987 Vehicle Positions

Technology:

• PySpark
• Spark SQL
• MLlib
• MySQL
• Streamlit

Task:

Service Risk Classification
"""
)


# =====================================================
# SUMMARY METRICS
# =====================================================

total_journeys = len(eda_df)
operators = eda_df["Operator_ID"].nunique()
high_risk = int((eda_df["Service_Risk"] == 2).sum())
risk_percentage = (high_risk / total_journeys) * 100
disrupted_journeys = int(eda_df["Has_Disruption"].sum()) if "Has_Disruption" in eda_df.columns else None
tracked_journeys = int(eda_df["Has_Live_Tracking"].sum()) if "Has_Live_Tracking" in eda_df.columns else None

tiles = f"""
<div class="board-row">
    <div class="board-tile">
        <div class="value">{total_journeys:,}</div>
        <div class="label">Total Journeys</div>
    </div>
    <div class="board-tile">
        <div class="value">{operators}</div>
        <div class="label">Operators</div>
    </div>
    <div class="board-tile">
        <div class="value">{high_risk:,}</div>
        <div class="label">High Risk Services</div>
    </div>
    <div class="board-tile">
        <div class="value">{risk_percentage:.2f}%</div>
        <div class="label">Risk Percentage</div>
    </div>
</div>
"""
st.markdown(tiles, unsafe_allow_html=True)

if disrupted_journeys is not None and tracked_journeys is not None:
    join_tiles = f"""
    <div class="board-row">
        <div class="board-tile">
            <div class="value">{disrupted_journeys:,}</div>
            <div class="label">Journeys Linked to a Disruption</div>
        </div>
        <div class="board-tile">
            <div class="value">{disrupted_journeys / total_journeys * 100:.1f}%</div>
            <div class="label">Share Disruption-Affected</div>
        </div>
        <div class="board-tile">
            <div class="value">{tracked_journeys:,}</div>
            <div class="label">Journeys with Live Tracking</div>
        </div>
        <div class="board-tile">
            <div class="value">{tracked_journeys / total_journeys * 100:.1f}%</div>
            <div class="label">Share Live-Tracked</div>
        </div>
    </div>
    """
    st.markdown(join_tiles, unsafe_allow_html=True)

st.markdown("---")


# =====================================================
# OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.header("📊 Dashboard Overview")

    left, right = st.columns(2)

    # -----------------------------
    # Risk Distribution
    # -----------------------------
    with left:
        st.subheader("Service Risk Distribution")

        risk_counts = (
            eda_df["Service_Risk"]
            .value_counts()
            .sort_index()
        )

        risk_labels = ["Low Risk", "Medium Risk", "High Risk"]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(risk_labels, risk_counts.values)
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Number of Journeys")
        ax.set_title("Bus Service Risk Classification")
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------
    # Route Complexity
    # -----------------------------
    with right:
        st.subheader("Route Complexity Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(eda_df["Route_Complexity"], bins=30)
        ax.set_xlabel("Route Complexity")
        ax.set_ylabel("Frequency")
        ax.set_title("Route Complexity Pattern")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # -----------------------------
    # Peak Hour Analysis
    # -----------------------------
    st.subheader("🕒 Peak Hour Risk Analysis")

    peak_df = (
        eda_df
        .groupby(["Peak_Hour", "Service_Risk"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for risk in sorted(peak_df["Service_Risk"].unique()):
        temp = peak_df[peak_df["Service_Risk"] == risk]
        ax.plot(temp["Peak_Hour"], temp["count"], marker="o", label=f"Risk {risk}")

    ax.legend()
    ax.set_xlabel("Peak Hour")
    ax.set_ylabel("Number of Journeys")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------
    # Disruption Impact (new — evidence of the cross-catalogue join)
    # -----------------------------
    if "Has_Disruption" in eda_df.columns:
        st.markdown("---")
        st.subheader("⚠️ Disruption Impact on Risk Score")
        st.caption(
            "This comes from joining the Timetable catalogue with the separately-ingested "
            "Disruptions catalogue on Operator_ID — not a single-source metric."
        )

        disruption_impact = (
            eda_df
            .groupby("Has_Disruption")["Risk_Score"]
            .mean()
            .reset_index()
        )
        disruption_impact["Has_Disruption"] = disruption_impact["Has_Disruption"].map(
            {0: "No Disruption", 1: "Disruption Linked"}
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(disruption_impact["Has_Disruption"], disruption_impact["Risk_Score"], color=["#5B6478", "#FFB300"])
        ax.set_ylabel("Average Risk Score")
        ax.set_title("Average Risk Score: Disrupted vs Non-Disrupted Journeys")
        st.pyplot(fig)
        plt.close(fig)


# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    st.header("📈 Bus Service Analytics")

    left, right = st.columns(2)

    # -------------------------------------------------
    # Operator Risk Analysis
    # -------------------------------------------------
    with left:
        st.subheader("Top 10 Operators by Service Risk")

        operator_risk = (
            eda_df
            .groupby("Operator_ID")["Service_Risk"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="Total_Risk")
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(operator_risk["Operator_ID"], operator_risk["Total_Risk"])
        ax.set_xlabel("Total Risk Score")
        ax.set_ylabel("Operator")
        ax.set_title("Highest Risk Operators")
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------
    # Departure Hour Analysis
    # -------------------------------------------------
    with right:
        st.subheader("Journey Distribution by Departure Hour")

        hour_df = (
            eda_df
            .groupby("Departure_Hour")
            .size()
            .reset_index(name="count")
            .sort_values("Departure_Hour")
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hour_df["Departure_Hour"], hour_df["count"], marker="o")
        ax.set_xlabel("Departure Hour")
        ax.set_ylabel("Number of Journeys")
        ax.set_title("Bus Operation Pattern by Hour")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # -------------------------------------------------
    # Complexity vs Risk
    # -------------------------------------------------
    st.subheader("Route Complexity vs Service Risk")

    sample_df = eda_df.sample(n=min(5000, len(eda_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(sample_df["Route_Complexity"], sample_df["Service_Risk"], alpha=0.5)
    ax.set_xlabel("Route Complexity")
    ax.set_ylabel("Service Risk")
    ax.set_title("Relationship Between Route Complexity and Risk")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # -------------------------------------------------
    # Operator-Level Disruption & Tracking Summary (new)
    # -------------------------------------------------
    if {"Disruption_Count", "Position_Reports"}.issubset(eda_df.columns):
        st.subheader("🔗 Operator-Level Disruption & Live-Tracking Summary")
        st.caption("Aggregated from the Disruptions and Vehicle Location catalogues joined onto each operator.")

        operator_summary = (
            eda_df
            .groupby("Operator_ID")
            .agg(
                Total_Journeys=("Journey_ID", "count"),
                Total_Disruptions=("Disruption_Count", "max"),
                Total_Position_Reports=("Position_Reports", "max"),
            )
            .sort_values("Total_Disruptions", ascending=False)
            .head(10)
            .reset_index()
        )
        st.dataframe(operator_summary, use_container_width=True)

        st.markdown("---")

    # -------------------------------------------------
    # Statistical Summary
    # -------------------------------------------------
    st.subheader("📊 Statistical Summary")

    stats_df = pd.DataFrame({
        "Mean Complexity": [eda_df["Route_Complexity"].mean()],
        "Standard Deviation": [eda_df["Route_Complexity"].std()],
        "Skewness": [eda_df["Route_Complexity"].skew()],
        "Kurtosis": [eda_df["Route_Complexity"].kurt()],
    })

    st.dataframe(stats_df, use_container_width=True)

    # -------------------------------------------------
    # Dataset Information
    # -------------------------------------------------
    st.subheader("Dataset Information")

    dataset_info = pd.DataFrame({
        "Metric": ["Total Journeys", "Operators", "Features", "Prediction Target"],
        "Value": [total_journeys, operators, len(eda_df.columns), "Service Risk"],
    })

    st.table(dataset_info)

    st.markdown("---")

    # -------------------------------------------------
    # Risk Percentage
    # -------------------------------------------------
    st.subheader("Risk Category Percentage")

    risk_percentage_df = (
        eda_df["Service_Risk"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    risk_percentage_df.columns = ["Service_Risk", "count"]
    risk_percentage_df["Percentage"] = risk_percentage_df["count"] / total_journeys * 100

    st.dataframe(risk_percentage_df)


# =====================================================
# MACHINE LEARNING PAGE
# =====================================================

elif page == "Machine Learning":

    st.header("🤖 Machine Learning Performance")
    st.success("Classification Task: Bus Service Risk Prediction (8 features, incl. Disruption + Tracking)")

    # -------------------------------------------------
    # MODEL PERFORMANCE TABLE
    # -------------------------------------------------
    comparison = safe_load_csv(
        os.path.join(RESULTS_PATH, "model_comparison.csv"),
        "Model comparison data"
    )

    selected_model = None

    if comparison is not None:

        comparison["Accuracy"] *= 100
        comparison["Precision"] *= 100
        comparison["Recall"] *= 100
        comparison["F1 Score"] *= 100
        comparison = comparison.round(2)

        st.subheader("📋 Model Comparison")
        st.dataframe(comparison, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------
        # MODEL ACCURACY GRAPH
        # -------------------------------------------------
        st.subheader("📊 Model Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(comparison["Model"], comparison["Accuracy"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Machine Learning Model Accuracy")
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")

        # -------------------------------------------------
        # PRECISION RECALL F1 COMPARISON
        # -------------------------------------------------
        st.subheader("Performance Metrics Comparison")

        metric_df = comparison.set_index("Model")

        fig, ax = plt.subplots(figsize=(10, 5))
        metric_df[["Precision", "Recall", "F1 Score"]].plot(kind="bar", ax=ax)
        ax.set_ylabel("Score (%)")
        ax.set_title("Precision Recall F1 Score Comparison")
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")

        # -------------------------------------------------
        # BEST MODEL (computed dynamically — never hardcoded)
        # -------------------------------------------------
        best_model = comparison.loc[comparison["Accuracy"].idxmax()]
        second_best = comparison.loc[
            comparison["Accuracy"].sort_values(ascending=False).index[1]
        ]
        selected_model = best_model["Model"]

        c1, c2 = st.columns(2)

        with c1:
            st.info(f"""
### 🏆 Best Performing Model

**{best_model['Model']}**

**Accuracy:** **{best_model['Accuracy']:.2f}%**

**F1 Score:** **{best_model['F1 Score']:.2f}%**

{best_model['Model']} achieved the highest classification performance among the models
compared, using route complexity, number of stops, departure hour, peak-hour operations,
disruption count/flag, and live-tracking count/flag as input features.
""")

        with c2:
            st.warning(f"""
### 📌 Observation

{second_best['Model']} was the next strongest performer at {second_best['Accuracy']:.2f}% accuracy.

Performance differences between models reflect how well each algorithm captures the
non-linear threshold structure underlying the engineered `Service_Risk` label — see the
Limitations section of the report for a full discussion of what these figures do and don't
demonstrate.
""")

        # -------------------------------------------------
        # Let the user pick which model's confusion matrix / feature
        # importance to view (defaults to the best model above)
        # -------------------------------------------------
        model_options = comparison["Model"].tolist()
        default_index = model_options.index(selected_model) if selected_model in model_options else 0
        selected_model = st.selectbox(
            "View Confusion Matrix / Feature Importance for:",
            model_options,
            index=default_index,
        )

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------
    st.markdown("---")

    display_model = selected_model or "Best Model"
    st.subheader(f"Confusion Matrix — {display_model}")

    confusion_df, matched_model = load_model_specific_csv(
        "confusion_matrix", display_model, "Confusion matrix data"
    )

    if confusion_df is not None:
        matrix = (
            confusion_df
            .pivot(index="Service_Risk", columns="prediction", values="count")
            .fillna(0)
        )
        matrix.index = [f"Actual {int(i)}" for i in matrix.index]
        matrix.columns = [f"Predicted {int(float(i))}" for i in matrix.columns]

        st.dataframe(matrix, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(matrix)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticklabels(matrix.index)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"{display_model} Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------
    st.markdown("---")
    st.subheader(f"Feature Importance — {display_model}")

    importance, _ = load_model_specific_csv(
        "feature_importance", display_model, "Feature importance data"
    )

    if importance is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance["Feature"], importance["Importance"])
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{display_model} Feature Importance")
        st.pyplot(fig)
        plt.close(fig)


# =====================================================
# VEHICLE MAP PAGE
# =====================================================

elif page == "Vehicle Map":

    st.header("🗺️ Live Vehicle Location Snapshot")

    st.markdown(
        "Real-time vehicle positions extracted from a BODS SIRI-VM snapshot "
        "(see `02_Vehicle_Location_Extraction.ipynb`). Aggregated position-report counts "
        "per operator are also joined into the main journey dataset (`Position_Reports`, "
        "`Has_Live_Tracking`) — this map is the supplementary geospatial view of that same data."
    )

    if os.path.exists(VEHICLE_MAP_PATH):
        with open(VEHICLE_MAP_PATH, "r", encoding="utf-8") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=600, scrolling=True)
    else:
        st.warning(
            f"Vehicle map not found at {VEHICLE_MAP_PATH}. "
            "Run 02_Vehicle_Location_Extraction.ipynb first to generate it."
        )

    vehicle_df = safe_load_csv(VEHICLE_CSV_PATH, "Vehicle location data")

    if vehicle_df is not None:
        st.markdown("---")
        st.subheader("Snapshot Summary")

        vc1, vc2, vc3 = st.columns(3)

        with vc1:
            st.metric("Vehicle Records", f"{len(vehicle_df):,}")
        with vc2:
            st.metric("Distinct Operators", vehicle_df["Operator_ID"].nunique())
        with vc3:
            st.metric("Distinct Lines", vehicle_df["Line_Name"].nunique())

        st.dataframe(vehicle_df.head(20), use_container_width=True)


# =====================================================
# ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    st.header("ℹ About This Project")

    st.markdown("""
## Project Title

**Intelligent Bus Service Risk Classification and Operational Analytics Using PySpark Big Data**
""")

    st.markdown("---")
    st.subheader("🎯 Business Problem")

    st.write("""
Public transport systems generate large amounts of operational data every day, across
separate catalogues — timetables, disruptions, and live vehicle location.

This project joins three separate BODS catalogues (rather than analysing them in isolation)
to identify service risk levels and understand factors affecting operational complexity,
including the impact of live disruptions and vehicle tracking coverage.

The system uses Big Data processing techniques and Machine Learning models to classify
journeys into:

• Low Risk
• Medium Risk
• High Risk

The analysis helps transport authorities and operators identify potentially problematic
services and improve decision making.
""")

    st.markdown("---")
    st.subheader("👥 Business Stakeholders")

    stakeholders = pd.DataFrame({
        "Stakeholder": [
            "Transport Authority",
            "Bus Operators",
            "Urban Planning Teams",
            "Smart City Departments",
        ],
        "Purpose": [
            "Monitor service performance",
            "Improve operational efficiency",
            "Plan better transport networks",
            "Support data-driven decisions",
        ],
    })

    st.table(stakeholders)

    st.markdown("---")
    st.subheader("🛠 Technologies Used")

    technology = pd.DataFrame({
        "Technology": [
            "PySpark", "Spark SQL", "Spark MLlib", "MySQL", "Python",
            "Matplotlib", "Streamlit", "Folium",
        ],
        "Purpose": [
            "Distributed data processing",
            "Large-scale data analysis",
            "Machine learning classification",
            "Relational storage of the 6-table ER schema (loaded via CSV export + Workbench Import Wizard)",
            "Data processing and programming",
            "Data visualization",
            "Interactive dashboard",
            "Geospatial vehicle map visualization",
        ],
    })

    st.table(technology)

    st.markdown("---")
    st.subheader("📂 Dataset Summary")

    dataset_summary = pd.DataFrame({
        "Information": [
            "Journey Records", "Disruption Records", "Vehicle Position Records",
            "Operators", "Input Features",
            "Prediction Target", "Machine Learning Models",
        ],
        "Value": [
            f"{total_journeys:,}", "1,530", "27,987",
            operators, "8 Features",
            "Service Risk", "Decision Tree, Random Forest, Logistic Regression",
        ],
    })

    st.dataframe(dataset_summary, use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Project Workflow")

    workflow = pd.DataFrame({
        "Stage": [
            "01 - Data Collection",
            "02 - Vehicle Location Extraction",
            "03 - SQL Analysis & Relational Schema",
            "04 - Data Preprocessing",
            "05 - EDA Visualization",
            "06 - Machine Learning",
            "Dashboard Development",
        ],
        "Description": [
            "Parsed Timetables (TransXChange) and Disruptions (SIRI-SX) BODS catalogues",
            "Extracted real-time vehicle positions (SIRI-VM) as a third, separate catalogue",
            "Joined the three catalogues in Spark SQL and loaded a 6-table MySQL ER schema via CSV export",
            "Repartitioned/cached the integrated dataset and assembled the 8-feature vector",
            "Profiled distributions, correlations and disruption/tracking patterns",
            "Trained and compared Decision Tree, Random Forest and Logistic Regression",
            "Built this interactive analytics dashboard",
        ],
    })

    st.table(workflow)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("""
🚌 Smart City Bus Service Analytics Dashboard

Developed using:

PySpark • Spark SQL • MLlib • MySQL • Python • Streamlit • Folium

Big Data Programming Coursework
""")