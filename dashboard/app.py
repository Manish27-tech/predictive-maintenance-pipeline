"""Streamlit dashboard for predictive maintenance.

Run from the repository root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from maintenance.evaluate import risk_tier
from maintenance.features import model_input
from maintenance.model import load_model
from maintenance.policy import action_for_tier

ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {background: linear-gradient(135deg, #123b4a, #1f6f8b); color: white;
           padding: 26px 30px; border-radius: 16px; margin-bottom: 18px;}
    .hero h1 {margin: 0 0 6px; font-size: 2rem;}
    .hero p {margin: 0; color: #d7eef5; max-width: 900px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_assets():
    model = load_model(ARTIFACTS / "maintenance_model.joblib")
    template = load_json(str(ARTIFACTS / "dashboard" / "input_template.json"))
    return model, template


def predict_probability(model, template: dict, overrides: dict) -> float:
    row = dict(template)
    row.update(overrides)
    return float(model.predict_proba(model_input(pd.DataFrame([row])))[0, 1])


feed = load_json(str(ARTIFACTS / "dashboard" / "feed.json"))
metrics = feed["metrics"]
summary = feed["summary"]
policy = feed["policy"]
comparison = pd.DataFrame(feed["comparison"])
modes = pd.DataFrame(feed["modes"])
tiers = pd.DataFrame(feed["tiers"])
importance = pd.DataFrame(feed["importance"])
policy_curve = load_csv(str(ARTIFACTS / "policy_curve.csv"))
slices = load_csv(str(ARTIFACTS / "operational_slices.csv"))
model, template = load_assets()

st.markdown(
    f"""
    <div class="hero">
      <h1>⚙️ Predictive Maintenance Pipeline</h1>
      <p>Machine-failure prediction on the AI4I 2020 benchmark with physics-inspired features,
      isotonic calibration and a cost-sensitive maintenance policy. Selected model:
      <strong>{feed['model'].replace('_', ' ')}</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview, eda, evaluation, simulator, live = st.tabs(
    ["Overview", "Sensor EDA", "Evaluation", "Policy simulator", "Live machine"]
)

with overview:
    columns = st.columns(5)
    columns[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    columns[1].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    columns[2].metric("Failure recall", f"{metrics['recall']:.1%}")
    columns[3].metric("Inspection precision", f"{metrics['precision']:.1%}")
    columns[4].metric("Cost / 1,000", f"{policy['cost_per_1000']:.0f}")

    st.write(
        f"At a threshold of **{metrics['threshold']:.2f}**, the model flags **{metrics['flagged_rate']:.1%}** "
        f"of machines, catches **{metrics['recall']:.1%}** of failures and misses "
        f"**{metrics['missed_failures']}** of {metrics['missed_failures'] + metrics['confusion_matrix']['tp']} failures in the test set. "
        f"Each missed failure is weighted **{policy['fn_cost']:.0f}x** more heavily than a false alarm."
    )

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Risk tiers")
        view = tiers.copy()
        view["failure_rate"] = view["failure_rate"].map(lambda value: f"{value:.1%}")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("Calibrated tiers separate routine operation from stop-and-service cases.")
    with right:
        st.subheader("Dataset")
        st.write(
            f"{summary['rows']:,} observations · {summary['failures']:,} failures · "
            f"{summary['failure_rate']:.2%} failure rate"
        )
        st.write(
            "Features include air/process temperature, rotational speed, torque and tool wear, "
            "plus thermal, power and strain interactions."
        )
    st.image(str(FIGURES / "13_risk_tiers.png"), caption="Observed failure rate by risk tier")

with eda:
    st.subheader("Sensor behaviour")
    stats = pd.DataFrame(
        {
            "metric": ["Rows", "Failures", "Failure rate", "Mean tool wear", "Mean torque", "Mean RPM"],
            "value": [
                f"{summary['rows']:,}",
                f"{summary['failures']:,}",
                f"{summary['failure_rate']:.2%}",
                f"{summary['mean_tool_wear']:.1f} min",
                f"{summary['mean_torque']:.1f} Nm",
                f"{summary['mean_rpm']:.0f}",
            ],
        }
    )
    st.dataframe(stats, use_container_width=True, hide_index=True)
    first, second = st.columns(2)
    first.image(str(FIGURES / "01_failure_balance.png"), caption="Failure imbalance")
    second.image(str(FIGURES / "02_failure_modes.png"), caption="Failure-mode breakdown")
    st.image(str(FIGURES / "03_sensor_distributions.png"), caption="Sensor distributions by outcome")
    st.image(str(FIGURES / "04_correlation_heatmap.png"), caption="Sensor correlation structure")

with evaluation:
    st.subheader("Model comparison")
    view = comparison.copy()
    for column in ["pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std"]:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("Held-out test metrics")
    metric_table = pd.DataFrame(
        {
            "metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "Flagged rate", "Missed failures"],
            "value": [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['pr_auc']:.4f}",
                f"{metrics['brier']:.4f}",
                f"{metrics['flagged_rate']:.4f}",
                str(metrics["missed_failures"]),
            ],
        }
    )
    st.dataframe(metric_table, use_container_width=True, hide_index=True)

    first, second = st.columns(2)
    first.image(str(FIGURES / "06_roc_pr.png"), caption="ROC and precision-recall curves")
    second.image(str(FIGURES / "07_calibration.png"), caption="Reliability diagram")
    third, fourth = st.columns(2)
    third.image(str(FIGURES / "08_confusion_matrix.png"), caption="Confusion matrix")
    fourth.image(str(FIGURES / "09_policy_curve.png"), caption="Cost-sensitive policy")
    st.image(str(FIGURES / "10_feature_importance.png"), caption="Permutation importance")

    st.subheader("Failure-mode coverage")
    mode_view = modes.copy()
    mode_view["recall"] = mode_view["recall"].map(lambda value: f"{value:.1%}")
    st.dataframe(mode_view, use_container_width=True, hide_index=True)
    st.image(str(FIGURES / "11_failure_mode_recall.png"), caption="Recall by labelled failure mode")

    st.subheader("Operational slices")
    st.dataframe(slices, use_container_width=True, hide_index=True)

with simulator:
    st.subheader("Maintenance policy simulator")
    st.write(
        "Move the threshold to see the trade-off between missed failures and unnecessary inspections. "
        "Costs are illustrative: a missed failure is 15x costlier than a false alarm."
    )
    threshold = st.slider("Probability threshold", min_value=0.01, max_value=0.99, value=float(policy["threshold"]), step=0.01)
    row = policy_curve.iloc[(policy_curve["threshold"] - threshold).abs().argsort()[:1]].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Recall", f"{row['recall']:.1%}")
    columns[1].metric("Precision", f"{row['precision']:.1%}")
    columns[2].metric("Flagged", f"{row['flagged_rate']:.1%}")
    columns[3].metric("Cost / 1,000", f"{row['cost_per_1000']:.0f}")
    st.write(
        f"At threshold **{row['threshold']:.2f}**, **{int(row['missed_failures'])}** failures are missed "
        f"and **{int(row['false_alarms'])}** unnecessary inspections are triggered on the test set."
    )
    st.line_chart(policy_curve.set_index("threshold")[["recall", "precision", "flagged_rate"]], height=260)
    st.line_chart(policy_curve.set_index("threshold")[["cost_per_1000"]], height=220)

with live:
    st.subheader("Live machine assessment")
    st.caption(
        "Adjust the six sensor inputs; other fields use portfolio medians. The score comes from the "
        "calibrated model in artifacts/maintenance_model.joblib."
    )
    first, second, third = st.columns(3)
    product_type = first.selectbox("Product type", ["L", "M", "H"], index=1)
    air_temp = second.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=298.0, step=0.1)
    process_temp = third.number_input("Process temperature (K)", min_value=250.0, max_value=400.0, value=309.0, step=0.1)
    fourth, fifth, sixth = st.columns(3)
    rpm = fourth.number_input("Rotational speed (rpm)", min_value=0, max_value=4000, value=1500, step=10)
    torque = fifth.number_input("Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=0.5)
    wear = sixth.number_input("Tool wear (min)", min_value=0, max_value=300, value=80, step=1)

    overrides = {
        "type": product_type,
        "air_temp_k": float(air_temp),
        "process_temp_k": float(process_temp),
        "rpm": float(rpm),
        "torque_nm": float(torque),
        "tool_wear_min": float(wear),
    }
    probability = predict_probability(model, template, overrides)
    tier = risk_tier(probability)
    columns = st.columns(3)
    columns[0].metric("Failure probability", f"{probability:.1%}")
    columns[1].metric("Risk tier", tier)
    columns[2].metric("Recommended action", action_for_tier(tier))
    st.progress(min(max(probability, 0.0), 1.0))

    scenario = []
    for value in [40, 60, 80, 100, 140, 180, 200, 220, 240]:
        scenario.append(
            {
                "tool_wear_min": value,
                "probability": predict_probability(model, template, {**overrides, "tool_wear_min": float(value)}),
            }
        )
    st.line_chart(pd.DataFrame(scenario).set_index("tool_wear_min"), height=220)
    st.caption("Score sensitivity to tool wear, holding the other five inputs fixed.")

    st.warning(
        "Decision support only. A production system needs failure-mode diagnosis, human confirmation, "
        "sensor-drift monitoring and a cost model built from real maintenance records."
    )
