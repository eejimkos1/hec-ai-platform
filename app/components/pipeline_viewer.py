"""Pipeline viewer — reusable step indicator and data-flow explanation."""

import streamlit as st
from app.config import PIPELINE_STEPS, PIPELINE_STEPS_5


def render_pipeline_steps(current_step: int = 0, total_steps: int = 4):
    """Render the pipeline indicator showing which step user is viewing.

    For total_steps=4 (default): 0=Source Data, 1=Feature Engineering, 2=ML Model, 3=Results
    For total_steps=5: uses the 5-step pipeline from config (includes Analytics step)
    Highlights the current step with bold text and a coloured border.
    """
    if total_steps == 5:
        step_defs = [(s["icon"], s["label"].split(". ", 1)[-1], s["description"]) for s in PIPELINE_STEPS_5]
    else:
        step_defs = [
            ("📊", "Source Data", "Raw operational data"),
            ("⚙️", "Feature Engineering", "Computed ML features"),
            ("🤖", "ML Model", "Training & performance"),
            ("🎯", "Results", "Predictions & optimization"),
        ]

    cols = st.columns(len(step_defs))

    for i, (icon, label, desc) in enumerate(step_defs):
        with cols[i]:
            is_current = i == current_step
            border_color = "#1B9AAA" if is_current else "#E0E4E8"
            bg_color = "#E8F6F7" if is_current else "#F5F7FA"
            font_weight = "700" if is_current else "400"
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {border_color};
                    border-radius: 8px;
                    padding: 12px 8px;
                    text-align: center;
                    background: {bg_color};
                    min-height: 90px;
                ">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <div style="font-weight:{font_weight}; margin:4px 0 2px; font-size:0.9rem; color:#0D1B2A;">{label}</div>
                    <div style="font-size:0.75rem; color:#546E7A;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Connector arrows between steps (outside the column loop)
    arrow_cols = st.columns(len(step_defs))
    for i, col in enumerate(arrow_cols):
        if i < len(step_defs) - 1:
            with col:
                st.markdown(
                    "<div style='text-align:right; font-size:1.2rem; color:#546E7A; margin-top:-6px;'>→</div>",
                    unsafe_allow_html=True,
                )


def render_data_flow_explanation(
    source_tables: list,
    features_added: list,
    model_name: str,
    result_description: str,
):
    """Show the logical flow: what data feeds what.

    Args:
        source_tables: list of (table_name, row_count, description) tuples
        features_added: list of (feature_name, formula/description) tuples
        model_name: e.g. "XGBoost Yield Predictor"
        result_description: e.g. "Optimized separation parameters"
    """
    with st.expander(
        "📋 Pipeline Logic — How we get from raw data to results", expanded=False
    ):
        col_src, col_feat, col_mod, col_res = st.columns(4)

        with col_src:
            st.markdown("**📊 Source Tables**")
            for name, rows, desc in source_tables:
                st.markdown(f"- `{name}`  \n  {rows:,} rows  \n  _{desc}_")

        with col_feat:
            st.markdown("**⚙️ Engineered Features**")
            for feat, formula in features_added:
                st.markdown(f"- `{feat}`  \n  = {formula}")

        with col_mod:
            st.markdown("**🤖 Model**")
            st.markdown(model_name)

        with col_res:
            st.markdown("**🎯 Result**")
            st.markdown(result_description)
