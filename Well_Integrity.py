"""
Well Integrity Visualization Tool
University of Kirkuk Style Update
"""

import streamlit as st
import lasio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import os

st.set_page_config(page_title="Well Integrity Visualizer", layout="wide")

def find_image_path(keywords):
    try:
        files = os.listdir('.')
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                for k in keywords:
                    if k.lower() in f.lower(): return f
    except: pass
    return None

# --- Helper: Generate Dummy Data ---
def generate_dummy_data():
    depths = np.arange(1000, 1200, 0.1)
    n_samples = len(depths)
    cols = {'DEPT': depths, 'GR': np.random.normal(50, 5, n_samples)}
    base_radius = 2.0
    for i in range(1, 25):
        finger_data = np.random.normal(base_radius, 0.01, n_samples)
        pits = np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])
        finger_data += pits * np.random.uniform(0.1, 0.3, n_samples)
        collar_idx = (depths % 30) < 0.5
        finger_data[collar_idx] -= 0.1
        cols[f'FING{i:02d}'] = finger_data
    return pd.DataFrame(cols).set_index('DEPT')

# --- Core Logic: Plotting Function ---
def plot_well_integrity(dataframe: pd.DataFrame, fingers: list, height=1500, width=1000, depth_range=None):
    if depth_range:
        df_plot = dataframe[(dataframe.index >= depth_range[0]) & (dataframe.index <= depth_range[1])].copy()
    else:
        df_plot = dataframe.copy()

    if df_plot.empty: return go.Figure()

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Gamma Ray", "Statistics", "Finger Calipers", "2D Heatmap"),
        horizontal_spacing=0.02, shared_yaxes=True,
        column_widths=[0.1, 0.15, 0.35, 0.4]
    )
    depth = df_plot.index

    if "GR" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot["GR"], y=depth, line=dict(color="green", width=1), name="GR"), row=1, col=1)
        fig.update_xaxes(title_text="GR (API)", row=1, col=1)
    else:
        fig.add_annotation(text="No GR", xref="x1", yref="y1", x=0.5, y=depth.mean(), showarrow=False)

    if fingers:
        min_curve = df_plot[fingers].min(axis=1)
        max_curve = df_plot[fingers].max(axis=1)
        avg_curve = df_plot[fingers].mean(axis=1)
        fig.add_trace(go.Scatter(x=min_curve, y=depth, name="Min", line=dict(color="orange", width=1, dash='dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=max_curve, y=depth, name="Max", line=dict(color="red", width=1, dash='dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=avg_curve, y=depth, name="Average", line=dict(color="blue", width=1.5)), row=1, col=2)
        fig.update_xaxes(title_text="Radius/Diameter", row=1, col=2)

    if fingers:
        data_min = df_plot[fingers].min().min()
        data_max = df_plot[fingers].max().max()
        offset_step = (data_max - data_min) * 0.2
        for i, fing_col in enumerate(fingers):
            x_values = df_plot[fing_col] + (i * offset_step)
            fig.add_trace(go.Scattergl(x=x_values, y=depth, mode='lines', line=dict(color='royalblue', width=0.8), showlegend=False, name=fing_col, hoverinfo='skip'), row=1, col=3)
        fig.update_xaxes(title_text="Individual Fingers", showticklabels=False, row=1, col=3)

    if fingers:
        heatmap_data = df_plot[fingers].values
        fig.add_trace(go.Heatmap(z=heatmap_data, x=fingers, y=depth, colorscale='Turbo', colorbar=dict(title="Amplitude", x=1.02)), row=1, col=4)
        fig.update_xaxes(title_text="Finger Index", row=1, col=4)

    fig.update_layout(height=height, width=width, title_text="Well Integrity Visualization", template="plotly_white", hovermode="y unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(autorange="reversed", matches='y')
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
    return fig

@st.cache_data
def load_las_data(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()
        string_data = bytes_data.decode("utf-8")
        las = lasio.read(io.StringIO(string_data))
        return las.df()
    except: return None

def main():
    # ---------------------------------------------------------
    # 🛠️ HEADER CONFIGURATION - EDIT EVERYTHING HERE 🛠️
    # ---------------------------------------------------------
    CONFIG = {
        # --- Text Content ---
        "TITLE": "Well Integrity Visualization Tool",
        "SUBTITLE": "University of Kirkuk - College of Engineering - Petroleum Department",
        "DEVELOPERS": "Developed by: Bilal Rabah & Omar Yilmaz",
        "SUPERVISOR": "Supervised by: Mohammed Adel",
        "DATE": "Date: November 2025",
        "ICON": "",

        # --- Visual Colors (Hex Codes) ---
        "BG_GRADIENT_1": "#1F4E78",      # Dark Blue
        "BG_GRADIENT_2": "#2c5f8d",      # Lighter Blue
        "TEXT_COLOR": "#ffffff",         # White

        # --- Image Search Keywords ---
        "LEFT_LOGO_QUERY": ['eng', 'logo', 'triangle'],
        "RIGHT_LOGO_QUERY": ['anniversary', '22', 'right', 'a.png']
    }
    # ---------------------------------------------------------

    # --- DYNAMIC CSS INJECTION ---
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');

        .blue-header-card {{
            background: linear-gradient(135deg, {CONFIG['BG_GRADIENT_1']} 0%, {CONFIG['BG_GRADIENT_2']} 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            color: {CONFIG['TEXT_COLOR']};
            margin-bottom: 1rem;
        }}
        
        .header-title {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            color: {CONFIG['TEXT_COLOR']};
        }}
        
        .header-subtitle {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 1.1rem;
            font-weight: 500;
            margin-top: 0.5rem;
            color: #e0e0e0;
        }}
        
        .header-dev {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 0.85rem;
            font-style: italic;
            margin-top: 1rem;
            color: #b0c4de;
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 0.5rem;
        }}
        
        /* Info Box */
        .info-box {{
            background-color: #ffffff;
            border-left: 5px solid {CONFIG['BG_GRADIENT_1']};
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            color: #333333;
        }}
        
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: {CONFIG['BG_GRADIENT_1']} !important;
            color: white !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    # --- HEADER SECTION (Dynamic) ---
    c1, c2, c3 = st.columns([1, 3, 1])

    with c1:
        eng_path = find_image_path(CONFIG['LEFT_LOGO_QUERY'])
        if eng_path: st.image(eng_path, use_container_width=True)

    with c2:
        st.markdown(f"""
            <div class="blue-header-card">
                <div style="font-size: 3rem; margin-bottom: 10px;">{CONFIG['ICON']}</div>
                <div class="header-title">{CONFIG['TITLE']}</div>
                <div class="header-subtitle">
                    {CONFIG['SUBTITLE']}
                </div>
                <div class="header-dev">
                    {CONFIG['DEVELOPERS']} | {CONFIG['SUPERVISOR']}<br>
                    {CONFIG['DATE']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        a_path = find_image_path(CONFIG['RIGHT_LOGO_QUERY'])
        if a_path: st.image(a_path, use_container_width=True)

    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload LAS File", type=["las"])

    use_example = False
    if not uploaded_file:
        if st.sidebar.button("📂 Load Example Data"): use_example = True

    df = None
    if uploaded_file:
        with st.spinner("Reading LAS file..."):
            df = load_las_data(uploaded_file)
            st.sidebar.success("File loaded!")
    elif use_example or st.session_state.get('use_example_data'):
        st.session_state['use_example_data'] = True
        with st.spinner("Generating synthetic data..."):
            df = generate_dummy_data()
            st.sidebar.info("Using Example Data")
            if st.sidebar.button("Clear Example"):
                st.session_state['use_example_data'] = False
                st.rerun()

    if df is not None:
        all_cols = df.columns.tolist()
        default_fingers = [c for c in all_cols if "FING" in c.upper() or "PAD" in c.upper()]
        fingers = st.sidebar.multiselect("Select Fingers/Pads", options=all_cols, default=default_fingers if default_fingers else all_cols[:min(24, len(all_cols))])
        min_d, max_d = float(df.index.min()), float(df.index.max())
        depth_range = st.sidebar.slider("Depth Range (MD)", min_value=min_d, max_value=max_d, value=(min_d, max_d), step=0.5)
        st.sidebar.subheader("Plot Dimensions")
        plot_h = st.sidebar.number_input("Height", 500, 5000, 1200, 100)
        plot_w = st.sidebar.number_input("Width", 500, 3000, 1400, 100)

        st.divider()
        if st.button("Generate Visualization", type="primary", use_container_width=True):
            if fingers:
                with st.spinner("Rendering complex well plot..."):
                    fig = plot_well_integrity(dataframe=df, fingers=fingers, height=plot_h, width=plot_w, depth_range=depth_range)
                    st.plotly_chart(fig, use_container_width=True)
            else: st.error("Please select at least one finger/pad curve.")

        with st.expander("View Raw Data"): st.dataframe(df.head())

    else:
        st.markdown('<div class="info-box"><strong>👈 Start here:</strong> Please upload a Multi-Finger Caliper (MFC) LAS file in the sidebar, or click <strong>"Load Example Data"</strong> to see a demo.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.info("**Tool Capabilities:**\n* Gamma Ray Track\n* Statistics (Min, Max, Avg)\n* Individual Fingers\n* 2D Heatmap")
        with c2: st.warning("**Instructions:**\n1. Upload .las file.\n2. Select caliper fingers.\n3. Adjust depth range.\n4. Click **Generate Visualization**.")

if __name__ == "__main__":

    main()
