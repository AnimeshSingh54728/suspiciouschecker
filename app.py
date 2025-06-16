import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from collections import defaultdict
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Agent Fraud Detector", layout="centered")
st.title("🕵️ Agent Fraud & Suspicious Task Detector-NA")

@st.cache_data(show_spinner=False)
def merge_files(files):
    merged_df = pd.DataFrame()
    for file in files:
        df = read_file(file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df

uploaded_files = st.file_uploader("Upload Excel Files (.xlsx or .xlsb)", type=["xlsx", "xlsb"], accept_multiple_files=True)

HEADER_MAPPING = {
    'timestamp': ['VerificationTimestamp', 'Gpay SpSaleDate'],
    'pincode': ['Pincode'],
    'mobile_no': ['Mobile No'],
    'agent_id': ['AgentEmail'],
    'entry_status': ['NTA Entry Status'],
    'lead_status': ['RV Status', 'Google Remarks', 'Sound Pod Sales Status'],
    'latitude': ['Latitude', 'Lattitude'],
    'longitude': ['Longitude'],
    'merchant_id': ['MerchantExternalId']
}

REVERSE_MAPPING = {col.lower(): unified for unified, cols in HEADER_MAPPING.items() for col in cols}

def normalize_columns(df):
    col_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in REVERSE_MAPPING:
            col_map[col] = REVERSE_MAPPING[key]
    df = df.rename(columns=col_map)
    return df

def read_file(file):
    if file.name.endswith(".xlsb"):
        df = pd.read_excel(file, engine="pyxlsb")
    else:
        df = pd.read_excel(file)
    return normalize_columns(df)

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def run_checks(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'agent_id'])
    df = df.sort_values(by=['agent_id', 'timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['is_suspicious'] = ''
    df['flag_reason'] = ''

    df['prev_time'] = df.groupby(['agent_id', 'date'])['timestamp'].shift(1)
    df['prev_lat'] = df.groupby(['agent_id', 'date'])['latitude'].shift(1)
    df['prev_lon'] = df.groupby(['agent_id', 'date'])['longitude'].shift(1)

    df['time_diff'] = (df['timestamp'] - df['prev_time']).dt.total_seconds() / 60
    valid_coords = df[['latitude', 'longitude', 'prev_lat', 'prev_lon']].notna().all(axis=1)
    df['dist'] = np.nan
    df.loc[valid_coords, 'dist'] = haversine_np(
        df.loc[valid_coords, 'latitude'],
        df.loc[valid_coords, 'longitude'],
        df.loc[valid_coords, 'prev_lat'],
        df.loc[valid_coords, 'prev_lon']
    )

    mask_short_time = df['time_diff'] < 5
    df.loc[mask_short_time, 'is_suspicious'] = 'Yes'
    df.loc[mask_short_time, 'flag_reason'] += 'Short time gap; '

    mask_large_dist = (df['dist'] > 30) & (df['time_diff'] < 60)
    df.loc[mask_large_dist, 'is_suspicious'] = 'Yes'
    df.loc[mask_large_dist, 'flag_reason'] += 'Large distance gap; '

    volume_by_agent = df.groupby(['agent_id', 'date']).size()
    threshold = volume_by_agent.quantile(0.95)
    flagged_agents = volume_by_agent[volume_by_agent >= threshold].index
    high_task_mask = df.set_index(['agent_id', 'date']).index.isin(flagged_agents)
    df.loc[high_task_mask, 'is_suspicious'] = 'Yes'
    df.loc[high_task_mask, 'flag_reason'] += 'High task volume; '

    if 'merchant_id' in df.columns:
        merchant_counts = df['merchant_id'].value_counts()
        repeated_merchants = merchant_counts[merchant_counts > 3].index
        repeat_mask = df['merchant_id'].isin(repeated_merchants)
        df.loc[repeat_mask, 'is_suspicious'] = 'Yes'
        df.loc[repeat_mask, 'flag_reason'] += 'Repeated Merchant ID; '

    if 'mobile_no' in df.columns and 'merchant_id' in df.columns:
        phone_merchant_counts = df.groupby('mobile_no')['merchant_id'].nunique()
        suspicious_phones = phone_merchant_counts[phone_merchant_counts > 1].index
        phone_mask = df['mobile_no'].isin(suspicious_phones)
        df.loc[phone_mask, 'is_suspicious'] = 'Yes'
        df.loc[phone_mask, 'flag_reason'] += 'Same phone, multiple merchants; '

    if 'entry_status' in df.columns:
        external_counts = df.groupby('agent_id').size()
        internal_counts = df[df['entry_status'].str.contains("yes", case=False, na=False)].groupby('agent_id').size()
        under_reporting = external_counts[external_counts >= 10].index.difference(internal_counts[internal_counts >= 2].index)
        low_internal_mask = df['agent_id'].isin(under_reporting)
        df.loc[low_internal_mask, 'is_suspicious'] = 'Yes'
        df.loc[low_internal_mask, 'flag_reason'] += 'Low internal, high leads; '

    # New: Clustering agents with similar behavior
    if 'latitude' in df.columns and 'longitude' in df.columns:
        cluster_data = df[['latitude', 'longitude']].dropna()
        if not cluster_data.empty:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cluster_data)
            kmeans = KMeans(n_clusters=min(5, len(cluster_data)), random_state=0, n_init='auto')
            df.loc[cluster_data.index, 'cluster'] = kmeans.fit_predict(X_scaled)

    # New: Back-to-back entries with identical lat/lon but different agents
    df['next_agent'] = df['agent_id'].shift(-1)
    df['next_lat'] = df['latitude'].shift(-1)
    df['next_lon'] = df['longitude'].shift(-1)
    match_mask = (df['agent_id'] != df['next_agent']) & \
                 (df['latitude'] == df['next_lat']) & \
                 (df['longitude'] == df['next_lon'])
    df.loc[match_mask, 'is_suspicious'] = 'Yes'
    df.loc[match_mask, 'flag_reason'] += 'Same location diff agent; '

    return df

def show_charts(df):
    st.markdown("### 📊 Summary Charts")

    if 'agent_id' in df.columns:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('agent_id:N', title='Agent ID'),
            y=alt.Y('count():Q', title='Suspicious Leads'),
            color=alt.value('crimson'),
            tooltip=['agent_id', 'count()']
        ).transform_filter(
            alt.datum.is_suspicious == 'Yes'
        ).properties(height=400, width=700, title="Suspicious Leads per Agent")
        st.altair_chart(chart, use_container_width=True)

    if 'flag_reason' in df.columns:
        reasons = df[df['is_suspicious'] == 'Yes']['flag_reason'].str.split(';', expand=True).stack()
        reasons = reasons.str.strip().value_counts().reset_index()
        reasons.columns = ['Reason', 'Count']
        chart2 = alt.Chart(reasons).mark_bar().encode(
            x=alt.X('Reason:N', sort='-y'),
            y=alt.Y('Count:Q'),
            color=alt.value('orange'),
            tooltip=['Reason', 'Count']
        ).properties(width=700, height=300, title="Breakdown of Flag Reasons")
        st.altair_chart(chart2, use_container_width=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")
    merged_df = merge_files(uploaded_files)
    st.write("Merged Preview:", merged_df.head())

    with st.spinner("Running fraud detection checks..."):
        flagged_df = run_checks(merged_df)

    suspicious = flagged_df[flagged_df['is_suspicious'] == 'Yes']

    st.markdown(f"### 🚨 {len(suspicious)} Suspicious Entries Found")
    st.dataframe(suspicious)

    show_charts(flagged_df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        flagged_df.to_excel(writer, index=False, sheet_name='All Data')
        suspicious.to_excel(writer, index=False, sheet_name='Suspicious Only')

    st.download_button("🗕️ Download Flagged Report", data=output.getvalue(),
                       file_name="suspicious_task_report.xlsx")
