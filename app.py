import streamlit as st
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import io
from collections import defaultdict

st.set_page_config(page_title="Agent Fraud Detector", layout="centered")
st.title("🕵️ Agent Fraud & Suspicious Task Detector")

uploaded_files = st.file_uploader("Upload Excel Files (.xlsx or .xlsb)", type=["xlsx", "xlsb"], accept_multiple_files=True)

# Unified column mapping (case-insensitive)
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


def merge_files(files):
    merged_df = pd.DataFrame()
    for file in files:
        df = read_file(file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df


def run_checks(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'agent_id'])
    df = df.sort_values(by=['agent_id', 'timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['is_suspicious'] = ''
    df['flag_reason'] = ''

    # Rule 1 & 2: short time or large distance
    for (agent, date), group in df.groupby(['agent_id', 'date']):
        group = group.sort_values('timestamp')
        prev_row = None
        for idx, row in group.iterrows():
            reasons = []
            if prev_row is not None:
                time_diff = (row['timestamp'] - prev_row['timestamp']).total_seconds() / 60
                if time_diff < 5:
                    reasons.append(f"Short time gap: {time_diff:.1f} mins")

                if 'latitude' in row and 'longitude' in row and pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    dist = geodesic(
                        (prev_row.get('latitude', 0), prev_row.get('longitude', 0)),
                        (row.get('latitude', 0), row.get('longitude', 0))
                    ).km
                    if dist > 30 and time_diff < 60:
                        reasons.append(f"Large distance: {dist:.1f} km in {time_diff:.1f} mins")

            if reasons:
                df.at[idx, 'is_suspicious'] = 'Yes'
                df.at[idx, 'flag_reason'] = '; '.join(reasons)
            prev_row = row

    # Rule 3: top volume agents
    volume_by_agent = df.groupby(['agent_id', 'date']).size()
    threshold = volume_by_agent.quantile(0.95)
    for (agent, date), count in volume_by_agent.items():
        if count >= threshold:
            df.loc[(df['agent_id'] == agent) & (df['date'] == date), 'flag_reason'] += '; High task volume'
            df.loc[(df['agent_id'] == agent) & (df['date'] == date), 'is_suspicious'] = 'Yes'

    # Rule 4: same MerchantExternalId multiple times
    if 'merchant_id' in df.columns:
        merchant_counts = df['merchant_id'].value_counts()
        flagged_merchants = merchant_counts[merchant_counts > 3].index
        df.loc[df['merchant_id'].isin(flagged_merchants), 'flag_reason'] += '; Repeated Merchant ID'
        df.loc[df['merchant_id'].isin(flagged_merchants), 'is_suspicious'] = 'Yes'

    # Rule 5: same phone number for different merchants
    if 'mobile_no' in df.columns and 'merchant_id' in df.columns:
        phone_merchant = df.groupby('mobile_no')['merchant_id'].nunique()
        suspicious_phones = phone_merchant[phone_merchant > 1].index
        df.loc[df['mobile_no'].isin(suspicious_phones), 'flag_reason'] += '; Same phone, multiple merchants'
        df.loc[df['mobile_no'].isin(suspicious_phones), 'is_suspicious'] = 'Yes'

    # Rule 6: low internal reporting vs high lead
    if 'entry_status' in df.columns:
        internal = df[df['entry_status'].str.contains("yes", case=False, na=False)]
        external = df.groupby('agent_id').size()
        internal_count = internal.groupby('agent_id').size()
        for agent in external.index:
            ext = external[agent]
            intr = internal_count.get(agent, 0)
            if intr < 2 and ext >= 10:
                df.loc[df['agent_id'] == agent, 'flag_reason'] += '; Low internal, high leads'
                df.loc[df['agent_id'] == agent, 'is_suspicious'] = 'Yes'

    return df


if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")
    merged_df = merge_files(uploaded_files)
    st.write("Merged Preview:", merged_df.head())

    flagged_df = run_checks(merged_df)
    suspicious = flagged_df[flagged_df['is_suspicious'] == 'Yes']

    st.markdown(f"### 🚨 {len(suspicious)} Suspicious Entries Found")
    st.dataframe(suspicious)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        flagged_df.to_excel(writer, index=False, sheet_name='All Data')
        suspicious.to_excel(writer, index=False, sheet_name='Suspicious Only')

    st.download_button("📥 Download Flagged Report", data=output.getvalue(),
                       file_name="suspicious_task_report.xlsx")

