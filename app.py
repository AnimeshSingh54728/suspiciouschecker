import streamlit as st
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import io

st.set_page_config(page_title="Fake Task Detector", layout="centered")
st.title("🚨 Fake/Suspicious Task Detector")

uploaded_file = st.file_uploader("Upload Excel File (.xlsx or .xlsb)", type=["xlsx", "xlsb"])

def read_excel(file):
    if file.name.endswith(".xlsb"):
        return pd.read_excel(file, engine="pyxlsb")
    return pd.read_excel(file)

def run_checks(df):
    df['is_suspicious'] = ''
    df['flag_reason'] = ''

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['agent_id', 'timestamp'])

    for agent, group in df.groupby("agent_id"):
        prev_row = None
        for idx, row in group.iterrows():
            flags = []
            if prev_row is not None:
                time_diff = (row['timestamp'] - prev_row['timestamp']).total_seconds() / 60
                distance = geodesic(
                    (prev_row['latitude'], prev_row['longitude']),
                    (row['latitude'], row['longitude'])
                ).km

                if distance > 5 and time_diff < 10:
                    flags.append(f'Far location in {time_diff:.1f} min ({distance:.1f} km)')
                if time_diff < 5:
                    flags.append(f'Tasks in <5 min ({time_diff:.1f} min)')

            if flags:
                df.at[idx, 'is_suspicious'] = 'Yes'
                df.at[idx, 'flag_reason'] = '; '.join(flags)

            prev_row = row

        if len(group) > 30:
            for idx in group.index:
                df.at[idx, 'is_suspicious'] = 'Yes'
                df.at[idx, 'flag_reason'] += '; High volume tasks'

    return df

if uploaded_file:
    df = read_excel(uploaded_file)
    required_cols = ['agent_id', 'timestamp', 'latitude', 'longitude']

    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing columns. Required: {required_cols}")
    else:
        st.success("✅ File uploaded and ready for checks.")
        df_checked = run_checks(df)

        suspicious_df = df_checked[df_checked['is_suspicious'] == 'Yes']
        st.write(f"🔎 Found {len(suspicious_df)} suspicious entries.")
        st.dataframe(suspicious_df)

        # Downloadable Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_checked.to_excel(writer, index=False, sheet_name='Flagged Data')
            suspicious_df.to_excel(writer, index=False, sheet_name='Suspicious Only')
        st.download_button("📥 Download Flagged Report", data=output.getvalue(),
                           file_name="suspicious_tasks_report.xlsx")
