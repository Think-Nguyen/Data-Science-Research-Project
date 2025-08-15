import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from io import BytesIO

# PAGE SETTINGS
st.set_page_config(
    page_title='FAERS Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# CSS STYLES
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    footer, header { visibility: hidden; }
    .stButton>button {
        padding: 0.1rem 0.5rem;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
    .stDataFrame { margin-bottom: 0.5rem; }
    .filter-box {
        border: 1px solid green;
        padding: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# DATA LOADING 

LIST_COLUMNS = ['SOC', 'pt', 'outc_cod']
EXCLUDE_COLUMNS = ['caseid', 'caseversion', 'seq', 'primaryid']

@st.cache_data
def load_sample_csv(path: str):
    df = pd.read_csv(path)
    for col in LIST_COLUMNS:
        if col in df.columns and pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    return df

@st.cache_data
def load_uploaded_csv(file_bytes: bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    for col in LIST_COLUMNS:
        if col in df.columns and pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    return df

# TOP: Dataset title + uploader
st.header('Dataset')
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'], help='Upload your FAERS csv file.')

def _reset_ui_state():
    # wipe prior widget selections
    to_clear = [k for k in st.session_state.keys()
                if k.startswith('filter_') or k in {
                    'stats_column', 'chart_type', 'hist_col',
                    'ycol_line', 'xcol_box', 'ycol_box', 'heatmap_cols'
                }]
    for k in to_clear:
        st.session_state.pop(k, None)
    st.session_state.filters = {}
    # clear cached dataframes so old data can't leak
    st.cache_data.clear()

# If file removed: show nothing else and clear state
if uploaded_file is None:
    _reset_ui_state()
    st.session_state.pop('data_sig', None)
    st.stop()

# Detect a new file and reset UI
file_bytes = uploaded_file.getvalue()
data_sig = (uploaded_file.name, len(file_bytes))  # simple signature
if st.session_state.get('data_sig') != data_sig:
    _reset_ui_state()
    st.session_state['data_sig'] = data_sig

# Now safe to load and render the rest
df_original = load_uploaded_csv(file_bytes)
st.caption(f'Using uploaded file: **{uploaded_file.name}**')
df = df_original.copy()


# HELPERS
def apply_filters(df, filters, until_col=None):
    df_filtered = df.copy()
    for col, val in filters.items():
        if until_col and col == until_col:
            break
        if val is not None and val != '':
            if col in LIST_COLUMNS:
                df_filtered = df_filtered[df_filtered[col].apply(lambda lst: val in lst if isinstance(lst, list) else False)]
            else:
                df_filtered = df_filtered[df_filtered[col] == val]
    return df_filtered

def get_valid_columns(numeric=None):
    cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    if numeric is True:
        return [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    elif numeric is False:
        return [col for col in cols if not pd.api.types.is_numeric_dtype(df[col])]
    return cols

def download_plot(fig, filename='plot.png'):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label='📥 Download Graph',
        data=buf,
        file_name=filename,
        mime='image/png'
    )


# SIDEBAR FILTERS
with st.sidebar:
    st.title('Choose Filters')
    st.markdown('Filters apply cumulatively to the dataset.')

    for idx, col in enumerate(df_original.columns):
        df_temp = apply_filters(df_original, st.session_state.filters, until_col=col)
        if col in LIST_COLUMNS:
            try:
                options = [''] + sorted(list(df_temp[col].explode().dropna().unique()))
            except Exception:
                options = ['']
        else:
            options = [''] + sorted(list(df_temp[col].dropna().unique()))
        selected_val = st.selectbox(f'{col}', options, key=f'filter_{col}')
        st.session_state.filters[col] = selected_val if selected_val else None

# APPLY FINAL FILTER
df = apply_filters(df_original, st.session_state.filters)


# PAGE LAYOUT
left_col, right_col = st.columns([1.5, 1.5])

with left_col:
    st.subheader('Dataset Preview')
    st.dataframe(df, height=300)

    st.subheader('Top 10 Values')
    stats_col = st.selectbox('Choose a column:', get_valid_columns(), key='stats_column')

    if stats_col in LIST_COLUMNS:
        top_vals = df[stats_col].explode().dropna().value_counts().head(10).reset_index()
        top_vals.columns = ['Value', 'Count']
    else:
        top_vals = df[stats_col].value_counts().head(10).reset_index()
        top_vals.columns = ['Value', 'Count']

    st.dataframe(top_vals, height=250)

with right_col:
    st.subheader('Visualization')

    chart_type = st.selectbox('Graph type', ['Histogram', 'Line', 'Boxplot', 'Heatmap'], key='chart_type')
    data = df.copy()

    if chart_type == 'Histogram':
        col = st.selectbox('Select column for histogram', get_valid_columns(), key='hist_col')
        if col in LIST_COLUMNS:
            data = data.explode(col).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data[col].dropna(), bins=30, edgecolor='black')
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=90)
        plt.xticks(rotation=90, ha='center')
        st.pyplot(fig)
        download_plot(fig, filename=f"{col}_histogram.png")

    elif chart_type == 'Line':
        y_col = st.selectbox('Select categorical column for y-axis', get_valid_columns(numeric=False), key='ycol_line')
        data = df[['primaryid', 'rept_yr', y_col]].dropna()
        if y_col in LIST_COLUMNS:
            data = data.explode(y_col).reset_index(drop=True)

        grouped = data.groupby(['rept_yr', y_col]).size().reset_index(name='count')
        pivot_df = grouped.pivot(index='rept_yr', columns=y_col, values='count').fillna(0)

        if pivot_df.shape[1] > 30:
            pivot_df = pivot_df[pivot_df.sum().sort_values(ascending=False).head(30).index]

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_df.plot(kind='line', marker='o', ax=ax)
        ax.set_title(f'Line graph of {y_col} over years')
        ax.set_xlabel('Report Year')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=90)
        plt.xticks(rotation=90, ha='center')
        ax.legend(title=y_col, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
        st.pyplot(fig)
        download_plot(fig, filename=f"{y_col}_over_years_line.png")

    elif chart_type == 'Boxplot':
        x_col = st.selectbox('Select categorical x-axis', get_valid_columns(numeric=False), key='xcol_box')
        y_col = st.selectbox('Select numeric y-axis', get_valid_columns(numeric=True), key='ycol_box')

        data = df[[x_col, y_col]].dropna()
        if x_col in LIST_COLUMNS:
            data = data.explode(x_col).reset_index(drop=True)

        if data[x_col].nunique() > 30:
            top_vals = data[x_col].value_counts().nlargest(30).index
            data = data[data[x_col].isin(top_vals)]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=x_col, y=y_col, data=data, ax=ax)
        ax.set_title(f'Boxplot of {y_col} by {x_col}')
        ax.tick_params(axis='x', rotation=90)
        plt.xticks(rotation=90, ha='center')
        st.pyplot(fig)
        download_plot(fig, filename=f"{y_col}_by_{x_col}_boxplot.png")

    elif chart_type == 'Heatmap':
        heatmap_cols = st.multiselect('Select columns for heatmap (2-7)', get_valid_columns(), key='heatmap_cols')

        if len(heatmap_cols) > 7:
            st.warning('⚠️ You have selected more than 7 columns. Please select at most 7.')
        elif len(heatmap_cols) >= 2:
            data = df[heatmap_cols].copy()
            for col in heatmap_cols:
                if col in LIST_COLUMNS:
                    data = data.explode(col).reset_index(drop=True)
                if not pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = pd.factorize(data[col])[0]

            corr = data.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Heatmap')
            ax.tick_params(axis='x', rotation=90)
            plt.xticks(rotation=90, ha='center')
            st.pyplot(fig)
            download_plot(fig, filename="correlation_heatmap.png")
        else:
            st.info('Please select at least two columns for the heatmap.')
