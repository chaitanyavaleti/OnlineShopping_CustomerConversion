import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.title("Customer Analytics Dashboard")
st.sidebar.header("Upload or Input Data")

# ----- Weighted Conversion -----
def create_weighted_converted(df, k=6):
    df_copy = df.copy()
    drop_cols = ["order", "price"]
    df_features = df_copy.drop(columns=drop_cols, errors='ignore')
    
    df_copy["_temp_target"] = ((df_copy.get("order", 0) > 0) & (df_copy.get("price", 0) > 0)).astype(int)
    y = df_copy["_temp_target"]
    X_num = df_features.select_dtypes(include=[np.number]).drop(columns=["_temp_target"], errors="ignore")
    
    mi = mutual_info_classif(X_num.fillna(0), y)
    mi = pd.Series(mi, index=X_num.columns)
    top_features = mi.sort_values(ascending=False).head(k).index.tolist()
    weights = mi[top_features] / mi[top_features].sum()
    df_copy['conversion_score'] = (df_copy[top_features] * weights).sum(axis=1)
    df_copy['converted'] = (df_copy['conversion_score'] > df_copy['conversion_score'].median()).astype(int)
    df_copy.drop(columns=['_temp_target'], inplace=True)
    return df_copy, top_features, weights

# ----- Align dataframe with MLflow model signature -----
def align_with_signature(model, df):
    input_schema = model.metadata.get_input_schema()
    expected_cols = [f.name for f in input_schema.inputs]
    df_aligned = df.reindex(columns=expected_cols, fill_value=0)
    return df_aligned

# ----- File Upload -----
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_orig = pd.read_csv(uploaded_file)
    st.write("✅ Raw Uploaded Data:")
    st.dataframe(df_orig.head())

    # Dynamically create 'converted' if missing
    if 'converted' not in df_orig.columns:
        df_orig, top_features, feature_weights = create_weighted_converted(df_orig, k=6)
        st.success(f"'converted' column created dynamically using top features: {top_features}")

else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ----- Select Task -----
task = st.sidebar.selectbox(
    "Select Task",
    ["Classification: Conversion Prediction",
     "Regression: Revenue Estimation",
     "Clustering: Customer Segmentation"]
)

client = MlflowClient()
model = None

# ----- Load MLflow model -----
if "Classification" in task:
    model_name = "Best_classification_Model"
elif "Regression" in task:
    model_name = "Best_regression_Model"
else:
    model_name = "Best_clustering_Model"

try:
    versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in versions if v.tags.get("stage") == "Production"]
    if not prod_versions:
        st.error("No Production model found!")
    latest_prod_version = max(prod_versions, key=lambda v: int(v.version))
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_prod_version.version}")
    st.success(f"Loaded model: {model_name}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----- Predictions -----
if model:
    try:
        X_input = align_with_signature(model, df_orig)
        if "Classification" in task:
            df_orig['predicted_conversion'] = model.predict(X_input)
            st.write(df_orig[['predicted_conversion']])
            st.bar_chart(df_orig['predicted_conversion'].value_counts())
        elif "Regression" in task:
            df_orig['predicted_revenue'] = model.predict(X_input)
            st.write(df_orig[['predicted_revenue']])
            st.bar_chart(df_orig['predicted_revenue'])
        else:  # Clustering
            df_orig['cluster'] = model.predict(X_input)
            st.write(df_orig[['cluster']])
            st.bar_chart(df_orig['cluster'].value_counts())
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----- Exploratory Data Analysis -----
st.subheader("Exploratory Data Analysis")
numeric_cols = df_orig.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    col = st.selectbox("Select numeric column for visualization", numeric_cols)
    st.write("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df_orig[col], kde=True, ax=ax)
    st.pyplot(fig)
    st.write("Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df_orig[col], ax=ax)
    st.pyplot(fig)
