import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
import warnings

st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.title("Customer Analytics Dashboard")

st.sidebar.header("Upload or Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
else:
    st.warning("Please upload a CSV file to proceed.")


task = st.sidebar.selectbox(
    "Select Task",
    ["Classification: Conversion Prediction",
     "Regression: Revenue Estimation",
     "Clustering: Customer Segmentation"]
)

client = MlflowClient()

if task.startswith("Classification"):
    cls_model_name = "Best_classification_Model"
    try:
        # Find the latest version tagged as Production
        versions = client.search_model_versions(f"name='{cls_model_name}'")
        prod_versions = [v for v in versions if v.tags.get("stage") == "Production"]

        if not prod_versions:
            raise RuntimeError("No model version tagged as Production!")

        # Pick latest Production version
        latest_prod_version = max(prod_versions, key=lambda v: int(v.version))
        cls_model = mlflow.pyfunc.load_model(f"models:/{cls_model_name}/{latest_prod_version.version}")
        st.success(f"Loaded model: {cls_model_name}")
    except MlflowException as e:
        cls_model = None
        warnings.warn(f"Model '{cls_model_name}' not found in MLflow. Skipping classification predictions.")
    except RuntimeError as e:
        cls_model = None
        warnings.warn(f"RuntimeError: No version tagged as Production for '{cls_model_name}'. Skipping classification.\nDetails: {e}")

if task.startswith("Regression"):
    reg_model_name = "Best_regression_Model"
    try:
        # Find the latest version tagged as Production
        versions = client.search_model_versions(f"name='{reg_model_name}'")
        prod_versions = [v for v in versions if v.tags.get("stage") == "Production"]

        if not prod_versions:
            raise RuntimeError("No model version tagged as Production!")

        # Pick latest Production version
        latest_prod_version = max(prod_versions, key=lambda v: int(v.version))
        reg_model  = mlflow.pyfunc.load_model(f"models:/{reg_model_name}/{latest_prod_version.version}")
        st.success(f"Loaded model: {reg_model_name}")
    except MlflowException as e:
        reg_model = None
        warnings.warn(f"MLflowException: Could not load '{reg_model_name}' in Production stage. Skipping Regression.\nDetails: {e}")
    except RuntimeError as e:
        reg_model = None
        warnings.warn(f"RuntimeError: No version tagged as Production for '{reg_model_name}'. Skipping Regression.\nDetails: {e}")
    

if task.startswith("Clustering"):
    cluster_model_name = "Best_clustering_Model"
    try:
        # Find the latest version tagged as Production
        versions = client.search_model_versions(f"name='{cls_model_name}'")
        prod_versions = [v for v in versions if v.tags.get("stage") == "Production"]

        if not prod_versions:
            raise RuntimeError("No model version tagged as Production!")

        # Pick latest Production version
        latest_prod_version = max(prod_versions, key=lambda v: int(v.version))
        cluster_model  = mlflow.pyfunc.load_model(f"models:/{cluster_model_name}/{latest_prod_version.version}")
        st.success(f"Loaded model: {cluster_model_name}")
    except MlflowException as e:
        cluster_model = None
        warnings.warn(f"MLflowException: Could not load '{cluster_model_name}' in Production stage. Skipping Regression.\nDetails: {e}")
    except RuntimeError as e:
        cluster_model = None
        warnings.warn(f"RuntimeError: No version tagged as Production for '{cluster_model_name}'. Skipping Regression.\nDetails: {e}")


if uploaded_file:
    st.subheader("Model Results")

    if task.startswith("Classification") and cls_model:
        df['predicted_conversion'] = cls_model.predict(df)
        st.write(df[['predicted_conversion']].head())
        # Show proportion
        st.bar_chart(df['predicted_conversion'].value_counts())

    elif task.startswith("Regression") and reg_model:
        df['predicted_revenue'] = reg_model.predict(df)
        st.write(df[['predicted_revenue']].head())
        # Visualize distribution
        st.hist_chart(df['predicted_revenue'])

    elif task.startswith("Clustering") and cluster_model:
        df['cluster'] = cluster_model.predict(df)
        st.write(df[['cluster']].head())
        # Cluster counts
        st.bar_chart(df['cluster'].value_counts())
        # Optional: PCA for 2D visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.select_dtypes(include=np.number))
        df['pca_1'] = pca_result[:,0]
        df['pca_2'] = pca_result[:,1]
        st.scatter_chart(df[['pca_1','pca_2','cluster']])
    else:
        print("skipped due to missing model or task mismatch.")


if uploaded_file:
    st.subheader("Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    col = st.selectbox("Select numeric column for visualization", numeric_cols)
    
    st.write("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)
    
    st.write("Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)