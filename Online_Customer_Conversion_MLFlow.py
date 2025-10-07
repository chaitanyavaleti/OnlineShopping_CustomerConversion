import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBClassifier, XGBRegressor
from mlflow.models import infer_signature
from sklearn.metrics import pairwise_distances_argmin_min
from mlflow.tracking import MlflowClient
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")

def create_weighted_converted(df, k=6, price_mode="normalized"):
    df_copy = df.copy()

    drop_cols = ["order", "price"]
    df_features = df_copy.drop(columns=drop_cols, errors='ignore')
    
    # Define temporary target for conversion
    df_copy["_temp_target"] = ((df_copy["order"] > 0) & (df_copy["price"] > 0)).astype(int)
    y_price = df_copy["_temp_target"]

    X_price = df_features.drop(columns=["_temp_target"], errors='ignore')
    X_price = df_features.select_dtypes(include=[np.number])
    
    
    # Compute mutual info fast
    mi = mutual_info_classif(X_price.select_dtypes(include=[np.number]), y_price)
    mi = pd.Series(mi, index=X_price.select_dtypes(include=[np.number]).columns)
    top_features = mi.sort_values(ascending=False).head(k).index.tolist()
    
    # Weighted conversion score (based on MI)
    weights = mi[top_features] / mi[top_features].sum()
    df_copy['conversion_score'] = (df_copy[top_features] * weights).sum(axis=1)
    
    # Create converted column
    df_copy['converted'] = (df_copy['conversion_score'] > df_copy['conversion_score'].median()).astype(int)
    
    df_copy.drop(columns=['_temp_target'], inplace=True)
    return df_copy, top_features, weights

def apply_weighted_conversion(df_new, top_features, weights):

    df_new = df_new.copy()
    
    # Only use the preselected features
    common_features = [f for f in top_features if f in df_new.columns]
    df_new["conversion_score"] = (df_new[common_features] * weights).sum(axis=1)
    df_new["converted"] = (df_new["conversion_score"] > df_new["conversion_score"].median()).astype(int)
    
    return df_new

def evaluate_model(model_type, y_true=None, y_pred=None, X=None, cluster_labels=None):
    results = {}
    
    if model_type.lower() == "regression":
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred are required for regression evaluation.")
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        results = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R-squared": r2
        }
        
    elif model_type.lower() == "clustering":
        if X is None or cluster_labels is None:
            raise ValueError("X and cluster_labels are required for clustering evaluation.")
        sil_score = silhouette_score(X, cluster_labels)
        db_index = davies_bouldin_score(X, cluster_labels)
        wcss = 0
        for i in np.unique(cluster_labels):
            cluster_points = X[cluster_labels == i]
            centroid = cluster_points.mean(axis=0)
            wcss += np.sum((cluster_points - centroid) ** 2)
        
        results = {
            "Silhouette Score": sil_score,
            "Davies-Bouldin Index": db_index,
            "WCSS": wcss
        }
        
    else:
        raise ValueError("model_type must be either 'regression' or 'clustering'.")
    
    return results

# ========== Unified Pipeline ==========
def unified_pipeline(df, task="classification", target_col=None, n_clusters=3, k_features=6):
    results = []

    mlflow.set_experiment("Customer Conversion Analysis for Online Shopping")
    client = MlflowClient()

    if target_col is None or target_col not in df.columns:
        print("⚡ Creating 'converted' dynamically using top K features...")
        
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        
        df_train, top_features, feature_weights = create_weighted_converted(df_train,k=k_features)
        df_test = apply_weighted_conversion(df_test, top_features, feature_weights)
        target_col = "converted"
        print(f"Top features selected: {top_features}")
        print(f"Assigned weights: {feature_weights}")
    else:
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        top_features = df_train.drop(columns=[target_col, "session_id"], errors="ignore").columns.tolist()


    if task in ["classification", "regression"]:
        X_train = df_train[top_features]
        y_train = df_train[target_col]
        X_test = df_test[top_features]
        y_test = df_test[target_col]

        if y_train.nunique() < 2:
            print(f"⚠️ Target column '{target_col}' has only one unique value. {task} cannot be performed.")
            return None
    elif task == "clustering":
        X = df[top_features]
        X_train = X_test = X
        y_train = y_test = None
    else:
        raise ValueError("task must be one of ['classification','regression','clustering']")

    # Preprocessing
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        [("num", numeric_transformer, numeric_features),
         ("cat", categorical_transformer, categorical_features)]
    )

    # ===== Models =====
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression( max_iter=1000 , class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier( n_estimators=200),
            "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500)
        }
    elif task == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200),
            "XGBoost Regressor": XGBRegressor(eval_metric="rmse")
        }
    elif task == "clustering":
        models = {
            "KMeans": KMeans(n_clusters=n_clusters, n_init=50, max_iter=500,random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
        }

    else:
        raise ValueError("task must be one of ['classification','regression','clustering']")

    # ===== Train & Evaluate =====
    run_info = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        with mlflow.start_run(run_name=f"{task}_{name}") as run:
            run_id = run.info.run_id
            if task == "classification":
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                 # Infer signature
                signature = infer_signature(X_test, y_pred)

                # Take a small sample as input_example
                input_example = X_test.iloc[:5]

                try:
                    y_proba = pipe.predict_proba(X_test)[:,1]
                    auc = roc_auc_score(y_test, y_proba)
                except:
                    auc = np.nan

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                mlflow.log_metrics({"ROC-AUC": auc, "Accuracy": acc, "F1-Score": f1})
                mlflow.sklearn.log_model(pipe, name="model", registered_model_name=f"{task}_{name}", signature=signature, input_example=input_example)
                run_info.append((name, auc, run_id, {"ROC-AUC": auc, "Accuracy": acc, "F1-Score": f1}))
                results.append({"Model": name, "ROC-AUC": auc, "Accuracy": acc, "F1-Score": f1})

            elif task == "regression":
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                # Infer signature
                signature = infer_signature(X_test, y_pred)

                # Take a small sample as input_example
                input_example = X_test.iloc[:5]

                metrics = evaluate_model("regression", y_true=y_test, y_pred=y_pred)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipe, name="model", registered_model_name=f"{task}_{name}", signature=signature, input_example=input_example)
                run_info.append((name,  metrics["R-squared"], run_id, metrics))  # include metrics dict
                results.append({"Model": name, **metrics})

            elif task == "clustering":
                # Fit the pipeline
                cluster_labels = pipe.fit_predict(X)
                
                if categorical_features:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    X_cat = encoder.fit_transform(X[categorical_features])
                    X_proc = np.hstack([X[numeric_features].values, X_cat])
                    input_cols = numeric_features + list(encoder.get_feature_names_out(categorical_features))
                else:
                    X_proc = X[numeric_features].values
                    input_cols = numeric_features

                signature = infer_signature(pd.DataFrame(X_proc, columns=input_cols), cluster_labels)
                input_example = pd.DataFrame(X_proc[:5], columns=input_cols)
                metrics = evaluate_model("clustering", X=X_proc, cluster_labels=cluster_labels) if len(set(cluster_labels)) > 1 else {"Silhouette Score": np.nan, "Davies-Bouldin Index": np.nan, "WCSS": np.nan}
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipe, name="model", registered_model_name=f"Best_clustering_Model", signature=signature, input_example=input_example)
                run_info.append((name, metrics.get("Silhouette Score", np.nan), run_id, metrics))  # clustering score optional
                results.append({"Model": name, **metrics})

    # Convert to DataFrame & Rank
    results_df = pd.DataFrame(results)
    
    best_model = max(run_info, key=lambda x: x[1])

    if task == "classification":
        results_df = results_df.sort_values(by="ROC-AUC", ascending=False)
        best_name, best_score, best_run_id, best_metrics = best_model
        tags = {"stage": "Production", "roc_auc": str(best_score), "model_name": best_name}
    elif task == "regression":
        results_df = results_df.sort_values(by="R-squared", ascending=False)
        best_name, best_score, best_run_id, best_metrics = best_model
        tags = {"stage": "Production", "r2": str(best_score), "model_name": best_name, "rmse": str(best_metrics["RMSE"])}
    elif task == "clustering":
        results_df = results_df.sort_values(by="Silhouette Score", ascending=False)
        best_name, best_score, best_run_id, best_metrics = best_model
        tags = {"stage": "Production", "silhouette": str(best_score), "model_name": best_name}

    print(results_df)


    # Register best model
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = f"Best_{task}_Model"
    try:
        mv = mlflow.register_model(model_uri, registered_model_name)
    except:
        mv = client.get_latest_versions(registered_model_name)[-1]

    for key, value in tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=mv.version,
            key=key,
            value=value
        )

    return {"Best Model": best_name, "Score": best_score, "Registered As": registered_model_name, "Version": mv.version}


train = pd.read_csv("train_data.csv")

best_cls = unified_pipeline(train, task="classification", target_col="converted")
print(best_cls)

best_reg = unified_pipeline(train, task="regression", target_col="price")
print(best_reg)

best_cluster = unified_pipeline(train, task="clustering", n_clusters=5)
print(best_cluster)

