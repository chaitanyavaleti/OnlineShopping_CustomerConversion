import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score,
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



# ========== Your evaluate_model Function ==========
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
def unified_pipeline(df, task="classification", target_col=None, n_clusters=3):
    results = []

    mlflow.set_experiment("Customer Conversion Analysis for Online Shopping")
    client = MlflowClient()

    # Feature/target split
    if task in ["classification", "regression"]:
        if target_col is None:
            raise ValueError("target_col must be provided for classification/regression")

        if df[target_col].nunique() < 2:
            print(f"WARNING: Target column '{target_col}' has only one unique value. Classification cannot be performed.")
            return None

        X = df.drop(columns=[target_col, "session_id", "page2_clothing_model"], errors="ignore")
        y = df[target_col]
    else:
        X = df.drop(columns=["session_id", "page2_clothing_model"], errors="ignore")
        y = None

    # Train-test split for supervised tasks
    if task in ["classification", "regression"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    # Preprocessing
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        [("num", numeric_transformer, numeric_features),
         ("cat", categorical_transformer, categorical_features)]
    )

    # ===== Models =====
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
            "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200),
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
            "KMeans": KMeans(n_clusters=n_clusters, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
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

                if hasattr(model, "predict_proba"):
                    y_proba = pipeline.predict_proba(X_test)[:,1]
                else:  # for models that don't support predict_proba
                    y_proba = pipeline.decision_function(X_test)

                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                mlflow.log_metric("ROC-AUC", auc)
                mlflow.sklearn.log_model(pipe, name="model", registered_model_name=f"{task}_{name}", signature=signature, input_example=input_example)
                run_info.append((name, auc, run_id))
                results.append({"Model": name, "ROC-AUC": auc})

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
                run_info.append((name, metrics["R-squared"], run_id, metrics))
                results.append({"Model": name, **metrics})

            elif task == "clustering":
                cluster_labels = pipe.fit_predict(X)
                # Infer signature
                signature = infer_signature(X_train, cluster_labels)

                # Take a small sample as input_example
                input_example = X_train.iloc[:5]

                if len(set(cluster_labels)) > 1:
                    metrics = evaluate_model("clustering", X=X.values, cluster_labels=cluster_labels)
                    score = metrics["Silhouette Score"]
                else:
                    metrics = {"Silhouette Score": np.nan, "Davies-Bouldin Index": np.nan, "WCSS": np.nan}
                    score = np.nan

                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipe, name="model", registered_model_name=f"{task}_{name}", signature=signature, input_example=input_example)
                run_info.append((name, score, run_id, metrics))
                results.append({"Model": name, **metrics})

    # Convert to DataFrame & Rank
    results_df = pd.DataFrame(results)
    
    best_model = max(run_info, key=lambda x: x[1])

    if task == "classification":
        results_df = results_df.sort_values(by="ROC-AUC", ascending=False)
        best_name, best_score, best_run_id = best_model
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
    mv = mlflow.register_model(model_uri, registered_model_name)

    for key, value in tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=mv.version,
            key=key,
            value=value
        )

    return {"Best Model": best_name, "Score": best_score, "Registered As": registered_model_name, "Version": mv.version}
    
train = pd.read_csv("train_data.csv")
train["converted"] = train["order"].apply(lambda x: 1 if x > 0 else 0)


best_cls = unified_pipeline(train, task="classification", target_col="converted")
print(best_cls)

best_reg = unified_pipeline(train, task="regression", target_col="price")
print(best_reg)

best_cluster = unified_pipeline(train, task="clustering", n_clusters=3)
print(best_cluster)

