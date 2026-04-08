from pathlib import Path
import argparse
import json
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
OUTPUT_DIR = Path("outputs")
DATA_CACHE_PATH = Path("data") / "weatherAUS-2.csv"


def date_to_season(date):
    month = date.month
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def load_dataset_with_cache(url, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Using cached dataset: {cache_path}")
        return pd.read_csv(cache_path)

    print("Downloading dataset from remote source...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        print(f"Dataset downloaded and cached at: {cache_path}")
        return pd.read_csv(cache_path)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Dataset download failed (network timeout). Please check your internet and rerun."
        ) from exc


def load_and_prepare_data(url):
    df = load_dataset_with_cache(url, DATA_CACHE_PATH).dropna()

    # We predict next day rain, so we rename labels for clearer semantics.
    df = df.rename(columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"})
    df = df[df.Location.isin(["Melbourne", "MelbourneAirport", "Watsonia"])]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Season"] = df["Date"].apply(date_to_season)
    df = df.drop(columns=["Date"])

    X = df.drop(columns=["RainToday"])
    y = df["RainToday"]
    return X, y


def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def run_model(name, pipeline, param_grid, X_train, y_train, X_test, y_test, cv_splits):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    model = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", verbose=2, n_jobs=-1)
    print(f"\n[{name}] Training started...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n{name} - Best Params: {model.best_params_}")
    print(f"{name} - CV Accuracy: {model.best_score_:.4f}")
    print(f"{name} - Test Accuracy: {model.score(X_test, y_test):.4f}")
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    return model, y_pred


def build_metrics_row(name, model, y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "model": name,
        "cv_accuracy": round(model.best_score_, 4),
        "test_accuracy": round((y_test == y_pred).mean(), 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_precision": round(report["weighted avg"]["precision"], 4),
        "weighted_recall": round(report["weighted avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "best_params": json.dumps(model.best_params_, sort_keys=True),
    }


def save_metrics_csv(rows, output_path):
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)


def save_summary_report(rows, output_path):
    best_model = max(rows, key=lambda row: row["cv_accuracy"])
    lines = [
        "Rainfall Prediction Classifier - Model Summary",
        "",
        "Results by model:",
    ]

    for row in rows:
        lines.extend(
            [
                f"- {row['model']}",
                f"  CV Accuracy: {row['cv_accuracy']:.4f}",
                f"  Test Accuracy: {row['test_accuracy']:.4f}",
                f"  Macro F1: {row['macro_f1']:.4f}",
                f"  Weighted F1: {row['weighted_f1']:.4f}",
                f"  Best Params: {row['best_params']}",
                "",
            ]
        )

    lines.extend(
        [
            "Best model by CV accuracy:",
            f"- {best_model['model']} ({best_model['cv_accuracy']:.4f})",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix(y_test, y_pred, title, output_path):
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_importance(model, numeric_features, categorical_features, output_path, n_top=20):
    feature_names = numeric_features + list(
        model.best_estimator_["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    feature_importances = model.best_estimator_["classifier"].feature_importances_

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances}).sort_values(
        by="Importance", ascending=False
    )
    top_features = importance_df.head(n_top)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.title(f"Top {n_top} Most Important Features (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_logistic_confusion_matrix(y_test, y_pred, output_path):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.title("Rainfall Classification Confusion Matrix (Logistic Regression)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train rainfall prediction models.")
    parser.add_argument(
        "--full-search",
        action="store_true",
        help="Use the larger hyperparameter grids (slower but potentially better).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading and preparing data...")
    X, y = load_and_prepare_data(DATA_URL)
    print(f"Dataset ready: {len(X)} rows, {X.shape[1]} features")

    if not args.full_search and len(X) > 3000:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=3000,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        print("Quick mode enabled: using a 3000-row stratified sample for faster training")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    if args.full_search:
        rf_param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
        }
    else:
        rf_param_grid = {
            "classifier__n_estimators": [50],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_split": [2],
        }
    cv_splits = 5 if args.full_search else 3

    rf_model, rf_pred = run_model(
        "Random Forest", rf_pipeline, rf_param_grid, X_train, y_train, X_test, y_test, cv_splits
    )
    metrics_rows = [build_metrics_row("Random Forest", rf_model, y_test, rf_pred)]
    save_confusion_matrix(
        y_test,
        rf_pred,
        "Rainfall Classification Confusion Matrix (Random Forest)",
        OUTPUT_DIR / "confusion_matrix_random_forest.png",
    )
    save_feature_importance(
        rf_model,
        numeric_features,
        categorical_features,
        OUTPUT_DIR / "feature_importance_random_forest.png",
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
        ]
    )
    if args.full_search:
        lr_param_grid = {
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l1", "l2"],
            "classifier__class_weight": [None, "balanced"],
        }
    else:
        lr_param_grid = {
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l2"],
            "classifier__class_weight": [None, "balanced"],
        }
    lr_model, lr_pred = run_model(
        "Logistic Regression", lr_pipeline, lr_param_grid, X_train, y_train, X_test, y_test, cv_splits
    )
    metrics_rows.append(build_metrics_row("Logistic Regression", lr_model, y_test, lr_pred))
    save_logistic_confusion_matrix(y_test, lr_pred, OUTPUT_DIR / "confusion_matrix_logistic_regression.png")
    save_metrics_csv(metrics_rows, OUTPUT_DIR / "model_metrics.csv")
    save_summary_report(metrics_rows, OUTPUT_DIR / "model_summary.txt")

    print("\nSaved outputs in ./outputs:")
    print("- confusion_matrix_random_forest.png")
    print("- feature_importance_random_forest.png")
    print("- confusion_matrix_logistic_regression.png")
    print("- model_metrics.csv")
    print("- model_summary.txt")
    print(f"- best_model_logistic_regression_cv_accuracy: {lr_model.best_score_:.4f}")


if __name__ == "__main__":
    main()