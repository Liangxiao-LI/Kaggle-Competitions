from __future__ import annotations

from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

TRAIN_PATH = Path(__file__).with_name("train.csv")
TEST_PATH = Path(__file__).with_name("test.csv")
SUBMISSION_PATH = Path(__file__).with_name("submission.csv")
MODEL_PATH = Path(__file__).with_name("irrigation_model.joblib")
OPTUNA_RESULTS_PATH = Path(__file__).with_name("optuna_tuning_results.csv")

TARGET_COLUMN = "Irrigation_Need"
ID_COLUMN = "id"
VALIDATION_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 3
TUNING_SAMPLE_SIZE = 150_000
OPTUNA_TRIALS = 15


def print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists():
        raise SystemExit(f"Could not find training data: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise SystemExit(f"Could not find test data: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def run_eda(train_df: pd.DataFrame) -> None:
    feature_df = train_df.drop(columns=[TARGET_COLUMN])
    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in feature_df.columns if column not in numeric_columns]

    print("EDA for irrigation training data")
    print("=" * 32)

    print_header("1. Dataset Overview")
    print(f"Rows: {len(train_df)}")
    print(f"Columns: {train_df.shape[1]}")
    print(f"Numeric columns: {len(numeric_columns)} -> {', '.join(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns) + 1} -> {', '.join(categorical_columns + [TARGET_COLUMN])}")

    print_header("2. Missing Value Check")
    missing_counts = train_df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if missing_counts.empty:
        print("No missing values detected in train.csv.")
    else:
        for column, count in missing_counts.items():
            print(f"{column}: {count} missing ({count / len(train_df):.2%})")

    print_header("3. Target Distribution")
    target_distribution = train_df[TARGET_COLUMN].value_counts()
    for label, count in target_distribution.items():
        print(f"{label}: {count} ({count / len(train_df):.2%})")

    imbalance_ratio = target_distribution.max() / target_distribution.min()
    print(f"Imbalance ratio (largest / smallest class): {imbalance_ratio:.2f}")
    print("Use stratified split and prefer macro-F1 / balanced accuracy during validation.")

    print_header("4. Numeric Column Summary")
    numeric_summary = feature_df[numeric_columns].describe().T[["mean", "std", "min", "max"]]
    print(numeric_summary.to_string(float_format=lambda value: f"{value:.4f}"))

    print_header("5. Outlier Check (IQR Rule)")
    report_lines: list[str] = []
    for column in numeric_columns:
        if column == ID_COLUMN:
            continue

        q1 = train_df[column].quantile(0.25)
        q3 = train_df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (train_df[column] < lower) | (train_df[column] > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_ratio = outlier_count / len(train_df)
        report_lines.append(
            f"{column}: outliers={outlier_count} ({outlier_ratio:.2%}), bounds=[{lower:.4f}, {upper:.4f}]"
        )

    if report_lines:
        for line in report_lines:
            print(line)
    else:
        print("No numeric columns available for outlier detection.")


def build_pipeline(X: pd.DataFrame, num_classes: int, model_params: dict[str, object] | None = None) -> Pipeline:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    params = get_default_xgb_params(num_classes)
    if model_params:
        params.update(model_params)

    model = XGBClassifier(**params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_default_xgb_params(num_classes: int) -> dict[str, object]:
    return {
        "objective": "multi:softmax",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }


def suggest_xgb_params(trial: optuna.Trial, num_classes: int) -> dict[str, object]:
    params = get_default_xgb_params(num_classes)
    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 250, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.00, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.00, step=0.05),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.5, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
    )
    return params


def to_python_scalar(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return value


def sort_tuning_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        by=["mean_macro_f1", "mean_balanced_accuracy", "std_macro_f1", "std_balanced_accuracy"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def extract_best_params(results_df: pd.DataFrame, num_classes: int) -> dict[str, object]:
    best_row = results_df.iloc[0]
    parameter_keys = get_default_xgb_params(num_classes).keys() | {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    }
    return {
        key: to_python_scalar(best_row[key])
        for key in parameter_keys
        if key in results_df.columns
    }


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    label_encoder = LabelEncoder()
    encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)
    return encoded, label_encoder


def decode_predictions(predictions: pd.Series | list[int], label_encoder: LabelEncoder) -> list[str]:
    return label_encoder.inverse_transform(predictions).tolist()


def build_sample_weights(y: pd.Series) -> pd.Series:
    weights = compute_sample_weight(class_weight="balanced", y=y)
    return pd.Series(weights, index=y.index)


def split_features_and_target(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = train_df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = train_df[TARGET_COLUMN]
    return X, y


def sample_for_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_encoded: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if len(X_train) <= TUNING_SAMPLE_SIZE:
        return X_train, y_train, y_train_encoded

    X_tune, _, y_tune, _, y_tune_encoded, _ = train_test_split(
        X_train,
        y_train,
        y_train_encoded,
        train_size=TUNING_SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_SEED,
    )
    return X_tune, y_tune, y_tune_encoded


def cross_validate_candidates(
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    y_tune_encoded: pd.Series,
    label_encoder: LabelEncoder,
) -> tuple[dict[str, object], pd.DataFrame]:
    num_classes = len(label_encoder.classes_)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    print_header("7. Hyperparameter Tuning")
    print(f"Tuning rows: {len(X_tune)}")
    print(f"Trials: {OPTUNA_TRIALS}")
    print(f"CV folds: {CV_FOLDS}")
    print("Using Optuna TPE sampler. Scoring priority: macro-F1, then balanced accuracy, then lower fold variance.")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial, num_classes)
        macro_f1_scores: list[float] = []
        balanced_accuracy_scores: list[float] = []

        for fold_train_idx, fold_valid_idx in skf.split(X_tune, y_tune):
            X_fold_train = X_tune.iloc[fold_train_idx]
            X_fold_valid = X_tune.iloc[fold_valid_idx]
            y_fold_train = y_tune.iloc[fold_train_idx]
            y_fold_valid = y_tune.iloc[fold_valid_idx]
            y_fold_train_encoded = y_tune_encoded.iloc[fold_train_idx]

            pipeline = build_pipeline(X_fold_train, num_classes=num_classes, model_params=params)
            sample_weights = build_sample_weights(y_fold_train)
            pipeline.fit(
                X_fold_train,
                y_fold_train_encoded,
                model__sample_weight=sample_weights.to_numpy(),
            )

            fold_predictions_encoded = pipeline.predict(X_fold_valid)
            fold_predictions = decode_predictions(fold_predictions_encoded, label_encoder)
            macro_f1_scores.append(f1_score(y_fold_valid, fold_predictions, average="macro"))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_fold_valid, fold_predictions))

        mean_macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores)
        mean_balanced_accuracy = sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)
        std_macro_f1 = pd.Series(macro_f1_scores).std(ddof=0)
        std_balanced_accuracy = pd.Series(balanced_accuracy_scores).std(ddof=0)

        trial.set_user_attr("mean_balanced_accuracy", mean_balanced_accuracy)
        trial.set_user_attr("std_macro_f1", std_macro_f1)
        trial.set_user_attr("std_balanced_accuracy", std_balanced_accuracy)
        print(
            f"Trial {trial.number + 1}/{OPTUNA_TRIALS}: "
            f"macro-F1={mean_macro_f1:.4f}, "
            f"balanced_acc={mean_balanced_accuracy:.4f}, "
            f"std_macro_F1={std_macro_f1:.4f}"
        )
        return mean_macro_f1

    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    completed_rows: list[dict[str, object]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": trial.number,
            "mean_macro_f1": trial.value,
            "mean_balanced_accuracy": trial.user_attrs["mean_balanced_accuracy"],
            "std_macro_f1": trial.user_attrs["std_macro_f1"],
            "std_balanced_accuracy": trial.user_attrs["std_balanced_accuracy"],
            **trial.params,
            **get_default_xgb_params(num_classes),
        }
        completed_rows.append(row)

    results_df = sort_tuning_results(pd.DataFrame(completed_rows))
    best_params = extract_best_params(results_df, num_classes)
    return best_params, results_df


def evaluate_model(
    model: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    label_encoder: LabelEncoder,
) -> tuple[float, float, float]:
    encoded_predictions = model.predict(X_valid)
    decoded_predictions = decode_predictions(encoded_predictions, label_encoder)
    macro_f1 = f1_score(y_valid, decoded_predictions, average="macro")
    balanced_acc = balanced_accuracy_score(y_valid, decoded_predictions)
    accuracy = accuracy_score(y_valid, decoded_predictions)

    print_header("10. Validation Metrics")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print_header("11. Validation Classification Report")
    print(classification_report(y_valid, decoded_predictions, digits=4))
    return macro_f1, balanced_acc, accuracy


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    X, y = split_features_and_target(train_df)
    y_encoded, label_encoder = encode_target(y)
    X_train, X_valid, y_train, y_valid, y_train_encoded, _ = train_test_split(
        X,
        y,
        y_encoded,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    train_sample_weights = build_sample_weights(y_train)

    print_header("6. Stratified Split")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_valid)}")

    if OPTUNA_RESULTS_PATH.exists():
        tuning_results_df = sort_tuning_results(pd.read_csv(OPTUNA_RESULTS_PATH))
        best_params = extract_best_params(tuning_results_df, num_classes=len(label_encoder.classes_))
        print_header("7. Hyperparameter Tuning")
        print(f"Loaded existing Optuna tuning results from: {OPTUNA_RESULTS_PATH.name}")
    else:
        X_tune, y_tune, y_tune_encoded = sample_for_tuning(X_train, y_train, y_train_encoded)
        best_params, tuning_results_df = cross_validate_candidates(X_tune, y_tune, y_tune_encoded, label_encoder)
        tuning_results_df.to_csv(OPTUNA_RESULTS_PATH, index=False)

    print_header("8. Best Parameters")
    for key in sorted(best_params):
        print(f"{key}: {best_params[key]}")
    print(f"Saved tuning results to: {OPTUNA_RESULTS_PATH.name}")

    pipeline = build_pipeline(X_train, num_classes=len(label_encoder.classes_), model_params=best_params)

    print_header("9. Final Training On Train Split")
    print("Training tuned XGBoost classifier with ordinal-encoded categorical features...")
    pipeline.fit(X_train, y_train_encoded, model__sample_weight=train_sample_weights.to_numpy())

    evaluate_model(pipeline, X_valid, y_valid, label_encoder)

    print_header("12. Refit On Full Training Data")
    final_pipeline = build_pipeline(X, num_classes=len(label_encoder.classes_), model_params=best_params)
    full_sample_weights = build_sample_weights(y)
    final_pipeline.fit(X, y_encoded, model__sample_weight=full_sample_weights.to_numpy())
    joblib.dump(
        {
            "model": final_pipeline,
            "label_encoder": label_encoder,
            "best_params": best_params,
            "tuning_results_path": OPTUNA_RESULTS_PATH.name,
        },
        MODEL_PATH,
    )
    print(f"Saved model to: {MODEL_PATH.name}")

    test_features = test_df.drop(columns=[ID_COLUMN], errors="ignore")
    test_predictions_encoded = final_pipeline.predict(test_features)
    test_predictions = decode_predictions(test_predictions_encoded, label_encoder)

    submission_df = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            TARGET_COLUMN: test_predictions,
        }
    )
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print_header("13. Submission File")
    print(f"Saved predictions to: {SUBMISSION_PATH.name}")
    print(submission_df.head(10).to_string(index=False))


def main() -> None:
    train_df, test_df = load_data()
    run_eda(train_df)
    train_and_predict(train_df, test_df)


if __name__ == "__main__":
    main()
