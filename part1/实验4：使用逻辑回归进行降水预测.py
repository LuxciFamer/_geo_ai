import warnings
from typing import cast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def ensure_dirs(base_dir: Path) -> dict:
    output_dir = base_dir / "outputs" / "experiment4"
    report_dir = output_dir / "reports"
    figure_dir = output_dir / "figures"
    prediction_dir = output_dir / "predictions"

    for d in [output_dir, report_dir, figure_dir, prediction_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "output": output_dir,
        "report": report_dir,
        "figure": figure_dir,
        "prediction": prediction_dir,
    }


def find_data_file(base_dir: Path) -> Path:
    candidates = [
        base_dir / "weatherAUS.csv",
        base_dir / "WeatherData.csv",
        base_dir / "Weather_Data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到 weatherAUS.csv / WeatherData.csv / Weather_Data.csv")


def load_data(base_dir: Path) -> pd.DataFrame:
    data_path = find_data_file(base_dir)
    df = cast(pd.DataFrame, pd.read_csv(str(data_path), low_memory=False))
    print(f"已加载数据: {data_path}")
    print(f"原始数据形状: {df.shape}")
    return df


def normalize_binary_target(series: pd.Series) -> pd.Series:
    mapping = {
        "Yes": 1,
        "No": 0,
        "Y": 1,
        "N": 0,
        True: 1,
        False: 0,
    }
    cleaned = series.astype(str).str.strip()
    return cleaned.map(mapping)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required_cols = ["RainTomorrow", "Date"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise KeyError(f"数据缺少必要列: {missing_required}")

    work_df = df.copy()
    work_df["Date"] = pd.to_datetime(work_df["Date"], errors="coerce")
    work_df = work_df.dropna(subset=["RainTomorrow", "Date"]).reset_index(drop=True)

    # 目标变量：次日是否降水（Yes/No -> 1/0）
    y = normalize_binary_target(work_df["RainTomorrow"])
    valid_mask = y.notna()
    work_df = work_df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    dt = work_df["Date"]
    work_df["Year"] = dt.dt.year
    work_df["Month"] = dt.dt.month
    work_df["Day"] = dt.dt.day
    work_df["Weekday"] = dt.dt.weekday
    work_df["DayOfYear"] = dt.dt.dayofyear
    work_df["IsWeekend"] = work_df["Weekday"].isin([5, 6]).astype(int)
    work_df["Month_sin"] = np.sin(2 * np.pi * work_df["Month"] / 12.0)
    work_df["Month_cos"] = np.cos(2 * np.pi * work_df["Month"] / 12.0)
    work_df["DayOfYear_sin"] = np.sin(2 * np.pi * work_df["DayOfYear"] / 365.25)
    work_df["DayOfYear_cos"] = np.cos(2 * np.pi * work_df["DayOfYear"] / 365.25)

    # 删除可能造成泄漏或不适合直接建模的目标相关字段，仅保留作为输入的天气信息。
    drop_cols = ["RainTomorrow", "Date"]
    feature_df = work_df.drop(columns=drop_cols)

    # 处理 RainToday 这种二值类别，后续交给类别管道独热编码。
    if "RainToday" in feature_df.columns:
        feature_df["RainToday"] = feature_df["RainToday"].astype(str).str.strip()

    # 剔除明显的泄漏字段：RainTomorrow 已删除；其余均作为同日特征用于预测次日是否降水。
    return feature_df, y


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_and_save(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    report_dir: Path,
    figure_dir: Path,
    prediction_dir: Path,
) -> None:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame([
        {
            "dataset": "train",
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1": f1_score(y_train, y_train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
        },
        {
            "dataset": "test",
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_test_prob),
        },
    ])
    metrics.to_csv(report_dir / "classification_metrics.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv(report_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    report_text = classification_report(y_test, y_test_pred, target_names=["No Rain", "Rain"], zero_division=0)
    with open(report_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("实验4：使用逻辑回归进行降水预测\n")
        f.write("\n分类报告:\n")
        f.write(str(report_text))
        f.write("\n\n评估指标:\n")
        f.write(metrics.to_string(index=False))

    # 预测结果保存
    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_test_pred,
        "Predicted_Prob_Rain": y_test_prob,
    })
    result_df.to_csv(prediction_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_dir / "roc_curve.png", dpi=150)
    plt.close()

    # 混淆矩阵热力图
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # 概率分布
    plt.figure(figsize=(7, 5))
    sns.histplot(y_test_prob, bins=40, kde=True)
    plt.xlabel("Predicted Probability of Rain")
    plt.title("Predicted Probability Distribution")
    plt.tight_layout()
    plt.savefig(figure_dir / "predicted_probability_distribution.png", dpi=150)
    plt.close()

    # 系数导出
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coef = model.named_steps["model"].coef_.ravel()
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(report_dir / "logistic_regression_coefficients.csv", index=False, encoding="utf-8-sig")

    print("\n===== 逻辑回归模型评估结果 =====")
    print(metrics.to_string(index=False))
    print("\n混淆矩阵:")
    print(cm_df)
    print(f"\n分类报告已保存到: {report_dir / 'classification_report.txt'}")
    print(f"预测结果已保存到: {prediction_dir}")
    print(f"图表已保存到: {figure_dir}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_dirs(base_dir)

    df = load_data(base_dir)
    X, y = build_features(df)

    print(f"特征矩阵形状: {X.shape}")
    print(f"目标分布:\n{y.value_counts(normalize=True).rename('ratio')}")
    print(f"使用特征数量: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    evaluate_and_save(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        report_dir=dirs["report"],
        figure_dir=dirs["figure"],
        prediction_dir=dirs["prediction"],
    )

    print("\n实验4运行完成。")
    print(f"输出目录: {dirs['output']}")


if __name__ == "__main__":
    main()


