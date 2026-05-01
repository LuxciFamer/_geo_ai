import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def ensure_dirs(base_dir: Path) -> dict:
    output_dir = base_dir / "outputs" / "experiment3"
    report_dir = output_dir / "reports"
    figure_dir = output_dir / "figures"
    prediction_dir = output_dir / "predictions"

    for d in [output_dir, report_dir, figure_dir, prediction_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {"output": output_dir, "report": report_dir, "figure": figure_dir, "prediction": prediction_dir}


def find_data_file(base_dir: Path) -> Path:
    candidates = [
        base_dir / "WeatherData.csv",
        base_dir / "WeatherData.Csv",
        base_dir / "Weather_Data.csv",
        base_dir / "weatherAUS.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("未找到数据文件，请确认工作区中存在 WeatherData.csv 或 Weather_Data.csv")


def load_data(base_dir: Path) -> pd.DataFrame:
    data_path = find_data_file(base_dir)
    df: pd.DataFrame = pd.read_csv(str(data_path), encoding="utf-8-sig")
    print(f"已加载数据: {data_path}")
    print(f"原始数据形状: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    required_cols = ["Date/Time", "Temp_C"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise KeyError(f"数据缺少必要列: {missing_required}")

    work_df = df.copy()
    work_df["Date/Time"] = pd.to_datetime(work_df["Date/Time"], errors="coerce")
    work_df = work_df.dropna(subset=["Date/Time", "Temp_C"]).sort_values("Date/Time").reset_index(drop=True)

    dt = work_df["Date/Time"]
    work_df["Year"] = dt.dt.year
    work_df["Month"] = dt.dt.month
    work_df["Day"] = dt.dt.day
    work_df["Hour"] = dt.dt.hour
    work_df["Weekday"] = dt.dt.weekday
    work_df["DayOfYear"] = dt.dt.dayofyear
    work_df["IsWeekend"] = work_df["Weekday"].isin([5, 6]).astype(int)
    work_df["Hour_sin"] = np.sin(2 * np.pi * work_df["Hour"] / 24.0)
    work_df["Hour_cos"] = np.cos(2 * np.pi * work_df["Hour"] / 24.0)
    work_df["DayOfYear_sin"] = np.sin(2 * np.pi * work_df["DayOfYear"] / 365.25)
    work_df["DayOfYear_cos"] = np.cos(2 * np.pi * work_df["DayOfYear"] / 365.25)

    target = work_df["Temp_C"].astype(float)
    timestamps = work_df["Date/Time"]

    # 预测气温时，保留同一时刻的气象特征与时间特征作为解释变量。
    feature_cols = [
        "Dew Point Temp_C",
        "Rel Hum_%",
        "Wind Speed_km/h",
        "Visibility_km",
        "Press_kPa",
        "Weather",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Weekday",
        "DayOfYear",
        "IsWeekend",
        "Hour_sin",
        "Hour_cos",
        "DayOfYear_sin",
        "DayOfYear_cos",
    ]

    available_features = [c for c in feature_cols if c in work_df.columns]
    X = work_df[available_features].copy()

    return X, target, timestamps


def time_based_split(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series, test_ratio: float = 0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    ts_train = timestamps.iloc[:split_idx].copy()
    ts_test = timestamps.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test, ts_train, ts_test


def make_pipeline(X_train: pd.DataFrame) -> Pipeline:
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

    model = LinearRegression()

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_and_save(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    ts_test: pd.Series,
    report_dir: Path,
    figure_dir: Path,
    prediction_dir: Path,
    base_dir: Path,
):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = pd.DataFrame([
        {
            "dataset": "train",
            "MAE": mean_absolute_error(y_train, y_train_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "R2": r2_score(y_train, y_train_pred),
        },
        {
            "dataset": "test",
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "R2": r2_score(y_test, y_test_pred),
        },
    ])
    metrics.to_csv(report_dir / "regression_metrics.csv", index=False, encoding="utf-8-sig")

    predictions = pd.DataFrame({
        "Date/Time": ts_test.values,
        "Actual_Temp_C": y_test.values,
        "Predicted_Temp_C": y_test_pred,
        "Residual": y_test.values - y_test_pred,
    })
    predictions.to_csv(prediction_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 导出线性回归系数，便于解释各特征对气温的影响。
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(report_dir / "linear_regression_coefficients.csv", index=False, encoding="utf-8-sig")

    # 保存简要文字报告。
    summary_path = report_dir / "model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("实验三：使用线性回归进行气温预测\n")
        f.write(f"数据文件: {find_data_file(base_dir)}\n")
        f.write(f"训练样本数: {len(X_train)}\n")
        f.write(f"测试样本数: {len(X_test)}\n")
        f.write("\n评估指标:\n")
        for _, row in metrics.iterrows():
            f.write(
                f"{row['dataset']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R2={row['R2']:.4f}\n"
            )

    # 图1：真实值 vs 预测值散点图
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test, y=y_test_pred, s=18, alpha=0.6)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    plt.xlabel("Actual Temp_C")
    plt.ylabel("Predicted Temp_C")
    plt.title("Actual vs Predicted Temperature")
    plt.tight_layout()
    plt.savefig(figure_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 图2：残差分布
    residuals = y_test.values - y_test_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=40)
    plt.xlabel("Residual")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(figure_dir / "residual_distribution.png", dpi=150)
    plt.close()

    # 图3：时间序列对比
    plt.figure(figsize=(12, 5))
    plot_n = min(300, len(predictions))
    plt.plot(predictions["Date/Time"].iloc[:plot_n], predictions["Actual_Temp_C"].iloc[:plot_n], label="Actual", linewidth=1.6)
    plt.plot(predictions["Date/Time"].iloc[:plot_n], predictions["Predicted_Temp_C"].iloc[:plot_n], label="Predicted", linewidth=1.2)
    plt.xlabel("Date/Time")
    plt.ylabel("Temp_C")
    plt.title("Temperature Prediction on Test Set (First 300 Points)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(figure_dir / "time_series_comparison.png", dpi=150)
    plt.close()

    print("\n===== 回归模型评估结果 =====")
    print(metrics.to_string(index=False))
    print(f"\n详细结果已保存到: {report_dir}")
    print(f"预测结果已保存到: {prediction_dir}")
    print(f"图表已保存到: {figure_dir}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_dirs(base_dir)

    df = load_data(base_dir)
    X, y, timestamps = build_features(df)

    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"使用特征: {list(X.columns)}")

    X_train, X_test, y_train, y_test, ts_train, ts_test = time_based_split(X, y, timestamps)
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练时间范围: {ts_train.iloc[0]} ~ {ts_train.iloc[-1]}")
    print(f"测试时间范围: {ts_test.iloc[0]} ~ {ts_test.iloc[-1]}")

    model = make_pipeline(X_train)
    model.fit(X_train, y_train)

    evaluate_and_save(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ts_test=ts_test,
        report_dir=dirs["report"],
        figure_dir=dirs["figure"],
        prediction_dir=dirs["prediction"],
        base_dir=base_dir,
    )

    print("\n实验三运行完成。")
    print(f"输出目录: {dirs['output']}")


if __name__ == "__main__":
    main()


