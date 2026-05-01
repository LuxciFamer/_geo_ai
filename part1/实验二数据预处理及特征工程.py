import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


def ensure_dirs(base_dir: Path) -> dict:
    output_dir = base_dir / "outputs"
    report_dir = output_dir / "reports"
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "processed"

    for d in [output_dir, report_dir, fig_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {"output": output_dir, "report": report_dir, "fig": fig_dir, "data": data_dir}


def load_data(base_dir: Path) -> pd.DataFrame:
    # 按用户要求优先读取 WeatherData.Csv；若不存在则回退到工作区内实际文件名。
    preferred = base_dir / "WeatherData.Csv"
    fallback = base_dir / "Weather_Data.csv"

    if preferred.exists():
        data_path = preferred
    elif fallback.exists():
        data_path = fallback
    else:
        raise FileNotFoundError("未找到 WeatherData.Csv 或 Weather_Data.csv")

    df = pd.read_csv(data_path)
    print(f"已加载数据: {data_path}")
    print(f"数据形状: {df.shape}\n")
    return df


def descriptive_statistics(df: pd.DataFrame, report_dir: Path) -> None:
    print("=" * 20, "描述性统计", "=" * 20)
    print(df.describe(include="all").T.head(20))

    numeric_desc = df.describe().T
    object_desc = df.describe(include=["object"]).T

    numeric_desc.to_csv(report_dir / "numeric_descriptive_stats.csv", encoding="utf-8-sig")
    object_desc.to_csv(report_dir / "categorical_descriptive_stats.csv", encoding="utf-8-sig")


def data_type_analysis(df: pd.DataFrame, report_dir: Path) -> None:
    print("\n", "=" * 20, "数据类型分析", "=" * 20)
    type_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null_count": df.notna().sum().values,
    })
    print(type_df)
    type_df.to_csv(report_dir / "data_types.csv", index=False, encoding="utf-8-sig")


def missing_value_analysis(df: pd.DataFrame, report_dir: Path) -> None:
    print("\n", "=" * 20, "缺失值分析", "=" * 20)
    missing_count = df.isna().sum()
    missing_ratio = (missing_count / len(df) * 100).round(4)
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_ratio_percent": missing_ratio.values,
    }).sort_values("missing_ratio_percent", ascending=False)

    print(missing_df)
    missing_df.to_csv(report_dir / "missing_value_report.csv", index=False, encoding="utf-8-sig")


def detect_and_treat_outliers(df: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    print("\n", "=" * 20, "异常值检测(IQR)", "=" * 20)
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    outlier_records = []
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        outlier_count = int(mask.sum())
        outlier_ratio = outlier_count / len(df_clean) * 100

        outlier_records.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outlier_count,
            "outlier_ratio_percent": round(outlier_ratio, 4),
        })

        # 采用截尾法处理异常值，避免直接删除样本造成信息损失。
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    outlier_df = pd.DataFrame(outlier_records)
    print(outlier_df[["column", "outlier_count", "outlier_ratio_percent"]])
    outlier_df.to_csv(report_dir / "outlier_iqr_report.csv", index=False, encoding="utf-8-sig")

    return df_clean


def exploratory_data_analysis(df: pd.DataFrame, fig_dir: Path) -> None:
    print("\n", "=" * 20, "探索性数据分析(EDA)", "=" * 20)

    # 1) 目标变量分布
    if "Weather" in df.columns:
        plt.figure(figsize=(12, 6))
        weather_counts = df["Weather"].value_counts().head(20)
        sns.barplot(x=weather_counts.values, y=weather_counts.index)
        plt.title("Top 20 Weather Categories")
        plt.xlabel("Count")
        plt.ylabel("Weather")
        plt.tight_layout()
        plt.savefig(fig_dir / "target_weather_top20.png", dpi=150)
        plt.close()

    # 2) 数值特征分布
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"hist_{col.replace('/', '_')}.png", dpi=150)
        plt.close()

    # 3) 相关性热力图
    if numeric_cols:
        corr = df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap (Numeric Features)")
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_heatmap.png", dpi=150)
        plt.close()


def feature_engineering_and_split(df: pd.DataFrame, data_dir: Path, report_dir: Path) -> None:
    print("\n", "=" * 20, "特征工程 + 标准化 + 数据集划分", "=" * 20)

    work_df = df.copy()

    if "Date/Time" not in work_df.columns:
        raise KeyError("数据中缺少 Date/Time 列")
    if "Weather" not in work_df.columns:
        raise KeyError("数据中缺少 Weather 列")

    # 解析时间并构建时间特征
    dt = pd.to_datetime(work_df["Date/Time"], errors="coerce")
    work_df["Year"] = dt.dt.year
    work_df["Month"] = dt.dt.month
    work_df["Day"] = dt.dt.day
    work_df["Hour"] = dt.dt.hour
    work_df["Weekday"] = dt.dt.weekday
    work_df["IsWeekend"] = work_df["Weekday"].isin([5, 6]).astype(int)
    work_df["Hour_sin"] = np.sin(2 * np.pi * work_df["Hour"] / 24.0)
    work_df["Hour_cos"] = np.cos(2 * np.pi * work_df["Hour"] / 24.0)

    # 将复合天气标签简化为主天气现象，便于单标签分类
    y_raw = work_df["Weather"].astype(str).str.split(",").str[0].str.strip()

    # 分层抽样要求每一类至少2个样本；稀有类先合并为 Rare_Other。
    y_counts = y_raw.value_counts()
    rare_classes = y_counts[y_counts < 2].index
    if len(rare_classes) > 0:
        y_raw = y_raw.where(~y_raw.isin(rare_classes), "Rare_Other")

    feature_cols = [
        "Temp_C", "Dew Point Temp_C", "Rel Hum_%", "Wind Speed_km/h",
        "Visibility_km", "Press_kPa", "Year", "Month", "Day", "Hour",
        "Weekday", "IsWeekend", "Hour_sin", "Hour_cos"
    ]

    missing_features = [c for c in feature_cols if c not in work_df.columns]
    if missing_features:
        raise KeyError(f"缺少特征列: {missing_features}")

    X = work_df[feature_cols].copy()

    # 时间解析失败时可能出现缺失，使用中位数填补
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # 编码目标变量
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    label_map = pd.DataFrame({
        "encoded_label": np.arange(len(le.classes_)),
        "weather_class": le.classes_,
    })
    label_map.to_csv(report_dir / "target_label_mapping.csv", index=False, encoding="utf-8-sig")

    # 优先进行分层划分；若类别分布仍不满足条件，则回退到普通随机划分。
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        split_mode = "stratified"
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        split_mode = "random_no_stratify"

    # 标准化（仅基于训练集拟合，避免数据泄漏）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

    y_train_df = pd.DataFrame({"y": y_train}, index=X_train.index)
    y_test_df = pd.DataFrame({"y": y_test}, index=X_test.index)

    X_train_scaled_df.to_csv(data_dir / "X_train_scaled.csv", index=False, encoding="utf-8-sig")
    X_test_scaled_df.to_csv(data_dir / "X_test_scaled.csv", index=False, encoding="utf-8-sig")
    y_train_df.to_csv(data_dir / "y_train.csv", index=False, encoding="utf-8-sig")
    y_test_df.to_csv(data_dir / "y_test.csv", index=False, encoding="utf-8-sig")

    # 同时保存未缩放版本，便于对比
    X_train.to_csv(data_dir / "X_train_raw.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(data_dir / "X_test_raw.csv", index=False, encoding="utf-8-sig")

    split_info = pd.DataFrame([
        {"item": "X_shape", "value": str(X.shape)},
        {"item": "X_train_shape", "value": str(X_train.shape)},
        {"item": "X_test_shape", "value": str(X_test.shape)},
        {"item": "y_class_count", "value": str(len(le.classes_))},
        {"item": "split_mode", "value": split_mode},
    ])
    split_info.to_csv(report_dir / "train_test_split_info.csv", index=False, encoding="utf-8-sig")

    print(f"特征矩阵形状: {X.shape}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"目标类别数: {len(le.classes_)}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_dirs(base_dir)

    df = load_data(base_dir)
    descriptive_statistics(df, dirs["report"])
    data_type_analysis(df, dirs["report"])
    missing_value_analysis(df, dirs["report"])

    df_clean = detect_and_treat_outliers(df, dirs["report"])
    exploratory_data_analysis(df_clean, dirs["fig"])
    feature_engineering_and_split(df_clean, dirs["data"], dirs["report"])

    print("\n全部流程执行完成。输出目录:")
    print(dirs["output"])


if __name__ == "__main__":
    main()


