import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "当前环境未安装 PyTorch，请先执行 `pip install torch` 后再运行此脚本。"
    ) from exc

warnings.filterwarnings("ignore")


@dataclass
class PipelineArtifacts:
    data_path: Path
    output_dir: Path
    report_dir: Path
    processed_dir: Path
    train_size: int
    test_size: int
    feature_dim: int
    class_distribution: dict


class WeatherAUSDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        if len(features) != len(labels):
            raise ValueError("features 和 labels 的样本数必须一致")
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def ensure_dirs(base_dir: Path) -> dict:
    output_dir = base_dir / "outputs" / "experiment5"
    report_dir = output_dir / "reports"
    processed_dir = output_dir / "processed"
    for d in [output_dir, report_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"output": output_dir, "report": report_dir, "processed": processed_dir}


def find_data_file(base_dir: Path) -> Path:
    candidates = [
        base_dir / "weatherAUS.csv",
        base_dir / "WeatherData.csv",
        base_dir / "Weather_Data.csv",
        base_dir.parent / "weatherAUS.csv",
        base_dir.parent / "WeatherData.csv",
        base_dir.parent / "Weather_Data.csv",
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


def normalize_target(series: pd.Series) -> pd.Series:
    mapping = {"Yes": 1, "No": 0, "Y": 1, "N": 0, True: 1, False: 0}
    cleaned = series.astype(str).str.strip()
    return cleaned.map(mapping)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required_cols = ["Date", "RainTomorrow"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise KeyError(f"数据缺少必要列: {missing_required}")

    work_df = df.copy()
    work_df["Date"] = pd.to_datetime(work_df["Date"], errors="coerce")
    work_df = work_df.dropna(subset=["Date", "RainTomorrow"]).reset_index(drop=True)

    y = normalize_target(work_df["RainTomorrow"])
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

    # 保留当天气象信息和日期派生特征，不直接使用目标列。
    drop_cols = ["RainTomorrow", "Date"]
    X = work_df.drop(columns=drop_cols)

    # 将明显的二值类别统一为字符串，交由类别预处理器处理。
    if "RainToday" in X.columns:
        X["RainToday"] = X["RainToday"].astype(str).str.strip()

    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def to_tensor_dataset(X_array: np.ndarray, y_array: np.ndarray) -> WeatherAUSDataset:
    features = torch.tensor(X_array, dtype=torch.float32)
    labels = torch.tensor(y_array, dtype=torch.long)
    return WeatherAUSDataset(features, labels)


def save_artifacts(
    artifacts: PipelineArtifacts,
    feature_names: Iterable[str],
    preprocessor: ColumnTransformer,
    train_dataset: WeatherAUSDataset,
    test_dataset: WeatherAUSDataset,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    processed_dir = artifacts.processed_dir
    report_dir = artifacts.report_dir

    torch.save(train_dataset.features, processed_dir / "X_train.pt")
    torch.save(test_dataset.features, processed_dir / "X_test.pt")
    torch.save(train_dataset.labels, processed_dir / "y_train.pt")
    torch.save(test_dataset.labels, processed_dir / "y_test.pt")

    pd.DataFrame({"feature_name": list(feature_names)}).to_csv(
        report_dir / "feature_names.csv", index=False, encoding="utf-8-sig"
    )

    split_summary = pd.DataFrame([
        {"item": "train_size", "value": artifacts.train_size},
        {"item": "test_size", "value": artifacts.test_size},
        {"item": "feature_dim", "value": artifacts.feature_dim},
        {"item": "class_0_count", "value": artifacts.class_distribution.get(0, 0)},
        {"item": "class_1_count", "value": artifacts.class_distribution.get(1, 0)},
        {"item": "train_batches", "value": len(train_loader)},
        {"item": "test_batches", "value": len(test_loader)},
    ])
    split_summary.to_csv(report_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    with open(report_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(artifacts.data_path),
                "train_size": artifacts.train_size,
                "test_size": artifacts.test_size,
                "feature_dim": artifacts.feature_dim,
                "class_distribution": artifacts.class_distribution,
                "preprocessor": preprocessor.get_params(deep=False),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    with open(report_dir / "pipeline_summary.txt", "w", encoding="utf-8") as f:
        f.write("实验5：降水预测数据流水线构建\n")
        f.write(f"数据文件: {artifacts.data_path}\n")
        f.write(f"训练集大小: {artifacts.train_size}\n")
        f.write(f"测试集大小: {artifacts.test_size}\n")
        f.write(f"特征维度: {artifacts.feature_dim}\n")
        f.write(f"类别分布: {artifacts.class_distribution}\n")
        f.write(f"训练批次数: {len(train_loader)}\n")
        f.write(f"测试批次数: {len(test_loader)}\n")


def preview_loader(loader: DataLoader, name: str) -> None:
    batch_x, batch_y = next(iter(loader))
    print(f"{name} 首个批次特征形状: {tuple(batch_x.shape)}")
    print(f"{name} 首个批次标签形状: {tuple(batch_y.shape)}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_dirs(base_dir)

    df = load_data(base_dir)
    X, y = engineer_features(df)

    print(f"特征矩阵形状: {X.shape}")
    print(f"目标分布:\n{y.value_counts(normalize=True).rename('ratio')}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_dataset = to_tensor_dataset(X_train_processed, y_train.to_numpy())
    test_dataset = to_tensor_dataset(X_test_processed, y_test.to_numpy())

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_distribution = y.value_counts().sort_index().to_dict()
    artifacts = PipelineArtifacts(
        data_path=find_data_file(base_dir),
        output_dir=dirs["output"],
        report_dir=dirs["report"],
        processed_dir=dirs["processed"],
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        feature_dim=train_dataset.features.shape[1],
        class_distribution=class_distribution,
    )

    save_artifacts(
        artifacts=artifacts,
        feature_names=feature_names,
        preprocessor=preprocessor,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    preview_loader(train_loader, "训练集 DataLoader")
    preview_loader(test_loader, "测试集 DataLoader")

    print("\n实验5运行完成。")
    print(f"输出目录: {artifacts.output_dir}")


if __name__ == "__main__":
    main()


