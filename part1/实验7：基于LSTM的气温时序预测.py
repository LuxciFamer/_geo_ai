"""
实验7：基于 LSTM 深度神经网络的气温时序预测
==============================================
创新点：
  1. 使用 LSTM 捕捉气象数据中的时间序列长期依赖关系
  2. 滑动窗口序列构建，将逐小时数据转换为有监督序列
  3. ReduceLROnPlateau 学习率调度 + Early Stopping
  4. 与实验三线性回归形成对比，展示深度学习在回归任务上的优势
"""

import warnings
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "当前环境未安装 PyTorch，请先执行 `pip install torch` 后再运行此脚本。"
    ) from exc

warnings.filterwarnings("ignore")


# ──────────────────────────── 目录管理 ────────────────────────────

def ensure_dirs(base_dir: Path) -> dict:
    output_dir = base_dir / "outputs" / "experiment7"
    report_dir = output_dir / "reports"
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    prediction_dir = output_dir / "predictions"

    for d in [output_dir, report_dir, figure_dir, model_dir, prediction_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "output": output_dir,
        "report": report_dir,
        "figure": figure_dir,
        "model": model_dir,
        "prediction": prediction_dir,
    }


# ──────────────────────────── 数据加载 ────────────────────────────

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
    raise FileNotFoundError("未找到 WeatherData.csv 或 Weather_Data.csv")


def load_data(base_dir: Path) -> pd.DataFrame:
    data_path = find_data_file(base_dir)
    df = cast(pd.DataFrame, pd.read_csv(data_path, encoding="utf-8-sig"))  # type: ignore[call-overload]
    print(f"已加载数据: {data_path}")
    print(f"原始数据形状: {df.shape}")
    return df


# ──────────────────────────── 特征工程 ────────────────────────────

def build_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """构建时间特征并按时间排序。"""
    required = ["Date/Time", "Temp_C"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少必要列: {missing}")

    work = df.copy()
    work["Date/Time"] = pd.to_datetime(work["Date/Time"], errors="coerce")
    work = work.dropna(subset=["Date/Time", "Temp_C"]).sort_values("Date/Time").reset_index(drop=True)

    dt = work["Date/Time"]
    work["Hour"] = dt.dt.hour
    work["DayOfYear"] = dt.dt.dayofyear
    work["Month"] = dt.dt.month
    work["Hour_sin"] = np.sin(2 * np.pi * work["Hour"] / 24.0)
    work["Hour_cos"] = np.cos(2 * np.pi * work["Hour"] / 24.0)
    work["DayOfYear_sin"] = np.sin(2 * np.pi * work["DayOfYear"] / 365.25)
    work["DayOfYear_cos"] = np.cos(2 * np.pi * work["DayOfYear"] / 365.25)

    feature_cols = [
        "Temp_C", "Dew Point Temp_C", "Rel Hum_%", "Wind Speed_km/h",
        "Visibility_km", "Press_kPa",
        "Hour_sin", "Hour_cos", "DayOfYear_sin", "DayOfYear_cos",
    ]
    available = [c for c in feature_cols if c in work.columns]

    target = work["Temp_C"].astype(float)
    timestamps = work["Date/Time"]
    features = work[available].copy()

    # 用中位数填补缺失
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True))

    return features, target, timestamps


# ──────────────────────────── 滑动窗口 ────────────────────────────

class TimeSeriesDataset(Dataset):
    """滑动窗口时间序列数据集。"""
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int):
        if window_size <= 0:
            raise ValueError(f"window_size 必须为正整数，当前为 {window_size}")
        if len(X) != len(y):
            raise ValueError(f"特征与目标长度不一致: X={len(X)}, y={len(y)}")
        if len(X) <= window_size:
            raise ValueError(
                f"序列长度不足以构造滑动窗口: 样本数={len(X)}, window_size={window_size}"
            )

        self.X_windows = []
        self.y_targets = []
        for i in range(len(X) - window_size):
            self.X_windows.append(X[i : i + window_size])
            self.y_targets.append(y[i + window_size])
        self.X_windows = np.array(self.X_windows, dtype=np.float32)
        self.y_targets = np.array(self.y_targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y_targets)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X_windows[idx], dtype=torch.float32),
            torch.tensor(self.y_targets[idx], dtype=torch.float32),
        )


def build_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算回归指标；在样本过少时返回可用的 NaN 保护值。"""
    if len(y_true) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    mae = float(np.asarray(mean_absolute_error(y_true, y_pred)).reshape(-1)[0])
    rmse = float(np.asarray(np.sqrt(mean_squared_error(y_true, y_pred))).reshape(-1)[0])
    r2 = float(np.asarray(r2_score(y_true, y_pred)).reshape(-1)[0]) if len(y_true) >= 2 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ──────────────────────────── LSTM 模型 ────────────────────────────

class LSTMTemperaturePredictor(nn.Module):
    """
    双层 LSTM + 全连接输出层，用于气温回归预测。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]     # 取最后一个时间步
        return self.fc(last_hidden).squeeze(1)


# ──────────────────────────── 训练引擎 ────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
):
    if len(train_loader) == 0:
        raise ValueError("训练集窗口数量为 0，请检查切分比例或 window_size。")
    if len(val_loader) == 0:
        raise ValueError("验证集窗口数量为 0，请检查切分比例或 window_size。")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    lr_history = []

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        running_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * len(xb)
            n_train += len(xb)
        if n_train == 0:
            raise ValueError("训练阶段未获得任何样本，请检查 DataLoader 或数据集构造。")
        train_loss = running_loss / n_train

        # ---- 验证 ----
        model.eval()
        val_loss_total = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_total += criterion(pred, yb).item() * len(xb)
                n_val += len(xb)
        if n_val == 0:
            raise ValueError("验证阶段未获得任何样本，请检查 DataLoader 或数据集构造。")
        val_loss = val_loss_total / n_val

        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            tag = " ★" if improved else ""
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} | "
                f"lr={current_lr:.2e}{tag}"
            )

        if patience_counter >= patience:
            print(f"验证损失连续 {patience} 轮未改善，提前停止 (best epoch 约 {epoch - patience})。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return pd.DataFrame(history)


# ──────────────────────────── 评估与保存 ────────────────────────────

@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device):
    if len(loader) == 0:
        raise ValueError("预测数据集为空，无法生成预测结果。")

    model.eval()
    preds_list, targets_list = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds_list.append(model(xb).cpu().numpy())
        targets_list.append(yb.numpy())
    return np.concatenate(preds_list), np.concatenate(targets_list)


def evaluate_and_save(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    history_df: pd.DataFrame,
    target_scaler: StandardScaler,
    timestamps_test: np.ndarray,
    dirs: dict,
    device: torch.device,
    window_size: int,
):
    report_dir = dirs["report"]
    figure_dir = dirs["figure"]
    model_dir = dirs["model"]
    prediction_dir = dirs["prediction"]

    # 预测（在标准化空间下，然后反变换）
    train_preds_scaled, train_targets_scaled = predict_all(model, train_loader, device)
    test_preds_scaled, test_targets_scaled = predict_all(model, test_loader, device)

    if len(test_preds_scaled) != len(timestamps_test):
        raise ValueError(
            f"测试集预测数与时间戳数不一致: preds={len(test_preds_scaled)}, timestamps={len(timestamps_test)}"
        )

    train_preds = target_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).ravel()
    train_targets = target_scaler.inverse_transform(train_targets_scaled.reshape(-1, 1)).ravel()
    test_preds = target_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).ravel()
    test_targets = target_scaler.inverse_transform(test_targets_scaled.reshape(-1, 1)).ravel()

    # 指标计算
    metrics = pd.DataFrame([
        {"dataset": "train", **build_regression_metrics(train_targets, train_preds)},
        {"dataset": "test", **build_regression_metrics(test_targets, test_preds)},
    ])
    metrics.to_csv(report_dir / "lstm_metrics.csv", index=False, encoding="utf-8-sig")
    history_df.to_csv(report_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    # 预测结果
    pred_df = pd.DataFrame({
        "Date/Time": timestamps_test[:len(test_preds)],
        "Actual_Temp_C": test_targets,
        "Predicted_Temp_C": test_preds,
        "Residual": test_targets - test_preds,
    })
    pred_df.to_csv(prediction_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 文字摘要
    with open(report_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write("实验7：基于 LSTM 深度神经网络的气温时序预测\n")
        f.write(f"滑动窗口大小: {window_size}\n")
        f.write(f"训练样本数: {len(train_targets)}\n")
        f.write(f"测试样本数: {len(test_targets)}\n")
        f.write(f"\n{'='*40}\n")
        for _, row in metrics.iterrows():
            f.write(f"{row['dataset']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R²={row['R2']:.4f}\n")

    # 保存模型权重
    torch.save(model.state_dict(), model_dir / "lstm_best_model.pt")

    # ─── 图1：训练/验证损失曲线 ───
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_train = "#4361ee"
    color_val = "#f72585"

    ax1.plot(history_df["epoch"], history_df["train_loss"], color=color_train, linewidth=1.8, label="Train Loss")
    ax1.plot(history_df["epoch"], history_df["val_loss"], color=color_val, linewidth=1.8, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("LSTM Training & Validation Loss", fontsize=14, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["lr"], color="#06d6a0", linewidth=1.2, linestyle="--", alpha=0.7, label="Learning Rate")
    ax2.set_ylabel("Learning Rate", fontsize=12, color="#06d6a0")
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(figure_dir / "loss_and_lr_curve.png", dpi=150)
    plt.close()

    # ─── 图2：实际 vs 预测散点图 ───
    plt.figure(figsize=(7, 7))
    plt.scatter(test_targets, test_preds, s=12, alpha=0.4, c="#4361ee", edgecolors="none")
    min_v = min(test_targets.min(), test_preds.min())
    max_v = max(test_targets.max(), test_preds.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5)
    plt.xlabel("Actual Temp (°C)", fontsize=12)
    plt.ylabel("Predicted Temp (°C)", fontsize=12)
    plt.title("LSTM: Actual vs Predicted Temperature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figure_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # ─── 图3：时间序列对比（前 500 点）───
    plot_n = min(500, len(pred_df))
    plt.figure(figsize=(14, 5))
    plt.plot(range(plot_n), pred_df["Actual_Temp_C"].iloc[:plot_n].values,
             label="Actual", linewidth=1.4, color="#4361ee")
    plt.plot(range(plot_n), pred_df["Predicted_Temp_C"].iloc[:plot_n].values,
             label="LSTM Predicted", linewidth=1.0, color="#f72585", alpha=0.85)
    plt.fill_between(range(plot_n),
                      pred_df["Actual_Temp_C"].iloc[:plot_n].values,
                      pred_df["Predicted_Temp_C"].iloc[:plot_n].values,
                      alpha=0.15, color="#f72585")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.title("LSTM Temperature Prediction — Test Set (First 500 Points)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figure_dir / "time_series_comparison.png", dpi=150)
    plt.close()

    # ─── 图4：残差分布 ───
    plt.figure(figsize=(8, 5))
    residuals = test_targets - test_preds
    plt.hist(residuals, bins=50, color="#4361ee", alpha=0.75, edgecolor="white", linewidth=0.5)
    plt.axvline(0, color="#f72585", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (°C)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("LSTM Residual Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figure_dir / "residual_distribution.png", dpi=150)
    plt.close()

    print("\n===== LSTM 模型评估结果 =====")
    print(metrics.to_string(index=False))
    print(f"\n详细结果已保存到: {report_dir}")
    print(f"图表已保存到: {figure_dir}")
    print(f"模型已保存到: {model_dir}")


# ──────────────────────────── 主流程 ────────────────────────────

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_dirs(base_dir)

    df = load_data(base_dir)
    features, target, timestamps = build_time_features(df)

    print(f"特征矩阵形状: {features.shape}")
    print(f"目标变量形状: {target.shape}")
    print(f"时间范围: {timestamps.iloc[0]} ~ {timestamps.iloc[-1]}")

    # ── 按时间切分：60% 训练 / 20% 验证 / 20% 测试 ──
    n = len(features)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            f"时间切分结果不合法: n={n}, train_end={train_end}, val_end={val_end}"
        )

    X_all = features.values
    y_all = target.values

    # ── 标准化（仅在训练集上拟合）──
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_raw = X_all[:train_end]
    y_train_raw = y_all[:train_end]

    feature_scaler.fit(X_train_raw)
    target_scaler.fit(y_train_raw.reshape(-1, 1))

    X_scaled = feature_scaler.transform(X_all)
    y_scaled = target_scaler.transform(y_all.reshape(-1, 1)).ravel()

    # ── 滑动窗口构建 ──
    window_size = 24  # 24 小时滑动窗口

    if min(train_end, val_end - train_end, n - val_end) <= window_size:
        raise ValueError(
            f"任一数据分割段长度必须大于 window_size={window_size}，当前切分为 "
            f"train={train_end}, val={val_end - train_end}, test={n - val_end}"
        )

    train_dataset = TimeSeriesDataset(X_scaled[:train_end], y_scaled[:train_end], window_size)
    val_dataset = TimeSeriesDataset(X_scaled[train_end:val_end], y_scaled[train_end:val_end], window_size)
    test_dataset = TimeSeriesDataset(X_scaled[val_end:], y_scaled[val_end:], window_size)

    print(f"\n滑动窗口大小: {window_size}")
    print(f"训练集序列数: {len(train_dataset)}")
    print(f"验证集序列数: {len(val_dataset)}")
    print(f"测试集序列数: {len(test_dataset)}")

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── 模型 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    input_dim = X_scaled.shape[1]
    model = LSTMTemperaturePredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # ── 训练 ──
    print("\n开始 LSTM 训练...")
    history_df = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        lr=1e-3,
        patience=10,
    )

    # ── 评估 ──
    timestamps_test = timestamps.iloc[val_end + window_size : val_end + window_size + len(test_dataset)].to_numpy()
    evaluate_and_save(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        history_df=history_df,
        target_scaler=target_scaler,
        timestamps_test=timestamps_test,
        dirs=dirs,
        device=device,
        window_size=window_size,
    )

    print("\n实验7运行完成。")
    print(f"输出目录: {dirs['output']}")


if __name__ == "__main__":
    main()
