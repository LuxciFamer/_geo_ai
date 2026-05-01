import json
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def ensure_dirs(base_dir: Path) -> dict:
    output_dir = base_dir / "outputs" / "experiment6"
    report_dir = output_dir / "reports"
    prediction_dir = output_dir / "predictions"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"

    for d in [output_dir, report_dir, prediction_dir, model_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "output": output_dir,
        "report": report_dir,
        "prediction": prediction_dir,
        "model": model_dir,
        "figure": figure_dir,
    }


def load_experiment5_data(base_dir: Path):
    exp5_dir = base_dir / "outputs" / "experiment5"
    processed_dir = exp5_dir / "processed"
    report_dir = exp5_dir / "reports"

    X_train = torch.load(processed_dir / "X_train.pt", map_location="cpu")
    X_test = torch.load(processed_dir / "X_test.pt", map_location="cpu")
    y_train = torch.load(processed_dir / "y_train.pt", map_location="cpu")
    y_test = torch.load(processed_dir / "y_test.pt", map_location="cpu")

    feature_names_path = report_dir / "feature_names.csv"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"未找到特征名文件: {feature_names_path}")
    with open(feature_names_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        feature_names = [row["feature_name"] for row in reader]

    print(f"已加载实验5数据: {exp5_dir}")
    print(f"X_train: {tuple(X_train.shape)}")
    print(f"X_test: {tuple(X_test.shape)}")
    print(f"y_train: {tuple(y_train.shape)}")
    print(f"y_test: {tuple(y_test.shape)}")
    print(f"特征数量: {len(feature_names)}")

    return X_train, X_test, y_train, y_test, feature_names


def to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


def prepare_split(X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.2):
    y_np = y.cpu().numpy().astype(int)
    idx = np.arange(len(y_np))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=42,
        stratify=y_np,
    )
    return train_idx, val_idx


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        logits = model(xb)
        loss = criterion(logits, yb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        total_loss += loss.item() * len(xb)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(yb.long().cpu().numpy().tolist())

    avg_loss = total_loss / len(all_targets)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_targets, all_probs)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "probs": np.array(all_probs),
        "preds": np.array(all_preds),
        "targets": np.array(all_targets),
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience):
    history = []
    best_state = None
    best_metric = -np.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_targets = []
        train_probs = []
        train_preds = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            running_loss += loss.item() * len(xb)
            train_probs.extend(probs.detach().cpu().numpy().tolist())
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_targets.extend(yb.long().detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_targets)
        train_metrics = {
            "accuracy": accuracy_score(train_targets, train_preds),
            "precision": precision_score(train_targets, train_preds, zero_division=0),
            "recall": recall_score(train_targets, train_preds, zero_division=0),
            "f1": f1_score(train_targets, train_preds, zero_division=0),
            "roc_auc": roc_auc_score(train_targets, train_probs),
        }

        val_metrics = evaluate_model(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(row)

        improved = val_metrics["f1"] > best_metric + 1e-6
        if improved:
            best_metric = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if patience_counter >= patience:
            print(f"验证集指标连续 {patience} 轮未提升，提前停止。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return pd.DataFrame(history), best_epoch, best_metric


def save_outputs(
    model: nn.Module,
    history_df: pd.DataFrame,
    test_metrics: dict,
    y_test_tensor: torch.Tensor,
    test_probs: np.ndarray,
    test_preds: np.ndarray,
    feature_names: list,
    dirs: dict,
    config: dict,
):
    report_dir = dirs["report"]
    pred_dir = dirs["prediction"]
    model_dir = dirs["model"]
    figure_dir = dirs["figure"]

    history_df.to_csv(report_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame([
        {
            "dataset": "test",
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"],
            "loss": test_metrics["loss"],
        }
    ])
    metrics_df.to_csv(report_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(test_metrics["targets"], test_preds)
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv(report_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    report_text = classification_report(
        test_metrics["targets"], test_preds, target_names=["No Rain", "Rain"], zero_division=0
    )
    with open(report_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("实验6：基于PyTorch的逻辑回归降水预测\n")
        f.write("\n分类报告:\n")
        f.write(str(report_text))
        f.write("\n\n测试集指标:\n")
        f.write(metrics_df.to_string(index=False))

    pred_df = pd.DataFrame({
        "Actual": test_metrics["targets"],
        "Predicted": test_preds,
        "Predicted_Prob_Rain": test_probs,
    })
    pred_df.to_csv(pred_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    torch.save(model.state_dict(), model_dir / "best_model.pt")
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(report_dir / "feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))

    with open(report_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("实验6：基于PyTorch的逻辑回归降水预测\n")
        f.write(f"特征维度: {len(feature_names)}\n")
        f.write(f"最佳验证F1: {history_df['val_f1'].max():.4f}\n")
        f.write(f"最终测试AUC: {test_metrics['roc_auc']:.4f}\n")
        f.write(f"最终测试F1: {test_metrics['f1']:.4f}\n")
        f.write(f"最终测试Accuracy: {test_metrics['accuracy']:.4f}\n")

    # 曲线图
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "loss_curve.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(test_metrics["targets"], test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_dir / "roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5.5, 4.5))
    sns = __import__("seaborn")
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(test_probs, bins=40, kde=True)
    plt.xlabel("Predicted Probability of Rain")
    plt.title("Test Probability Distribution")
    plt.tight_layout()
    plt.savefig(figure_dir / "probability_distribution.png", dpi=150)
    plt.close()

    print("\n===== 测试集评估结果 =====")
    print(metrics_df.to_string(index=False))
    print("\n混淆矩阵:")
    print(cm_df)
    print(f"\n模型已保存到: {model_dir / 'best_model.pt'}")
    print(f"预测结果已保存到: {pred_dir}")
    print(f"报告已保存到: {report_dir}")
    print(f"图表已保存到: {figure_dir}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_dirs(base_dir)

    X_train_full, X_test, y_train_full, y_test, feature_names = load_experiment5_data(base_dir)

    X_train_full = to_float_tensor(X_train_full)
    X_test = to_float_tensor(X_test)
    y_train_full = y_train_full.long()
    y_test = y_test.long()

    train_idx, val_idx = prepare_split(X_train_full, y_train_full, val_ratio=0.2)
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    print(f"训练集切分后: train={len(train_idx)}, val={len(val_idx)}")
    print(f"测试集大小: {len(X_test)}")

    batch_size = 256
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = LogisticRegressionModel(input_dim=X_train.shape[1]).to(device)

    pos = y_train.sum().item()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    config = {
        "batch_size": batch_size,
        "epochs": 30,
        "patience": 5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": float(pos_weight.item()),
        "input_dim": int(X_train.shape[1]),
        "device": str(device),
    }

    history_df, best_epoch, best_val_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        patience=config["patience"],
    )

    print(f"最佳验证F1对应的epoch: {best_epoch}, best_val_f1={best_val_f1:.4f}")

    test_metrics = evaluate_model(model, test_loader, criterion, device)
    test_probs = np.asarray(test_metrics["probs"], dtype=np.float64)
    test_preds = np.asarray(test_metrics["preds"], dtype=np.int64)

    save_outputs(
        model=model,
        history_df=history_df,
        test_metrics=test_metrics,
        y_test_tensor=y_test,
        test_probs=test_probs,
        test_preds=test_preds,
        feature_names=feature_names,
        dirs=dirs,
        config=config,
    )

    print("\n实验6运行完成。")
    print(f"输出目录: {dirs['output']}")


if __name__ == "__main__":
    main()




