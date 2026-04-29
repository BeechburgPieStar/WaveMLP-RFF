import argparse
import os
import random
import inspect
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from utils.load_data import load_single_dataset
from backbones.MyModel import MyModel


def parse_args():
    parser = argparse.ArgumentParser("MyModel")

    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=None)

    parser.add_argument('--patience_early_stop', type=int, default=10)
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--dataset_name', type=str, default="ManySig")
    parser.add_argument('--use_eq', action='store_true')
    parser.add_argument('--exp', type=str, default="CRD", choices=["CRD", "CR"])
    parser.add_argument('--train_date', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--all_test_round', type=int, default=4)
    parser.add_argument('--test_round', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2023)

    # ========== Wavelet reconstruction branch ==========
    parser.add_argument('--use_wavelet', type=int, default=1)
    parser.add_argument('--wavelet_levels', type=int, default=4)

    parser.add_argument('--wavelet_init', type=str, default='randn',
                        choices=['haar', 'randn', 'db2', 'sym2', 'db4', 'sym4', 'sinc'])

    parser.add_argument('--wavelet_learnable', type=int, default=1)

    parser.add_argument(
        '--wavelet_kernel_size',
        type=int,
        default=16,
        help=(
            "Wavelet conv kernel size. "
            "If <= 0, it will be inferred from wavelet_init. "
            "Suggested mapping: haar=2, db2 or sym2=4, db4 or sym4=8, randn or sinc=8. "
            "You can override it manually, for example 16 or 32 for a longer filter."
        )
    )

    parser.add_argument('--wave_lr_mult', type=float, default=1.0)

    parser.add_argument('--code_state', type=str, default="only_test",
                        choices=["only_train", "only_test", "train_test"])
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_receivers(all_num=12, all_test_round=4, test_round=0):
    if not (0 <= test_round < all_test_round):
        raise ValueError(f"test_round {test_round} not in [0, {all_test_round - 1}]")
    if all_num % all_test_round != 0:
        raise ValueError(f"total rx num {all_num} not divisible by rounds {all_test_round}")

    receivers = list(range(all_num))
    per_round = all_num // all_test_round
    start = test_round * per_round
    end = all_num if test_round == all_test_round - 1 else start + per_round
    test = receivers[start:end]
    train = [r for r in receivers if r not in test]
    return train, test


def prepare_dataset(dataset_name, rx_indexes, date_indexes, tx_num, is_train, seed, use_eq=False):
    x_all = []
    y_all = []

    sig_type = 'equalized' if use_eq else 'non_equalized'

    for rx_index in rx_indexes:
        for date_index in date_indexes:
            x, y = load_single_dataset(dataset_name, rx_index, date_index, tx_num, sig_type)
            x_all.append(x)
            y_all.append(y)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    if is_train:
        indices = np.arange(len(y_all))
        train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=seed)

        x_train, x_val = x_all[train_idx, :, :], x_all[val_idx, :, :]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        return (x_train, y_train), (x_val, y_val)

    return x_all, y_all


def _to_device(data, target, device):
    data = data.to(device).float()
    target = target.to(device).long()
    return data, target


def train_epoch(model, criterion, train_loader, optimizer, epoch, device, grad_clip=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in train_loader:
        data, target = _to_device(data, target, device)

        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(data)
        loss = criterion(logits, target)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"Train Epoch: {epoch}\tLoss: {avg_loss:.6f}, Acc: {correct}/{total} ({acc:.2f}%)")
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(model, criterion, val_loader, epoch, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in val_loader:
        data, target = _to_device(data, target, device)
        logits, _, _ = model(data)
        loss = criterion(logits, target)

        total_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"\nValidation set: Loss: {avg_loss:.4f}, Acc: {correct}/{total} ({acc:.2f}%)\n")
    return avg_loss, acc


@torch.no_grad()
def test_epoch(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    for data, target in test_loader:
        data, target = _to_device(data, target, device)
        logits, _, _ = model(data)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    acc = correct / max(total, 1)
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def _build_optimizer_with_wave_groups(model, base_lr, weight_decay, wave_lr_mult):
    wave_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # MyModel(use_wavelet=1) will contain parameters named like "wavelet.waves..."
        if name.startswith("wavelet.") or ".wavelet." in name:
            wave_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append(
            {"params": other_params, "lr": float(base_lr), "weight_decay": float(weight_decay)}
        )
    if wave_params:
        param_groups.append(
            {"params": wave_params, "lr": float(base_lr) * float(wave_lr_mult), "weight_decay": float(weight_decay)}
        )

    return torch.optim.Adam(param_groups)


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    epochs,
    save_path,
    lr=1e-3,
    weight_decay=0.0,
    patience_early_stop=10,
    scheduler_patience=5,
    min_lr=1e-6,
    grad_clip=None,
    device="cuda",
    wave_lr_mult=1.0,
):
    device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    model = model.to(device)

    optimizer = _build_optimizer_with_wave_groups(
        model, base_lr=lr, weight_decay=weight_decay, wave_lr_mult=wave_lr_mult
    )
    criterion = nn.CrossEntropyLoss().to(device)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=scheduler_patience,
        verbose=True,
        min_lr=min_lr
    )

    best_val_loss = float('inf')
    no_improve = 0

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_epoch(model, criterion, train_loader, optimizer, epoch, device, grad_clip=grad_clip)
        val_loss, _ = evaluate_epoch(model, criterion, val_loader, epoch, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            print(f"Validation loss improved {best_val_loss:.6f} -> {val_loss:.6f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s).")

        if no_improve >= patience_early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

        print("------------------------------------------------")

    return best_val_loss


def _bind_kernel_size(wavelet_init: str) -> int:
    w = str(wavelet_init).lower()
    if w == "haar":
        return 2
    if w in ["db2", "sym2"]:
        return 4
    if w in ["db4", "sym4"]:
        return 8
    if w in ["sinc", "randn"]:
        return 8
    raise ValueError(f"Unknown wavelet_init: {wavelet_init}")


def _build_save_name(conf, tx_num: int, wavelet_kernel_size: int) -> str:
    train_dates_str = "_".join(map(str, conf.train_date))
    eq_tag = "eq" if conf.use_eq else "noeq"

    wave_tag = (
        f"wave{int(conf.use_wavelet)}"
        f"_lvl{int(conf.wavelet_levels)}"
        f"_learn{int(conf.wavelet_learnable)}"
        f"_{conf.wavelet_init}"
        f"_k{int(wavelet_kernel_size)}"
        f"_wlrm{conf.wave_lr_mult}"
    )

    return (
        f"{conf.dataset_name}_"
        f"{conf.exp}_"
        f"{eq_tag}_"
        f"tx{tx_num}_"
        f"{wave_tag}_"
        f"date{train_dates_str}_"
        f"round{conf.test_round}_"
        f"seed{conf.seed}.pth"
    )

def _filter_kwargs_for_class(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    out = {}
    for k, v in kwargs.items():
        if k in valid:
            out[k] = v
    return out


def _build_model_from_conf(conf, tx_num: int, wavelet_kernel_size: int) -> MyModel:
    raw_kwargs = dict(
        num_classes=tx_num,
        use_wavelet=bool(conf.use_wavelet),
        wavelet_levels=int(conf.wavelet_levels),
        wavelet_learnable=bool(conf.wavelet_learnable),
        wavelet_init=str(conf.wavelet_init),
        wavelet_kernel_size=int(wavelet_kernel_size),
    )

    kwargs = _filter_kwargs_for_class(MyModel, raw_kwargs)
    return MyModel(**kwargs)

def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu
    setup_seed(conf.seed)

    if conf.dataset_name == "ManySig":
        tx_num, rx_num = 6, 12
        all_days = [1, 2, 3, 4]
    elif conf.dataset_name == "ManyRx":
        tx_num, rx_num = 10, 32
        all_days = [1, 2, 3, 4]
    else:
        raise ValueError(f"Unknown dataset_name: {conf.dataset_name}")

    if conf.exp == "CRD":
        if len(conf.train_date) < 1 or len(conf.train_date) >= len(all_days):
            raise ValueError("For CRD, train_date must be a proper subset of available days")

    if int(conf.wavelet_levels) < 0:
        raise ValueError("wavelet_levels must be >= 0")
    if int(conf.wavelet_levels) == 0:
        conf.use_wavelet = 0

    # 如果用户没有手动指定 kernel_size，则根据 wavelet_init 自动绑定一个建议值
    if int(conf.wavelet_kernel_size) <= 0:
        conf.wavelet_kernel_size = _bind_kernel_size(conf.wavelet_init)

    print(f"Wavelet init: {conf.wavelet_init}, kernel_size={conf.wavelet_kernel_size}")
    print(
        f"use_wavelet={int(conf.use_wavelet)}, "
        f"wavelet_levels={int(conf.wavelet_levels)}, "
        f"wavelet_learnable={int(conf.wavelet_learnable)}"
    )

    rx_train, rx_test = split_receivers(rx_num, conf.all_test_round, conf.test_round)
    print(f"Train receivers: {rx_train}")
    print(f"Test receivers:  {rx_test}")

    model = _build_model_from_conf(conf, tx_num=tx_num, wavelet_kernel_size=conf.wavelet_kernel_size)

    (x_train, y_train), (x_val, y_val) = prepare_dataset(
        conf.dataset_name,
        rx_train,
        conf.train_date,
        tx_num,
        True,
        conf.seed,
        use_eq=conf.use_eq
    )

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=conf.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=conf.batch_size,
        shuffle=False,
        drop_last=False
    )

    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    # 修正这里：原来 wavelet_kernel_size 变量未定义
    save_name = _build_save_name(conf, tx_num=tx_num, wavelet_kernel_size=conf.wavelet_kernel_size)
    save_path = os.path.join(save_dir, save_name)
    print(f"Save path: {save_path}")

    if conf.code_state in ["only_train", "train_test"]:
        train_and_evaluate(
            model,
            train_loader,
            val_loader,
            epochs=conf.epochs,
            save_path=save_path,
            lr=conf.lr,
            weight_decay=conf.wd,
            patience_early_stop=conf.patience_early_stop,
            scheduler_patience=conf.scheduler_patience,
            min_lr=conf.min_lr,
            grad_clip=conf.grad_clip,
            device="cuda",
            wave_lr_mult=conf.wave_lr_mult,
        )

    if conf.code_state in ["only_test", "train_test"]:
        model_test = _build_model_from_conf(conf, tx_num=tx_num, wavelet_kernel_size=conf.wavelet_kernel_size)
        state = torch.load(save_path, map_location="cpu")
        model_test.load_state_dict(state)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_test = model_test.to(device)

        if conf.exp == "CR":
            test_days = conf.train_date
        else:
            test_days = [d for d in all_days if d not in conf.train_date]
            if len(test_days) == 0:
                raise ValueError("CRD needs at least one unseen day for test")

        x_test, y_test = prepare_dataset(
            conf.dataset_name,
            rx_test,
            test_days,
            tx_num,
            False,
            conf.seed,
            use_eq=conf.use_eq
        )

        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=32,
            shuffle=False,
            drop_last=False
        )
        test_epoch(model_test, test_loader, device)


if __name__ == '__main__':
    main()