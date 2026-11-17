"""Train the spread U-Net on Next Day Wildfire Spread and save a checkpoint the spread agent can load.

    pip install -e ".[ml]" tfrecord
    # quick check (1 epoch, 512 samples)
    python scripts/train_spread.py --epochs 1 --max-train-samples 512
    # full run
    python scripts/train_spread.py --epochs 30

Inputs are datacube layer names (see firespread/io/ndws.py), so the trained model runs inside the
pipeline: set `agents.spread.model: unet` and `agents.spread.checkpoint: models/spread_unet.pt`.

The script reports the test metrics of the U-Net next to the persistence baseline
("tomorrow = today") and writes them into the checkpoint and into <out>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.config import resolve_path  # noqa: E402
from firespread.common.metrics import evaluate  # noqa: E402
from firespread.common.nn import build_unet, masked_focal_dice_loss, save_checkpoint  # noqa: E402
from firespread.io.ndws import LAYERS, load_split, select_layers  # noqa: E402

DEFAULT_INPUTS = list(LAYERS)
METRIC_KEYS = ("precision", "recall", "f1", "iou", "auc_pr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/raw/next_day_wildfire_spread", help="folder with the NDWS .tfrecord files")
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS), help=f"comma-separated layers, among {','.join(LAYERS)}")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--pretrained", action="store_true", help="ImageNet encoder weights (needs internet once)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=None, help="subsample the training split (quick tests)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/spread_unet.pt")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = [name.strip() for name in args.inputs.split(",") if name.strip()]
    data_dir = resolve_path(args.data)
    print(f"device: {device}{' (' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ''}")
    print(f"inputs: {inputs}")

    # --- data --------------------------------------------------------------------------------
    t0 = time.time()
    splits = {}
    for split in ("train", "eval", "test"):
        x, y = load_split(data_dir, split)
        splits[split] = (select_layers(x, inputs), y)
        if split == "test":
            persistence = np.nan_to_num(select_layers(x, ["fire"])[:, 0])  # baseline: tomorrow = today
        print(f"{split:5s}: {x.shape[0]:6d} tiles, fire pixels in target: {np.mean(y == 1):.4%}")
    print(f"data loaded in {time.time() - t0:.0f} s")

    x_train, y_train = splits["train"]
    if args.max_train_samples and args.max_train_samples < len(x_train):
        keep = rng.choice(len(x_train), args.max_train_samples, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]

    mean = np.nanmean(x_train, axis=(0, 2, 3))
    std = np.nanstd(x_train, axis=(0, 2, 3))
    std = np.where(std > 1e-6, std, 1.0)
    print("input statistics (train):")
    for i, name in enumerate(inputs):
        channel = x_train[:, i]
        lo, hi = np.nanpercentile(channel, [1, 99])
        print(f"  {name:10s} mean {mean[i]:9.3f}  std {std[i]:9.3f}  p1 {lo:9.3f}  p99 {hi:9.3f}  NaN {np.mean(np.isnan(channel)):.2%}")
    if not np.all(np.isfinite(mean) & np.isfinite(std)):
        raise ValueError("non-finite normalisation statistics: check the input layers above")

    def normalise(x: np.ndarray) -> torch.Tensor:
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        return torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float32))

    train_loader = DataLoader(
        TensorDataset(normalise(x_train), torch.from_numpy(y_train[:, None])),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    eval_x, eval_y = normalise(splits["eval"][0]), splits["eval"][1]
    test_x, test_y = normalise(splits["test"][0]), splits["test"][1]

    # wind components change sign when the tile is mirrored
    u_idx = inputs.index("wind_u") if "wind_u" in inputs else None
    v_idx = inputs.index("wind_v") if "wind_v" in inputs else None
    u_zero = 0.0 if u_idx is None else float(-mean[u_idx] / std[u_idx])  # normalised value of u = 0
    v_zero = 0.0 if v_idx is None else float(-mean[v_idx] / std[v_idx])

    def augment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:  # east <-> west
            x, y = x.flip(-1), y.flip(-1)
            if u_idx is not None:
                x[:, u_idx] = 2 * u_zero - x[:, u_idx]
        if torch.rand(()) < 0.5:  # north <-> south
            x, y = x.flip(-2), y.flip(-2)
            if v_idx is not None:
                x[:, v_idx] = 2 * v_zero - x[:, v_idx]
        return x, y

    # --- model -------------------------------------------------------------------------------
    model = build_unet(len(inputs), encoder=args.encoder, encoder_weights="imagenet" if args.pretrained else None).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    use_amp = device == "cuda"

    def predict(x: torch.Tensor) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
            for i in range(0, len(x), 256):
                out.append(torch.sigmoid(model(x[i : i + 256].to(device)).float()).cpu().numpy()[:, 0])
        return np.concatenate(out)

    # --- training ----------------------------------------------------------------------------
    out_path = resolve_path(args.out)
    best_auc, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for xb, yb in train_loader:
            xb, yb = augment(xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(xb)
            loss = masked_focal_dice_loss(logits.float(), yb)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.item())

        val = evaluate(predict(eval_x), eval_y)
        flag = ""
        if val["auc_pr"] > best_auc:
            best_auc, best_epoch = val["auc_pr"], epoch
            torch.save(model.state_dict(), out_path.with_suffix(".best_state.pt"))
            flag = "  <- best"
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {np.mean(losses):.4f}  "
            f"val P {val['precision']:.3f} R {val['recall']:.3f} F1 {val['f1']:.3f} AUC-PR {val['auc_pr']:.3f}  "
            f"({time.time() - t0:.0f} s){flag}"
        )

    # --- test --------------------------------------------------------------------------------
    model.load_state_dict(torch.load(out_path.with_suffix(".best_state.pt"), map_location=device, weights_only=True))
    val_prob = predict(eval_x)
    thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)
    best_threshold = max(thresholds, key=lambda t: evaluate(val_prob, eval_y, threshold=t)["f1"])

    test_prob = predict(test_x)
    results = {
        "persistence": evaluate(persistence, test_y),
        "unet@0.5": evaluate(test_prob, test_y),
        f"unet@{best_threshold}": evaluate(test_prob, test_y, threshold=best_threshold),
    }

    print(f"\nTest split ({len(test_y)} tiles), best epoch {best_epoch} (val AUC-PR {best_auc:.3f})")
    print("| model | " + " | ".join(METRIC_KEYS) + " |")
    print("|---|" + "---|" * len(METRIC_KEYS))
    for name, metrics in results.items():
        print(f"| {name} | " + " | ".join(f"{metrics[k]:.3f}" for k in METRIC_KEYS) + " |")

    test_metrics = {k: results["unet@0.5"][k] for k in METRIC_KEYS}
    save_checkpoint(model, out_path, inputs=inputs, encoder=args.encoder, mean=mean, std=std, metrics=test_metrics)
    report = {
        "dataset": "next_day_wildfire_spread",
        "inputs": inputs,
        "encoder": args.encoder,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_samples": int(len(x_train)),
        "best_val_threshold": float(best_threshold),
        "test": {name: {k: float(m[k]) for k in METRIC_KEYS} for name, m in results.items()},
    }
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_path.with_suffix(".best_state.pt").unlink(missing_ok=True)
    print(f"\nSaved {out_path} and {out_path.with_suffix('.json')}")
    print(f"Use it: agents.spread.model: unet, agents.spread.checkpoint: {args.out}, confidence = test F1")


if __name__ == "__main__":
    main()
