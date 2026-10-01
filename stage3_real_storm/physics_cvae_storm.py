from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CHANNELS = ["S_original", "BU_original"]
S_IDX = 0
BU_IDX = 1


@dataclass
class Config:
    latent_dim: int = 16
    hidden_dim: int = 192
    batch_size: int = 32
    epochs: int = 120
    learning_rate: float = 1e-3
    beta_kl: float = 1e-3
    domain_weight: float = 5.0
    test_fraction: float = 0.2
    seed: int = 42
    max_interp_gap: int = 2


class ConditionalVAE(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + c_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        return self.decode(self.reparameterize(mu, logvar), c), mu, logvar


@dataclass
class PreparedData:
    x_train: torch.Tensor
    c_train: torch.Tensor
    x_test: torch.Tensor
    c_test: torch.Tensor
    real_test: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    steps_per_day: int
    station_names: list[str]
    test_keys: list[str]
    residual_mean: float
    residual_std: float
    s_ramp_q99: float
    bu_ramp_q99: float


def _read_pair(x_file: Path, y_file: Path | None) -> pd.DataFrame:
    x = pd.read_csv(x_file)
    for col in ["M_TIMESTAMP", *CHANNELS]:
        if col not in x.columns:
            raise ValueError(f"{x_file} does not contain {col}")

    x["M_TIMESTAMP"] = pd.to_datetime(x["M_TIMESTAMP"], errors="coerce", utc=True)
    x = x.dropna(subset=["M_TIMESTAMP"]).copy()

    if y_file is not None and y_file.exists():
        y = pd.read_csv(y_file)
        if "label" in y.columns:
            if "M_TIMESTAMP" in y.columns:
                y["M_TIMESTAMP"] = pd.to_datetime(y["M_TIMESTAMP"], errors="coerce", utc=True)
                x = x.merge(y[["M_TIMESTAMP", "label"]], on="M_TIMESTAMP", how="left")
            elif len(y) == len(x):
                x["label"] = y["label"].to_numpy()

    if "label" not in x.columns:
        x["label"] = 0

    x["label"] = pd.to_numeric(x["label"], errors="coerce").fillna(5).astype(int)
    x["station_id"] = x_file.stem
    return x.sort_values("M_TIMESTAMP").drop_duplicates("M_TIMESTAMP")[
        ["M_TIMESTAMP", "station_id", "S_original", "BU_original", "label"]
    ]


def load_storm(train_dir: Path) -> pd.DataFrame:
    x_dir, y_dir = train_dir / "X", train_dir / "y"
    if not x_dir.exists():
        raise FileNotFoundError(f"Missing directory: {x_dir}")

    frames = []
    for x_file in sorted(x_dir.glob("*.csv")):
        y_file = y_dir / x_file.name if y_dir.exists() else None
        try:
            frames.append(_read_pair(x_file, y_file))
        except Exception as exc:
            print(f"Skipping {x_file.name}: {exc}")

    if not frames:
        raise RuntimeError("No usable STORM files were loaded.")

    return pd.concat(frames, ignore_index=True)


def infer_steps_per_day(df: pd.DataFrame) -> int:
    intervals = []
    for _, part in df.groupby("station_id"):
        d = part["M_TIMESTAMP"].sort_values().diff().dropna().dt.total_seconds()
        d = d[(d > 0) & (d <= 86400)]
        intervals.extend(d.head(5000).tolist())
    seconds = float(np.median(intervals))
    steps = int(round(86400 / seconds))
    print(f"Inferred sampling interval: {seconds/60:.2f} min -> {steps} steps/day")
    return steps


def build_daily_windows(df: pd.DataFrame, steps_per_day: int, max_interp_gap: int):
    station_names = sorted(df["station_id"].unique().tolist())
    station_to_idx = {s: i for i, s in enumerate(station_names)}
    samples, conditions, keys = [], [], []

    for station_id, part in df.groupby("station_id"):
        part = part.set_index("M_TIMESTAMP").sort_index()
        part[CHANNELS] = part[CHANNELS].apply(pd.to_numeric, errors="coerce")
        part[CHANNELS] = part[CHANNELS].interpolate(
            method="time", limit=max_interp_gap, limit_direction="both"
        )

        for date, day in part.groupby(part.index.date):
            if len(day) != steps_per_day:
                continue
            if day[CHANNELS].isna().any().any():
                continue
            if not (day["label"].to_numpy() == 0).all():
                continue

            arr = day[CHANNELS].to_numpy(np.float32)
            ts = pd.Timestamp(date)
            dow, doy = ts.dayofweek, ts.dayofyear

            station_one_hot = np.zeros(len(station_names), dtype=np.float32)
            station_one_hot[station_to_idx[station_id]] = 1.0

            cond = np.concatenate([
                np.array([
                    float(dow >= 5),
                    math.sin(2 * math.pi * dow / 7),
                    math.cos(2 * math.pi * dow / 7),
                    math.sin(2 * math.pi * doy / 365.25),
                    math.cos(2 * math.pi * doy / 365.25),
                ], dtype=np.float32),
                station_one_hot,
            ])

            samples.append(arr)
            conditions.append(cond)
            keys.append(f"{station_id}:{date}")

    if len(samples) < 20:
        raise ValueError(
            f"Only {len(samples)} complete normal days found. "
            "Use more STORM files or relax preprocessing."
        )

    return np.stack(samples), np.stack(conditions), keys, station_names


def temporal_split(x, c, keys, test_fraction):
    station_groups = {}
    for i, key in enumerate(keys):
        station_groups.setdefault(key.split(":", 1)[0], []).append(i)

    train_idx, test_idx = [], []
    for _, idxs in station_groups.items():
        idxs = sorted(idxs, key=lambda i: keys[i])
        n_test = max(1, int(round(len(idxs) * test_fraction))) if len(idxs) >= 5 else 0
        if n_test:
            train_idx.extend(idxs[:-n_test])
            test_idx.extend(idxs[-n_test:])
        else:
            train_idx.extend(idxs)

    if not test_idx:
        cut = max(1, int(round(len(x) * test_fraction)))
        train_idx = list(range(len(x) - cut))
        test_idx = list(range(len(x) - cut, len(x)))

    return (
        x[train_idx], c[train_idx],
        x[test_idx], c[test_idx],
        [keys[i] for i in test_idx],
    )


def prepare_data(train_dir: Path, cfg: Config, device) -> PreparedData:
    df = load_storm(train_dir)
    steps_per_day = infer_steps_per_day(df)
    x, c, keys, station_names = build_daily_windows(df, steps_per_day, cfg.max_interp_gap)
    x_train_raw, c_train, x_test_raw, c_test, test_keys = temporal_split(
        x, c, keys, cfg.test_fraction
    )

    x_mean_np = x_train_raw.mean(axis=(0, 1), keepdims=True)
    x_std_np = x_train_raw.std(axis=(0, 1), keepdims=True) + 1e-6

    x_train = ((x_train_raw - x_mean_np) / x_std_np).reshape(len(x_train_raw), -1)
    x_test = ((x_test_raw - x_mean_np) / x_std_np).reshape(len(x_test_raw), -1)

    residual = x_train_raw[..., S_IDX] - x_train_raw[..., BU_IDX]
    s_ramps = np.abs(np.diff(x_train_raw[..., S_IDX], axis=1)).reshape(-1)
    bu_ramps = np.abs(np.diff(x_train_raw[..., BU_IDX], axis=1)).reshape(-1)

    return PreparedData(
        x_train=torch.tensor(x_train, dtype=torch.float32, device=device),
        c_train=torch.tensor(c_train, dtype=torch.float32, device=device),
        x_test=torch.tensor(x_test, dtype=torch.float32, device=device),
        c_test=torch.tensor(c_test, dtype=torch.float32, device=device),
        real_test=torch.tensor(x_test_raw, dtype=torch.float32, device=device),
        x_mean=torch.tensor(x_mean_np, dtype=torch.float32, device=device),
        x_std=torch.tensor(x_std_np, dtype=torch.float32, device=device),
        steps_per_day=steps_per_day,
        station_names=station_names,
        test_keys=test_keys,
        residual_mean=float(np.mean(residual)),
        residual_std=float(np.std(residual) + 1e-6),
        s_ramp_q99=float(np.quantile(s_ramps, 0.99)),
        bu_ramp_q99=float(np.quantile(bu_ramps, 0.99)),
    )


def destandardize(flat_x, data):
    shaped = flat_x.reshape(flat_x.shape[0], data.steps_per_day, len(CHANNELS))
    return shaped * data.x_std + data.x_mean


def vae_loss(recon, x, mu, logvar, beta_kl):
    rec = torch.mean((recon - x).square())
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return rec + beta_kl * kl, rec, kl


def domain_consistency_penalty(generated, data):
    # These are domain constraints, not full power-flow physics constraints.
    s = generated[..., S_IDX]
    bu = generated[..., BU_IDX]
    residual = s - bu
    scale = max(data.residual_std, 1e-6)

    nonnegative = (
        torch.relu(-s / scale).square().mean()
        + torch.relu(-bu / scale).square().mean()
    )
    residual_mean = ((residual.mean() - data.residual_mean) / scale).square()
    residual_std = ((residual.std(unbiased=False) - data.residual_std) / scale).square()

    s_ramp = torch.abs(s[:, 1:] - s[:, :-1])
    bu_ramp = torch.abs(bu[:, 1:] - bu[:, :-1])
    ramp = (
        torch.relu((s_ramp - data.s_ramp_q99) / max(data.s_ramp_q99, 1e-6)).square().mean()
        + torch.relu((bu_ramp - data.bu_ramp_q99) / max(data.bu_ramp_q99, 1e-6)).square().mean()
    )
    return 2.0 * nonnegative + residual_mean + residual_std + ramp


def train_model(model, data, cfg, domain_aware):
    ds = TensorDataset(data.x_train.detach().cpu(), data.c_train.detach().cpu())
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    device = next(model.parameters()).device
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "domain": 0.0}
        count = 0

        for xb_cpu, cb_cpu in loader:
            xb, cb = xb_cpu.to(device), cb_cpu.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb, cb)
            base, rec, kl = vae_loss(recon, xb, mu, logvar, cfg.beta_kl)

            domain = torch.zeros((), device=device)
            if domain_aware:
                recon_real = destandardize(recon, data)
                z = torch.randn(xb.shape[0], model.latent_dim, device=device)
                prior_real = destandardize(model.decode(z, cb), data)
                domain = 0.5 * (
                    domain_consistency_penalty(recon_real, data)
                    + domain_consistency_penalty(prior_real, data)
                )

            loss = base + (cfg.domain_weight * domain if domain_aware else 0.0)
            loss.backward()
            optimizer.step()

            n = xb.shape[0]
            count += n
            sums["loss"] += float(loss.detach().cpu()) * n
            sums["rec"] += float(rec.detach().cpu()) * n
            sums["kl"] += float(kl.detach().cpu()) * n
            sums["domain"] += float(domain.detach().cpu()) * n

        row = {"epoch": epoch, **{k: v / count for k, v in sums.items()}}
        history.append(row)

        if epoch == 1 or epoch % 20 == 0 or epoch == cfg.epochs:
            print(
                f"epoch={epoch:03d} loss={row['loss']:.5f} "
                f"rec={row['rec']:.5f} kl={row['kl']:.5f} "
                f"domain={row['domain']:.5f}"
            )

    return pd.DataFrame(history)


@torch.no_grad()
def generate(model, conditions, data, seed):
    torch.manual_seed(seed)
    model.eval()
    z = torch.randn(conditions.shape[0], model.latent_dim, device=conditions.device)
    return destandardize(model.decode(z, conditions), data)


def lag1(x):
    x0, x1 = x[:, :-1], x[:, 1:]
    x0 = x0 - x0.mean(dim=1, keepdim=True)
    x1 = x1 - x1.mean(dim=1, keepdim=True)
    num = (x0 * x1).mean(dim=1)
    den = torch.sqrt(
        (x0.square().mean(dim=1) * x1.square().mean(dim=1)).clamp_min(1e-8)
    )
    return float((num / den).mean().cpu())


def evaluate(real, synthetic, data):
    rs, ss = real[..., S_IDX], synthetic[..., S_IDX]
    rbu, sbu = real[..., BU_IDX], synthetic[..., BU_IDX]
    rres, sres = rs - rbu, ss - sbu

    return {
        "s_mean_profile_rmse": float(torch.sqrt(torch.mean((rs.mean(0) - ss.mean(0)).square())).cpu()),
        "bu_mean_profile_rmse": float(torch.sqrt(torch.mean((rbu.mean(0) - sbu.mean(0)).square())).cpu()),
        "s_std_profile_rmse": float(torch.sqrt(torch.mean((rs.std(0) - ss.std(0)).square())).cpu()),
        "negative_s_fraction": float((ss < 0).float().mean().cpu()),
        "negative_bu_fraction": float((sbu < 0).float().mean().cpu()),
        "residual_mean_gap": float(torch.abs(sres.mean() - rres.mean()).cpu()),
        "residual_std_gap": float(torch.abs(sres.std() - rres.std()).cpu()),
        "s_lag1_real": lag1(rs),
        "s_lag1_synthetic": lag1(ss),
        "s_ramp_violation_fraction": float(
            (torch.abs(ss[:, 1:] - ss[:, :-1]) > data.s_ramp_q99).float().mean().cpu()
        ),
        "bu_ramp_violation_fraction": float(
            (torch.abs(sbu[:, 1:] - sbu[:, :-1]) > data.bu_ramp_q99).float().mean().cpu()
        ),
    }


def run(train_dir: Path, output: Path, cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data = prepare_data(train_dir, cfg, device)
    print(
        f"stations={len(data.station_names)} train_days={len(data.x_train)} "
        f"test_days={len(data.x_test)} steps_per_day={data.steps_per_day}"
    )

    x_dim, c_dim = data.x_train.shape[1], data.c_train.shape[1]
    baseline = ConditionalVAE(x_dim, c_dim, cfg.latent_dim, cfg.hidden_dim).to(device)
    domain_model = ConditionalVAE(x_dim, c_dim, cfg.latent_dim, cfg.hidden_dim).to(device)
    domain_model.load_state_dict(baseline.state_dict())

    print("\nTraining baseline CVAE on REAL STORM data")
    h1 = train_model(baseline, data, cfg, False)

    print("\nTraining domain-informed CVAE on REAL STORM data")
    h2 = train_model(domain_model, data, cfg, True)

    output.mkdir(parents=True, exist_ok=True)
    torch.save(baseline.state_dict(), output / "storm_cvae.pt")
    torch.save(domain_model.state_dict(), output / "storm_domain_cvae.pt")
    h1.to_csv(output / "training_history_cvae.csv", index=False)
    h2.to_csv(output / "training_history_domain_cvae.csv", index=False)

    base_sample = generate(baseline, data.c_test, data, cfg.seed + 100)
    domain_sample = generate(domain_model, data.c_test, data, cfg.seed + 100)

    comparison = pd.DataFrame([
        {"model": "cvae", **evaluate(data.real_test, base_sample, data)},
        {"model": "domain_cvae", **evaluate(data.real_test, domain_sample, data)},
    ])
    comparison.to_csv(output / "model_comparison.csv", index=False)

    real_mean = data.real_test[..., S_IDX].mean(0).cpu().numpy()
    base_mean = base_sample[..., S_IDX].mean(0).cpu().numpy()
    domain_mean = domain_sample[..., S_IDX].mean(0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(11, 4))
    xx = np.arange(data.steps_per_day)
    ax.plot(xx, real_mean, label="Real STORM")
    ax.plot(xx, base_mean, label="CVAE")
    ax.plot(xx, domain_mean, label="Domain-informed CVAE")
    ax.set(
        xlabel="Step within day",
        ylabel="S_original",
        title="Mean daily profile: real STORM vs generated",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mean_profile_comparison.png", dpi=180)
    plt.close(fig)

    meta = {
        "training_source": "real STORM Train/X + Train/y",
        "normal_only_training": True,
        "channels": CHANNELS,
        "stations": data.station_names,
        "steps_per_day": data.steps_per_day,
        "note": (
            "STORM X/y does not provide full topology/voltage/line-capacity data, "
            "so the second model uses domain constraints rather than claiming "
            "full power-flow physics."
        ),
        "config": cfg.__dict__,
    }
    (output / "training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nModel comparison")
    print(comparison.to_string(index=False))
    print("\nSaved in:", output.resolve())


def parse_args():
    p = argparse.ArgumentParser(description="Train CVAE models directly on real STORM data.")
    p.add_argument("--train-dir", type=Path, default=Path("storm_data/Train"))
    p.add_argument("--output", type=Path, default=Path("storm_real_cvae_results"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--domain-weight", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-interp-gap", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        a.train_dir,
        a.output,
        Config(
            latent_dim=a.latent_dim,
            hidden_dim=a.hidden_dim,
            batch_size=a.batch_size,
            epochs=a.epochs,
            domain_weight=a.domain_weight,
            seed=a.seed,
            max_interp_gap=a.max_interp_gap,
        ),
    )
