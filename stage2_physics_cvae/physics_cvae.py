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


CHANNELS = ["consumption_kw", "pv_generation_kw", "net_load_kw"]
CONSUMPTION_IDX = 0
PV_IDX = 1
NET_LOAD_IDX = 2


@dataclass
class CVAEConfig:
    latent_dim: int = 24
    hidden_dim: int = 256
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 1e-3
    beta_kl: float = 1e-3
    physics_weight: float = 10.0
    test_fraction: float = 0.2
    seed: int = 42


class ConditionalVAE(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, latent_dim: int = 24, hidden_dim: int = 256) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
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

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class DifferentiableRadialGrid:
    def __init__(self, edges: pd.DataFrame, consumer_ids: list[str], sim_cfg: dict, device: torch.device) -> None:
        self.consumer_ids = consumer_ids
        self.nominal_voltage_v = float(sim_cfg["nominal_voltage_v"])
        self.power_factor = float(sim_cfg["power_factor"])
        self.minimum_voltage_pu = float(sim_cfg["minimum_allowed_voltage_pu"])
        self.maximum_voltage_pu = float(sim_cfg["maximum_allowed_voltage_pu"])
        self.parents = edges["parent"].astype(str).tolist()
        self.children = edges["child"].astype(str).tolist()
        self.resistance = torch.tensor(edges["resistance_ohm"].to_numpy(float), dtype=torch.float32, device=device)
        self.capacity_kw = torch.tensor(edges["capacity_kw"].to_numpy(float), dtype=torch.float32, device=device)
        child_map: dict[str, list[str]] = {}
        for parent, child in zip(self.parents, self.children):
            child_map.setdefault(parent, []).append(child)

        def descendants(node: str) -> set[str]:
            result: set[str] = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current in result:
                    continue
                result.add(current)
                stack.extend(child_map.get(current, []))
            return result

        consumer_index = {c: i for i, c in enumerate(consumer_ids)}
        incidence = np.zeros((len(edges), len(consumer_ids)), dtype=np.float32)
        for line_idx, child in enumerate(self.children):
            for node in descendants(child):
                if node in consumer_index:
                    incidence[line_idx, consumer_index[node]] = 1.0
        self.incidence = torch.tensor(incidence, dtype=torch.float32, device=device)

    def evaluate(self, net_load_kw: torch.Tensor) -> dict[str, torch.Tensor]:
        # net_load_kw: [batch, time, consumers]
        downstream_kw = torch.einsum("btn,ln->btl", net_load_kw, self.incidence)
        loading_ratio = torch.abs(downstream_kw) / torch.clamp(self.capacity_kw.view(1, 1, -1), min=1e-6)
        current_signed_a = downstream_kw * 1000.0 / (math.sqrt(3.0) * self.nominal_voltage_v * self.power_factor)
        current_abs_a = torch.abs(current_signed_a)
        line_loss_kw = 3.0 * current_abs_a.square() * self.resistance.view(1, 1, -1) / 1000.0
        voltage_drop_v = math.sqrt(3.0) * current_signed_a * self.resistance.view(1, 1, -1) * self.power_factor
        root_voltage = torch.full(net_load_kw.shape[:2], self.nominal_voltage_v, dtype=net_load_kw.dtype, device=net_load_kw.device)
        voltage_by_node: dict[str, torch.Tensor] = {"SUBSTATION": root_voltage}
        line_voltage = []
        for idx, (parent, child) in enumerate(zip(self.parents, self.children)):
            if parent not in voltage_by_node:
                raise ValueError(f"Network edge order is not topological: parent {parent} is missing")
            child_v = voltage_by_node[parent] - voltage_drop_v[:, :, idx]
            voltage_by_node[child] = child_v
            line_voltage.append(child_v)
        voltage_pu = torch.stack(line_voltage, dim=2) / self.nominal_voltage_v
        return {"loading_ratio": loading_ratio, "voltage_pu": voltage_pu, "line_loss_kw": line_loss_kw}

    def penalty(self, net_load_kw: torch.Tensor) -> torch.Tensor:
        state = self.evaluate(net_load_kw)
        overload = torch.relu(state["loading_ratio"] - 1.0).square().mean()
        undervoltage = torch.relu(self.minimum_voltage_pu - state["voltage_pu"]).square().mean()
        overvoltage = torch.relu(state["voltage_pu"] - self.maximum_voltage_pu).square().mean()
        return overload + 20.0 * (undervoltage + overvoltage)

    @torch.no_grad()
    def metrics(self, net_load_kw: torch.Tensor) -> dict[str, float]:
        state = self.evaluate(net_load_kw)
        return {
            "overload_fraction": float((state["loading_ratio"] > 1.0).float().mean().cpu()),
            "voltage_violation_fraction": float(((state["voltage_pu"] < self.minimum_voltage_pu) | (state["voltage_pu"] > self.maximum_voltage_pu)).float().mean().cpu()),
            "maximum_loading_percent": float((100.0 * state["loading_ratio"].max()).cpu()),
            "minimum_voltage_pu": float(state["voltage_pu"].min().cpu()),
            "maximum_voltage_pu": float(state["voltage_pu"].max().cpu()),
            "mean_line_loss_kw": float(state["line_loss_kw"].sum(dim=2).mean().cpu()),
        }


@dataclass
class PreparedData:
    x_train: torch.Tensor
    c_train: torch.Tensor
    x_test: torch.Tensor
    c_test: torch.Tensor
    real_test: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    days_test: list[str]
    consumer_ids: list[str]
    steps_per_day: int
    grid: DifferentiableRadialGrid
    balance_scale_kw: float


def prepare_data(data_dir: Path, cfg: CVAEConfig, device: torch.device) -> PreparedData:
    consumer = pd.read_csv(data_dir / "consumer_timeseries.csv", parse_dates=["M_TIMESTAMP"])
    context = pd.read_csv(data_dir / "context_timeseries.csv", parse_dates=["M_TIMESTAMP"])
    edges = pd.read_csv(data_dir / "network_edges.csv")
    sim_cfg = json.loads((data_dir / "simulation_config.json").read_text(encoding="utf-8"))
    consumer_ids = sorted(consumer["consumer_id"].astype(str).unique().tolist())
    context = context.set_index("M_TIMESTAMP").sort_index()
    interval_minutes = int(sim_cfg["interval_minutes"])
    steps_per_day = 1440 // interval_minutes

    pivots = {
        channel: consumer.pivot(index="M_TIMESTAMP", columns="consumer_id", values=channel).sort_index().reindex(columns=consumer_ids)
        for channel in CHANNELS
    }
    dates = sorted(set(pivots[CHANNELS[0]].index.date))
    x_days: list[np.ndarray] = []
    conditions: list[list[float]] = []
    day_names: list[str] = []
    for date in dates:
        index = pivots[CHANNELS[0]].index[pivots[CHANNELS[0]].index.date == date]
        if len(index) != steps_per_day:
            continue
        channel_arrays = [pivots[ch].loc[index].to_numpy(np.float32) for ch in CHANNELS]
        if any(np.isnan(arr).any() for arr in channel_arrays):
            continue
        x_days.append(np.stack(channel_arrays, axis=-1))  # [time, consumers, channels]
        day_ctx = context.loc[index]
        dow = pd.Timestamp(date).dayofweek
        conditions.append([
            float(dow >= 5),
            float(np.sin(2 * np.pi * dow / 7)),
            float(np.cos(2 * np.pi * dow / 7)),
            float(day_ctx["temperature_c"].mean()),
            float(day_ctx["temperature_c"].min()),
            float(day_ctx["temperature_c"].max()),
            float(day_ctx["solar_irradiance_w_m2"].mean()),
            float(day_ctx["solar_irradiance_w_m2"].max()),
        ])
        day_names.append(str(date))

    x = np.stack(x_days).astype(np.float32)
    c = np.asarray(conditions, dtype=np.float32)
    if len(x) < 20:
        raise ValueError(f"Only {len(x)} complete days found. Generate at least 30-60 days; 180+ is preferable.")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(x))
    n_test = max(1, int(round(len(x) * cfg.test_fraction)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    c_train_raw, c_test_raw = c[train_idx], c[test_idx]

    x_mean_np = x_train_raw.mean(axis=(0, 1), keepdims=True)
    x_std_np = x_train_raw.std(axis=(0, 1), keepdims=True) + 1e-6
    c_mean_np = c_train_raw.mean(axis=0, keepdims=True)
    c_std_np = c_train_raw.std(axis=0, keepdims=True) + 1e-6
    x_train = ((x_train_raw - x_mean_np) / x_std_np).reshape(len(x_train_raw), -1)
    x_test = ((x_test_raw - x_mean_np) / x_std_np).reshape(len(x_test_raw), -1)
    c_train = (c_train_raw - c_mean_np) / c_std_np
    c_test = (c_test_raw - c_mean_np) / c_std_np
    balance_scale_kw = float(max(1.0, np.std(x_train_raw[..., NET_LOAD_IDX])))

    return PreparedData(
        x_train=torch.tensor(x_train, dtype=torch.float32, device=device),
        c_train=torch.tensor(c_train, dtype=torch.float32, device=device),
        x_test=torch.tensor(x_test, dtype=torch.float32, device=device),
        c_test=torch.tensor(c_test, dtype=torch.float32, device=device),
        real_test=torch.tensor(x_test_raw, dtype=torch.float32, device=device),
        x_mean=torch.tensor(x_mean_np, dtype=torch.float32, device=device),
        x_std=torch.tensor(x_std_np, dtype=torch.float32, device=device),
        days_test=[day_names[i] for i in test_idx],
        consumer_ids=consumer_ids,
        steps_per_day=steps_per_day,
        grid=DifferentiableRadialGrid(edges, consumer_ids, sim_cfg, device),
        balance_scale_kw=balance_scale_kw,
    )


def destandardize(flat_x: torch.Tensor, data: PreparedData) -> torch.Tensor:
    shaped = flat_x.reshape(flat_x.shape[0], data.steps_per_day, len(data.consumer_ids), len(CHANNELS))
    return shaped * data.x_std + data.x_mean


def physical_consistency_penalty(generated: torch.Tensor, data: PreparedData) -> torch.Tensor:
    consumption = generated[..., CONSUMPTION_IDX]
    pv = generated[..., PV_IDX]
    net = generated[..., NET_LOAD_IDX]
    balance_error = (net - (consumption - pv)) / data.balance_scale_kw
    balance_penalty = balance_error.square().mean()
    nonnegative_penalty = torch.relu(-consumption / data.balance_scale_kw).square().mean() + torch.relu(-pv / data.balance_scale_kw).square().mean()
    grid_penalty = data.grid.penalty(net)
    return balance_penalty + 2.0 * nonnegative_penalty + grid_penalty


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = torch.mean((recon - x).square())
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return reconstruction + beta_kl * kl, reconstruction, kl


def train_model(model: ConditionalVAE, data: PreparedData, cfg: CVAEConfig, physics_aware: bool) -> pd.DataFrame:
    dataset = TensorDataset(data.x_train.detach().cpu(), data.c_train.detach().cpu())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    device = next(model.parameters()).device
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "physics": 0.0}
        count = 0
        for xb_cpu, cb_cpu in loader:
            xb, cb = xb_cpu.to(device), cb_cpu.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb, cb)
            base_loss, rec_loss, kl_loss = vae_loss(recon, xb, mu, logvar, cfg.beta_kl)
            physics_loss = torch.zeros((), device=device)
            if physics_aware:
                recon_phys = physical_consistency_penalty(destandardize(recon, data), data)
                prior_z = torch.randn(xb.shape[0], model.latent_dim, device=device)
                prior_phys = physical_consistency_penalty(destandardize(model.decode(prior_z, cb), data), data)
                physics_loss = 0.5 * (recon_phys + prior_phys)
            loss = base_loss + (cfg.physics_weight * physics_loss if physics_aware else 0.0)
            loss.backward()
            optimizer.step()
            n = xb.shape[0]
            count += n
            totals["loss"] += float(loss.detach().cpu()) * n
            totals["reconstruction"] += float(rec_loss.detach().cpu()) * n
            totals["kl"] += float(kl_loss.detach().cpu()) * n
            totals["physics"] += float(physics_loss.detach().cpu()) * n
        row = {"epoch": epoch, **{k: v / count for k, v in totals.items()}}
        history.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == cfg.epochs:
            print(f"epoch={epoch:03d} loss={row['loss']:.5f} rec={row['reconstruction']:.5f} kl={row['kl']:.5f} phys={row['physics']:.5f}")
    return pd.DataFrame(history)


@torch.no_grad()
def generate(model: ConditionalVAE, conditions: torch.Tensor, data: PreparedData, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    model.eval()
    z = torch.randn(conditions.shape[0], model.latent_dim, device=conditions.device)
    return destandardize(model.decode(z, conditions), data)


def lag1_autocorrelation(net_load: torch.Tensor) -> float:
    x0, x1 = net_load[:, :-1, :], net_load[:, 1:, :]
    x0 = x0 - x0.mean(dim=1, keepdim=True)
    x1 = x1 - x1.mean(dim=1, keepdim=True)
    numerator = (x0 * x1).mean(dim=1)
    denominator = torch.sqrt((x0.square().mean(dim=1) * x1.square().mean(dim=1)).clamp_min(1e-8))
    return float((numerator / denominator).mean().cpu())


def physical_metrics(sample: torch.Tensor, data: PreparedData) -> dict[str, float]:
    consumption = sample[..., CONSUMPTION_IDX]
    pv = sample[..., PV_IDX]
    net = sample[..., NET_LOAD_IDX]
    balance_rmse = torch.sqrt(torch.mean((net - (consumption - pv)).square()))
    return {
        "power_balance_rmse_kw": float(balance_rmse.cpu()),
        "negative_consumption_fraction": float((consumption < 0).float().mean().cpu()),
        "negative_pv_fraction": float((pv < 0).float().mean().cpu()),
        **data.grid.metrics(net),
    }



def hard_physics_projection(sample: torch.Tensor) -> torch.Tensor:
    projected = sample.clone()
    consumption = torch.clamp(projected[..., CONSUMPTION_IDX], min=0.0)
    pv = torch.clamp(projected[..., PV_IDX], min=0.0)
    projected[..., CONSUMPTION_IDX] = consumption
    projected[..., PV_IDX] = pv
    projected[..., NET_LOAD_IDX] = consumption - pv
    return projected

def fidelity_metrics(real: torch.Tensor, synthetic: torch.Tensor) -> dict[str, float]:
    real_net, syn_net = real[..., NET_LOAD_IDX], synthetic[..., NET_LOAD_IDX]
    real_mean, syn_mean = real_net.mean(dim=0), syn_net.mean(dim=0)
    real_std, syn_std = real_net.std(dim=0), syn_net.std(dim=0)
    real_energy, syn_energy = real_net.sum(dim=(1, 2)), syn_net.sum(dim=(1, 2))
    real_peak = real_net.sum(dim=2).max(dim=1).values
    syn_peak = syn_net.sum(dim=2).max(dim=1).values
    return {
        "mean_profile_rmse_kw": float(torch.sqrt(torch.mean((real_mean - syn_mean).square())).cpu()),
        "std_profile_rmse_kw": float(torch.sqrt(torch.mean((real_std - syn_std).square())).cpu()),
        "mean_daily_energy_gap_percent": float((100.0 * torch.abs(syn_energy.mean() - real_energy.mean()) / real_energy.mean().abs().clamp_min(1e-6)).cpu()),
        "mean_peak_load_gap_percent": float((100.0 * torch.abs(syn_peak.mean() - real_peak.mean()) / real_peak.mean().abs().clamp_min(1e-6)).cpu()),
        "lag1_autocorrelation_real": lag1_autocorrelation(real_net),
        "lag1_autocorrelation_synthetic": lag1_autocorrelation(syn_net),
    }


def evaluate_and_save(output: Path, data: PreparedData, baseline: ConditionalVAE, physics_model: ConditionalVAE, baseline_history: pd.DataFrame, physics_history: pd.DataFrame, seed: int) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    baseline_sample = generate(baseline, data.c_test, data, seed)
    physics_sample = generate(physics_model, data.c_test, data, seed)
    hard_sample = hard_physics_projection(physics_sample)
    rows = []
    for name, sample in [("real", data.real_test), ("cvae", baseline_sample), ("physics_cvae", physics_sample), ("physics_cvae_hard", hard_sample)]:
        row = {"model": name, **physical_metrics(sample, data)}
        if name != "real":
            row.update(fidelity_metrics(data.real_test, sample))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "model_comparison.csv", index=False)
    baseline_history.to_csv(output / "training_history_cvae.csv", index=False)
    physics_history.to_csv(output / "training_history_physics_cvae.csv", index=False)

    def to_long(sample: torch.Tensor, model_name: str) -> pd.DataFrame:
        arr = sample.detach().cpu().numpy()
        records = []
        for d, day in enumerate(data.days_test):
            for t in range(data.steps_per_day):
                for n, consumer_id in enumerate(data.consumer_ids):
                    records.append((model_name, day, t, consumer_id, float(arr[d, t, n, 0]), float(arr[d, t, n, 1]), float(arr[d, t, n, 2])))
        return pd.DataFrame(records, columns=["model", "day", "step", "consumer_id", *CHANNELS])

    pd.concat([to_long(baseline_sample, "cvae"), to_long(physics_sample, "physics_cvae"), to_long(hard_sample, "physics_cvae_hard")], ignore_index=True).to_csv(output / "generated_test_profiles.csv", index=False)

    real_agg = data.real_test[..., NET_LOAD_IDX].sum(dim=2).mean(dim=0).cpu().numpy()
    base_agg = baseline_sample[..., NET_LOAD_IDX].sum(dim=2).mean(dim=0).cpu().numpy()
    phys_agg = physics_sample[..., NET_LOAD_IDX].sum(dim=2).mean(dim=0).cpu().numpy()
    hard_agg = hard_sample[..., NET_LOAD_IDX].sum(dim=2).mean(dim=0).cpu().numpy()
    x = np.arange(data.steps_per_day)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, real_agg, label="Simulator data")
    ax.plot(x, base_agg, label="CVAE")
    ax.plot(x, phys_agg, label="Physics-CVAE (soft)")
    ax.plot(x, hard_agg, label="Physics-CVAE (hard projection)")
    ax.set(xlabel="15-minute step", ylabel="Aggregate net load (kW)", title="Mean daily aggregate profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mean_daily_profile_comparison.png", dpi=180)
    plt.close(fig)

    compare = metrics.set_index("model")[["power_balance_rmse_kw", "overload_fraction", "voltage_violation_fraction"]]
    ax = compare.plot(kind="bar", figsize=(9, 4))
    ax.set_title("Physical consistency comparison")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output / "physical_consistency_comparison.png", dpi=180)
    plt.close(fig)
    return metrics


def run(data_dir: Path, output: Path, cfg: CVAEConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    data = prepare_data(data_dir, cfg, device)
    x_dim, c_dim = data.x_train.shape[1], data.c_train.shape[1]
    print(f"days train={len(data.x_train)} test={len(data.x_test)} x_dim={x_dim} c_dim={c_dim}")
    baseline = ConditionalVAE(x_dim, c_dim, cfg.latent_dim, cfg.hidden_dim).to(device)
    physics_model = ConditionalVAE(x_dim, c_dim, cfg.latent_dim, cfg.hidden_dim).to(device)
    physics_model.load_state_dict(baseline.state_dict())
    print("\nTraining baseline CVAE")
    baseline_history = train_model(baseline, data, cfg, physics_aware=False)
    print("\nTraining physics-aware CVAE")
    physics_history = train_model(physics_model, data, cfg, physics_aware=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(baseline.state_dict(), output / "cvae.pt")
    torch.save(physics_model.state_dict(), output / "physics_cvae.pt")
    (output / "cvae_config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    metrics = evaluate_and_save(output, data, baseline, physics_model, baseline_history, physics_history, cfg.seed + 100)
    print("\nModel comparison")
    print(metrics.to_string(index=False))
    print("\nSaved in:", output.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a CVAE with a physics-aware CVAE on synthetic grid time series.")
    parser.add_argument("--data-dir", type=Path, default=Path("notebook_results"))
    parser.add_argument("--output", type=Path, default=Path("cvae_results"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--physics-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.data_dir,
        args.output,
        CVAEConfig(
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            physics_weight=args.physics_weight,
            seed=args.seed,
        ),
    )
