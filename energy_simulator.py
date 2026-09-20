from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


class ConsumerType(str, Enum):
    HOME = "home"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"


@dataclass
class SimulationConfig:
    seed: int = 42
    days: int = 14
    interval_minutes: int = 15
    n_consumers: int = 18
    start: str = "2026-01-01 00:00:00+00:00"
    nominal_voltage_v: float = 11000.0
    power_factor: float = 0.95
    line_resistance_ohm_per_km: float = 0.12
    min_line_capacity_kw: float = 80.0
    max_line_capacity_kw: float = 500.0
    minimum_allowed_voltage_pu: float = 0.90
    maximum_allowed_voltage_pu: float = 1.10
    measurement_noise_std_fraction: float = 0.012
    bottom_up_noise_std_fraction: float = 0.035
    anomaly_probability: float = 0.003
    switch_event_probability_per_day: float = 0.05
    missing_probability: float = 0.003
    uncertain_boundary_steps: int = 2

    def validate(self) -> None:
        if self.days < 1 or self.n_consumers < 3:
            raise ValueError("days must be >= 1 and n_consumers must be >= 3")
        if 1440 % self.interval_minutes:
            raise ValueError("interval_minutes must divide one day exactly")
        if not 0 < self.power_factor <= 1:
            raise ValueError("power_factor must be in (0, 1]")


@dataclass
class Consumer:
    consumer_id: str
    consumer_type: str
    household_size: int
    floor_area_m2: float
    employee_count: int
    installed_capacity_kw: float
    has_ev: bool
    ev_charger_kw: float
    has_pv: bool
    pv_capacity_kw: float
    heating_type: str
    insulation_factor: float
    opening_hour: int
    closing_hour: int
    phase: str


@dataclass
class Line:
    parent: str
    child: str
    distance_km: float
    capacity_kw: float
    resistance_ohm_per_km: float

    @property
    def resistance_ohm(self) -> float:
        return self.distance_km * self.resistance_ohm_per_km


class ConsumerFactory:
    @staticmethod
    def random(consumer_id: str, rng: np.random.Generator) -> Consumer:
        kind = rng.choice(
            [ConsumerType.HOME.value, ConsumerType.COMMERCIAL.value, ConsumerType.INDUSTRIAL.value],
            p=[0.72, 0.20, 0.08],
        )
        if kind == ConsumerType.HOME.value:
            floor = float(rng.uniform(45, 220))
            people = int(rng.integers(1, 7))
            employees, capacity, opening, closing = 0, float(rng.uniform(4, 14)), 0, 24
        elif kind == ConsumerType.COMMERCIAL.value:
            floor = float(rng.uniform(120, 1600))
            people = 0
            employees = int(rng.integers(3, 90))
            capacity, opening, closing = float(rng.uniform(20, 100)), int(rng.integers(7, 10)), int(rng.integers(17, 22))
        else:
            floor = float(rng.uniform(400, 4000))
            people, employees = 0, int(rng.integers(10, 180))
            capacity, opening, closing = float(rng.uniform(60, 250)), int(rng.integers(5, 8)), int(rng.integers(18, 24))
        has_ev = bool(rng.random() < (0.35 if kind == ConsumerType.HOME.value else 0.12))
        has_pv = bool(rng.random() < (0.45 if kind != ConsumerType.INDUSTRIAL.value else 0.25))
        return Consumer(
            consumer_id=consumer_id,
            consumer_type=kind,
            household_size=people,
            floor_area_m2=round(floor, 2),
            employee_count=employees,
            installed_capacity_kw=round(capacity, 2),
            has_ev=has_ev,
            ev_charger_kw=float(rng.choice([3.7, 7.4, 11.0])) if has_ev else 0.0,
            has_pv=has_pv,
            pv_capacity_kw=round(float(rng.uniform(2, 18)), 2) if has_pv else 0.0,
            heating_type=str(rng.choice(["gas", "heat_pump", "electric"])),
            insulation_factor=round(float(rng.uniform(0.75, 1.30)), 3),
            opening_hour=opening,
            closing_hour=closing,
            phase=str(rng.choice(["A", "B", "C"])),
        )


class ContextGenerator:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
        day = np.arange(len(timestamps)) / (24 * 60 / (timestamps[1] - timestamps[0]).total_seconds() * 60)
        temperature = 7 + 5 * np.sin(2 * np.pi * (hour - 14) / 24) + 1.5 * np.sin(2 * np.pi * day / 7)
        temperature += self.rng.normal(0, 0.8, len(timestamps))
        daylight = np.maximum(0, np.sin(np.pi * (hour - 7) / 10))
        cloud = np.clip(55 + 30 * np.sin(2 * np.pi * day / 3) + self.rng.normal(0, 12, len(timestamps)), 0, 100)
        irradiance = 600 * daylight * (1 - 0.0065 * cloud)
        return pd.DataFrame({
            "M_TIMESTAMP": timestamps,
            "temperature_c": temperature,
            "humidity_percent": np.clip(75 - 1.7 * temperature + self.rng.normal(0, 4, len(timestamps)), 25, 100),
            "solar_irradiance_w_m2": np.clip(irradiance, 0, None),
            "cloud_cover_percent": cloud,
            "is_weekend": timestamps.dayofweek.to_numpy() >= 5,
            "hour": hour,
        })


class RuleBasedDemandModel:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    @staticmethod
    def gaussian_peak(hour: np.ndarray, center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((hour - center) / width) ** 2)

    def calculate(self, consumer: Consumer, context: pd.DataFrame) -> pd.DataFrame:
        hour = context["hour"].to_numpy()
        temp = context["temperature_c"].to_numpy()
        weekend = context["is_weekend"].to_numpy()
        if consumer.consumer_type == ConsumerType.HOME.value:
            base = 0.25 + 0.12 * consumer.household_size + 0.0015 * consumer.floor_area_m2
            shape = 0.45 + 0.75 * self.gaussian_peak(hour, 7.5, 1.5) + 1.25 * self.gaussian_peak(hour, 19.0, 2.2)
            shape *= np.where(weekend, 1.10, 1.0)
        elif consumer.consumer_type == ConsumerType.COMMERCIAL.value:
            base = 1.5 + 0.005 * consumer.floor_area_m2 + 0.06 * consumer.employee_count
            open_mask = (hour >= consumer.opening_hour) & (hour < consumer.closing_hour)
            shape = np.where(open_mask, 1.0, 0.18) * np.where(weekend, 0.35, 1.0)
        else:
            base = 0.42 * consumer.installed_capacity_kw
            open_mask = (hour >= consumer.opening_hour) & (hour < consumer.closing_hour)
            shape = np.where(open_mask, 1.0, 0.20) * np.where(weekend, 0.65, 1.0)
        heating = np.maximum(0, 16 - temp) * 0.035 * consumer.insulation_factor
        if consumer.heating_type == "gas":
            heating *= 0.20
        cooling = np.maximum(0, temp - 24) * 0.025
        demand = base * shape * (1 + heating + cooling)
        if consumer.has_ev:
            charge = ((hour >= 22) | (hour < 1.5)).astype(float) * consumer.ev_charger_kw
            charge *= self.rng.binomial(1, 0.45, len(hour))
        else:
            charge = np.zeros(len(hour))
        pv = consumer.pv_capacity_kw * 0.82 * context["solar_irradiance_w_m2"].to_numpy() / 1000
        noise = self.rng.normal(0, np.maximum(0.03, 0.035 * demand), len(hour))
        consumption = np.clip(demand + charge + noise, 0.02, None)
        return pd.DataFrame({
            "M_TIMESTAMP": context["M_TIMESTAMP"],
            "consumer_id": consumer.consumer_id,
            "consumption_kw": consumption,
            "pv_generation_kw": np.clip(pv, 0, None),
            "ev_charging_kw": charge,
            "net_load_kw": consumption - pv,
        })


class RadialGrid:
    def __init__(self, lines: list[Line], config: SimulationConfig) -> None:
        self.lines = lines
        self.config = config
        self.children: dict[str, list[str]] = {}
        self.line_by_child = {line.child: line for line in lines}
        for line in lines:
            self.children.setdefault(line.parent, []).append(line.child)

    @classmethod
    def random(cls, consumers: list[Consumer], config: SimulationConfig, rng: np.random.Generator) -> "RadialGrid":
        connected = ["SUBSTATION"]
        lines: list[Line] = []
        for index, consumer in enumerate(consumers):
            candidate_parents = connected[max(0, len(connected) - 6):]
            parent = str(rng.choice(candidate_parents))
            depth_factor = 1 + 0.04 * index
            lines.append(Line(
                parent=parent,
                child=consumer.consumer_id,
                distance_km=round(float(rng.uniform(0.05, 0.65)), 3),
                capacity_kw=round(float(rng.uniform(config.min_line_capacity_kw, config.max_line_capacity_kw) * depth_factor), 2),
                resistance_ohm_per_km=config.line_resistance_ohm_per_km,
            ))
            connected.append(consumer.consumer_id)
        return cls(lines, config)

    def descendants(self, node: str) -> list[str]:
        result: list[str] = []
        stack = [node]
        while stack:
            current = stack.pop()
            result.append(current)
            stack.extend(self.children.get(current, []))
        return result

    def evaluate(self, consumer_frame: pd.DataFrame) -> pd.DataFrame:
        pivot = consumer_frame.pivot(index="M_TIMESTAMP", columns="consumer_id", values="net_load_kw")
        voltage_by_node = {"SUBSTATION": np.full(len(pivot), self.config.nominal_voltage_v)}
        records = []
        remaining = list(self.lines)
        while remaining:
            progress = False
            for line in remaining.copy():
                if line.parent not in voltage_by_node:
                    continue
                downstream = [node for node in self.descendants(line.child) if node in pivot.columns]
                active_kw = pivot[downstream].sum(axis=1).to_numpy()
                apparent_kva = np.abs(active_kw) / self.config.power_factor
                current_a = apparent_kva * 1000 / (np.sqrt(3) * self.config.nominal_voltage_v)
                loss_kw = 3 * current_a**2 * line.resistance_ohm / 1000
                voltage_drop_v = np.sqrt(3) * current_a * line.resistance_ohm * self.config.power_factor
                child_voltage = np.clip(voltage_by_node[line.parent] - voltage_drop_v, 0, None)
                voltage_by_node[line.child] = child_voltage
                loading_percent = apparent_kva / line.capacity_kw * 100
                records.append(pd.DataFrame({
                    "M_TIMESTAMP": pivot.index,
                    "parent": line.parent,
                    "child": line.child,
                    "active_power_kw": active_kw,
                    "apparent_power_kva": apparent_kva,
                    "current_a": current_a,
                    "loss_kw": loss_kw,
                    "voltage_v": child_voltage,
                    "voltage_pu": child_voltage / self.config.nominal_voltage_v,
                    "loading_percent": loading_percent,
                    "overloaded": loading_percent > 100,
                    "voltage_violation": child_voltage / self.config.nominal_voltage_v < self.config.minimum_allowed_voltage_pu,
                }))
                remaining.remove(line)
                progress = True
            if not progress:
                raise ValueError("The generated grid is not a connected radial network")
        return pd.concat(records, ignore_index=True)


class StormLikePublisher:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def create(self, consumer_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        aggregate = consumer_frame.groupby("M_TIMESTAMP", as_index=False)["net_load_kw"].sum()
        truth = aggregate["net_load_kw"].to_numpy()
        scale = np.maximum(np.abs(truth), 1.0)
        measured = truth + self.rng.normal(0, self.config.measurement_noise_std_fraction * scale)
        bottom_up = 0.96 * truth + 0.25 + self.rng.normal(0, self.config.bottom_up_noise_std_fraction * scale)
        labels = np.zeros(len(truth), dtype=int)
        point_events = self.rng.random(len(truth)) < self.config.anomaly_probability
        measured[point_events] += self.rng.choice([-1, 1], point_events.sum()) * self.rng.uniform(0.25, 0.65, point_events.sum()) * scale[point_events]
        labels[point_events] = 1
        steps_per_day = 1440 // self.config.interval_minutes
        n_switches = max(1, self.rng.poisson(self.config.days * self.config.switch_event_probability_per_day))
        for _ in range(n_switches):
            start = int(self.rng.integers(steps_per_day, max(steps_per_day + 1, len(truth) - steps_per_day)))
            duration = int(self.rng.integers(8, min(steps_per_day, len(truth) - start) + 1))
            end = min(len(truth), start + duration)
            measured[start:end] += self.rng.choice([-1, 1]) * self.rng.uniform(0.12, 0.32) * scale[start:end]
            labels[start:end] = 1
            boundary = self.config.uncertain_boundary_steps
            labels[max(0, start - boundary):start] = 5
            labels[end:min(len(labels), end + boundary)] = 5
        missing = self.rng.random(len(truth)) < self.config.missing_probability
        measured[missing] = np.nan
        bottom_up[missing] = np.nan
        x = pd.DataFrame({
            "M_TIMESTAMP": aggregate["M_TIMESTAMP"],
            "S_original": measured,
            "BU_original": bottom_up,
            "missing": missing.astype(int),
        })
        y = pd.DataFrame({"M_TIMESTAMP": aggregate["M_TIMESTAMP"], "label": labels})
        return x, y


def draw_plots(output: Path, consumers: list[Consumer], grid: RadialGrid, x: pd.DataFrame, y: pd.DataFrame, lines: pd.DataFrame) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    merged = x.merge(y, on="M_TIMESTAMP")
    residual = merged["S_original"] - merged["BU_original"]

    graph = nx.DiGraph()
    graph.add_node("SUBSTATION", kind="substation")
    for consumer in consumers:
        graph.add_node(consumer.consumer_id, kind=consumer.consumer_type)
    graph.add_edges_from((line.parent, line.child) for line in grid.lines)
    colors = {"substation": "#d62728", "home": "#1f77b4", "commercial": "#ff7f0e", "industrial": "#2ca02c"}
    pos = nx.spring_layout(graph, seed=11)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(graph, pos, node_color=[colors[graph.nodes[n]["kind"]] for n in graph], node_size=900, arrows=True, font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(l.parent, l.child): f"{l.distance_km:.2f} km\n{l.capacity_kw:.0f} kW" for l in grid.lines}, font_size=6)
    plt.title("Synthetic radial distribution network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "network.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(merged["M_TIMESTAMP"], merged["S_original"], label="S_original: measured", linewidth=0.8)
    ax.plot(merged["M_TIMESTAMP"], merged["BU_original"], label="BU_original: bottom-up", linewidth=0.8, alpha=0.8)
    events = merged[merged["label"] == 1]
    uncertain = merged[merged["label"] == 5]
    ax.scatter(events["M_TIMESTAMP"], events["S_original"], s=10, color="crimson", label="label 1")
    ax.scatter(uncertain["M_TIMESTAMP"], uncertain["S_original"], s=10, color="darkorange", label="label 5")
    ax.set(title="STORM-like synthetic substation series", xlabel="UTC time", ylabel="Apparent load (synthetic kVA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "storm_timeseries.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(merged["M_TIMESTAMP"], residual, linewidth=0.8, color="#2ca02c")
    ax.scatter(events["M_TIMESTAMP"], residual[events.index], s=10, color="crimson")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(title="Residual: S_original - BU_original", xlabel="UTC time", ylabel="Residual")
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_and_events.png", dpi=180)
    plt.close(fig)

    complete = merged.dropna(subset=["S_original"]).copy()
    complete["time"] = complete["M_TIMESTAMP"].dt.strftime("%H:%M")
    daily = complete.groupby("time")["S_original"].agg(["mean", "std"])
    order = pd.date_range("2000-01-01", periods=len(daily), freq=f"{int(1440 / len(daily))}min").strftime("%H:%M")
    daily = daily.reindex(order)
    positions = np.arange(len(daily))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(positions, daily["mean"], color="#1f77b4")
    ax.fill_between(positions, (daily["mean"] - daily["std"]).to_numpy(), (daily["mean"] + daily["std"]).to_numpy(), alpha=0.2)
    tick_positions = positions[::max(1, len(positions) // 8)]
    ax.set_xticks(tick_positions, daily.index[tick_positions], rotation=45)
    ax.set(title="Average daily load profile", xlabel="Time of day", ylabel="Apparent load")
    fig.tight_layout()
    fig.savefig(plot_dir / "daily_profile.png", dpi=180)
    plt.close(fig)

    worst_loading = lines.groupby("M_TIMESTAMP")["loading_percent"].max()
    minimum_voltage = lines.groupby("M_TIMESTAMP")["voltage_pu"].min()
    total_loss = lines.groupby("M_TIMESTAMP")["loss_kw"].sum()
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(worst_loading.index, worst_loading, linewidth=0.8)
    axes[0].axhline(100, color="crimson", linestyle="--")
    axes[0].set_ylabel("Max loading (%)")
    axes[1].plot(minimum_voltage.index, minimum_voltage, linewidth=0.8)
    axes[1].axhline(0.90, color="crimson", linestyle="--")
    axes[1].set_ylabel("Min voltage (p.u.)")
    axes[2].plot(total_loss.index, total_loss, linewidth=0.8)
    axes[2].set_ylabel("Total loss (kW)")
    axes[2].set_xlabel("UTC time")
    fig.suptitle("Approximate physical diagnostics")
    fig.tight_layout()
    fig.savefig(plot_dir / "physical_diagnostics.png", dpi=180)
    plt.close(fig)


def run(config: SimulationConfig, output: Path) -> None:
    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    timestamps = pd.date_range(config.start, periods=config.days * 1440 // config.interval_minutes, freq=f"{config.interval_minutes}min")
    context = ContextGenerator(rng).generate(timestamps)
    consumers = [ConsumerFactory.random(f"C{i:03d}", rng) for i in range(1, config.n_consumers + 1)]
    demand_model = RuleBasedDemandModel(rng)
    consumer_frame = pd.concat([demand_model.calculate(consumer, context) for consumer in consumers], ignore_index=True)
    grid = RadialGrid.random(consumers, config, rng)
    line_frame = grid.evaluate(consumer_frame)
    x, y = StormLikePublisher(config, rng).create(consumer_frame)

    x_dir, y_dir = output / "Train" / "X", output / "Train" / "y"
    x_dir.mkdir(parents=True, exist_ok=True)
    y_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(x_dir / "1.csv", index=False)
    y.to_csv(y_dir / "1.csv", index=False)
    pd.DataFrame(asdict(c) for c in consumers).to_csv(output / "consumer_metadata.csv", index=False)
    consumer_frame.to_csv(output / "consumer_timeseries.csv", index=False)
    context.to_csv(output / "context_timeseries.csv", index=False)
    pd.DataFrame([{"node_id": "SUBSTATION", "node_type": "substation"}] + [{"node_id": c.consumer_id, "node_type": c.consumer_type} for c in consumers]).to_csv(output / "network_nodes.csv", index=False)
    pd.DataFrame([{**asdict(line), "resistance_ohm": line.resistance_ohm} for line in grid.lines]).to_csv(output / "network_edges.csv", index=False)
    line_frame.to_csv(output / "line_timeseries.csv", index=False)
    (output / "simulation_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    summary = {
        "timestamps": len(x),
        "consumers": len(consumers),
        "normal_rows": int((y["label"] == 0).sum()),
        "event_rows": int((y["label"] == 1).sum()),
        "uncertain_rows": int((y["label"] == 5).sum()),
        "missing_rows": int(x["missing"].sum()),
        "overloaded_line_rows": int(line_frame["overloaded"].sum()),
        "voltage_violation_rows": int(line_frame["voltage_violation"].sum()),
        "maximum_line_loading_percent": float(line_frame["loading_percent"].max()),
        "minimum_voltage_pu": float(line_frame["voltage_pu"].min()),
        "total_energy_loss_kwh": float(line_frame["loss_kw"].sum() * config.interval_minutes / 60),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_plots(output, consumers, grid, x, y, line_frame)
    print(json.dumps(summary, indent=2))
    print(f"Results saved in: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a random physics-aware STORM-like energy dataset without ML.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--consumers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(SimulationConfig(seed=args.seed, days=args.days, n_consumers=args.consumers), args.output)
