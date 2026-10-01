from stage1_simulator.energy_simulator import SimulationConfig

def test_config_validation():
    cfg = SimulationConfig(days=1, n_consumers=3)
    cfg.validate()
