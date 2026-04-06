import pandas as pd
import numpy as np

def generate_data(cfg):
    """Generate dataset-faithful synthetic flows with label-correlated features.
    Columns: flow_duration, pkt_rate, byte_rate, entropy (variability proxy), label

    Attack flows have distinct feature signatures (higher variability, burstier
    traffic) so that the SymbolicModel can learn meaningful decision boundaries,
    mirroring the separability found in real UNSW-NB15, CSE-CIC-IDS-2018, and
    CIC-IDS2017 datasets.

    NOTE: The 'entropy' column is a traffic variability proxy, NOT Shannon
    information entropy. See real_data.py for the per-dataset derivation.
    """
    n = cfg.SAMPLES

    if cfg.DATASET == "CSE":
        # CSE-CIC-IDS2018-like: burstier, higher variance, 15% attack
        # Overlapping distributions mimic real-world noise
        labels = np.random.choice([0, 1], p=[0.85, 0.15], size=n)
        return pd.DataFrame({
            "flow_duration": np.where(labels == 0,
                np.random.exponential(3000, n),
                np.random.exponential(1200, n)),
            "pkt_rate": np.where(labels == 0,
                np.random.normal(1200, 600, n).clip(10),
                np.random.normal(2400, 700, n).clip(10)),
            "byte_rate": np.where(labels == 0,
                np.random.normal(1e6, 7e5, n).clip(1000),
                np.random.normal(2.8e6, 1.2e6, n).clip(1000)),
            "entropy": np.where(labels == 0,
                np.random.normal(0.45, 0.20, n).clip(0.05),
                np.random.normal(0.95, 0.25, n).clip(0.05)),
            "label": labels,
        })

    if cfg.DATASET == "CIC2017":
        # CIC-IDS2017-like: moderate traffic, mixed attack types, 20% attack
        labels = np.random.choice([0, 1], p=[0.80, 0.20], size=n)
        return pd.DataFrame({
            "flow_duration": np.where(labels == 0,
                np.random.normal(2000, 700, n).clip(50),
                np.random.normal(900, 500, n).clip(10)),
            "pkt_rate": np.where(labels == 0,
                np.random.normal(900, 400, n).clip(10),
                np.random.normal(2000, 600, n).clip(10)),
            "byte_rate": np.where(labels == 0,
                np.random.normal(7e5, 3e5, n).clip(1000),
                np.random.normal(1.8e6, 7e5, n).clip(1000)),
            "entropy": np.where(labels == 0,
                np.random.normal(0.42, 0.18, n).clip(0.05),
                np.random.normal(0.85, 0.22, n).clip(0.05)),
            "label": labels,
        })

    # UNSW-NB15-like: more stable, tighter distributions, 30% attack
    labels = np.random.choice([0, 1], p=[0.70, 0.30], size=n)
    return pd.DataFrame({
        "flow_duration": np.where(labels == 0,
            np.random.normal(1500, 400, n).clip(50),
            np.random.normal(700, 300, n).clip(10)),
        "pkt_rate": np.where(labels == 0,
            np.random.normal(600, 250, n).clip(10),
            np.random.normal(1400, 350, n).clip(10)),
        "byte_rate": np.where(labels == 0,
            np.random.normal(4e5, 1.5e5, n).clip(1000),
            np.random.normal(9e5, 2.5e5, n).clip(1000)),
        "entropy": np.where(labels == 0,
            np.random.normal(0.40, 0.15, n).clip(0.05),
            np.random.normal(0.75, 0.18, n).clip(0.05)),
        "label": labels,
    })
