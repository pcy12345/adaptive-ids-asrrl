from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

FEATURES = ["flow_duration", "pkt_rate", "byte_rate", "entropy"]

@dataclass
class BufferStats:
    mean_entropy: float
    byte_variance: float
    mean_pkt_rate: float
    mean_byte_rate: float

class AdaptiveBuffer:
    def __init__(self, cfg):
        self.size = cfg.INIT_BUFFER
        self.min = cfg.MIN_BUFFER
        self.max = cfg.MAX_BUFFER
        self.buf = deque(maxlen=self.size)

        # metrics
        self.resize_events = 0
        self.resize_history = [(0, self.size)]  # (window_index, size)

    def add(self, row):
        self.buf.append(row)

    def ready(self):
        return len(self.buf) == self.size

    def resize(self, action: str, window_index: int):
        old = self.size
        if action == "increase":
            self.size = min(self.size + 10, self.max)
        elif action == "decrease":
            self.size = max(self.size - 5, self.min)

        if old != self.size:
            self.resize_events += 1
            self.resize_history.append((window_index, self.size))
            self.buf = deque(self.buf, maxlen=self.size)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.buf)

class Normalizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def transform(self, window_df: pd.DataFrame) -> Tuple[pd.DataFrame, BufferStats]:
        X = window_df[FEATURES]
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=[f"z_{c}" for c in FEATURES])

        stats = BufferStats(
            mean_entropy=float(window_df["entropy"].mean()),
            byte_variance=float(window_df["byte_rate"].var()),
            mean_pkt_rate=float(window_df["pkt_rate"].mean()),
            mean_byte_rate=float(window_df["byte_rate"].mean()),
        )
        return X_scaled_df, stats

class SymbolicModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=42)

    def train(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("[SymbolicModel] Training dataframe is empty.")
        X = df[FEATURES]
        y = df["label"].astype(int)
        self.model.fit(X, y)

    def infer(self, window_df: pd.DataFrame) -> Tuple[pd.Series, float]:
        X = window_df[FEATURES]
        preds = self.model.predict(X)
        confidence = float(preds.mean())  # proxy: % of flows predicted attack in window
        return pd.Series(preds, name="sym_pred"), confidence
