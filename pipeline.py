from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from components import AdaptiveBuffer, Normalizer, SymbolicModel, FEATURES
from rl_agents import QLearningAgent, PPOAgent

@dataclass
class WindowLog:
    window_index: int
    end_row_index: int
    buffer_size: int
    mean_traffic_pat_var: float
    byte_variance: float
    confidence: float
    action: str
    threshold: float
    final_label: int
    label_changed: bool
    buffer_changed: bool
    traffic_pat_var_delta: float
    variance_delta: float

def _print_component_io(cfg, ingestion_row, window_df, X_scaled_df, sym_preds, meta):
    head = cfg.HEAD

    print("  [1] Ingestion INPUT (single flow):")
    print(ingestion_row.to_frame().T[FEATURES + ["label"]].head(1).to_string(index=False))

    print(f"  [2] AdaptiveBuffer OUTPUT (window df) head({head}) size={len(window_df)}:")
    print(window_df[FEATURES + ["label"]].head(head).to_string(index=False))

    print(f"  [3] Normalizer OUTPUT (z-scored) head({head}):")
    print(X_scaled_df.head(head).to_string(index=False))

    print(f"  [4] SymbolicModel OUTPUT head({head}): sym_pred + original label")
    tmp = window_df[["label"]].head(head).copy()
    tmp["sym_pred"] = sym_preds.head(head).values
    print(tmp.to_string(index=False))

    print("  [5] RL Controller OUTPUT:")
    print(f"     action={meta['action']} | threshold={meta['threshold']:.2f}")

    print("  [6] Decision OUTPUT:")
    print(f"     final_label={'ATTACK' if meta['final_label']==1 else 'BENIGN'}")
    print("  ------------------------------------------------------------")

def run_ids(df, cfg):
    buffer = AdaptiveBuffer(cfg)
    norm = Normalizer()
    sym = SymbolicModel()
    rl = PPOAgent() if cfg.RL_AGENT == "PPO" else QLearningAgent()

    threshold = 0.5
    sym.train(df)

    logs: List[WindowLog] = []
    prev_label: Optional[int] = None
    prev_tpv: Optional[float] = None
    prev_var: Optional[float] = None

    window_index = -1

    for i, row in df.iterrows():
        # [1] Ingestion: input row
        buffer.add(row)

        if not buffer.ready():
            continue

        window_index += 1

        # [2] Buffer output
        window_df = buffer.dataframe()

        # [3] Normalization output + stats
        X_scaled_df, stats = norm.transform(window_df)

        # [4] Symbolic inference output
        sym_preds, confidence = sym.infer(window_df)

        # [5] RL action (buffer/threshold control)
        action = rl.act(stats.mean_traffic_pat_var, stats.byte_variance)
        before_size = buffer.size
        buffer.resize(action, window_index)
        buffer_changed = (before_size != buffer.size)

        # adaptive threshold update (simple, visible)
        if confidence > 0.7:
            threshold = min(0.9, threshold + 0.05)
        elif confidence < 0.3:
            threshold = max(0.3, threshold - 0.05)

        # [6] Final label decision
        final_label = 1 if confidence >= threshold else 0
        label_changed = (prev_label is not None and final_label != prev_label)

        traffic_pat_var_delta = (stats.mean_traffic_pat_var - prev_tpv) if prev_tpv is not None else 0.0
        var_delta = (stats.byte_variance - prev_var) if prev_var is not None else 0.0

        logs.append(WindowLog(
            window_index=window_index,
            end_row_index=i,
            buffer_size=buffer.size,
            mean_traffic_pat_var=stats.mean_traffic_pat_var,
            byte_variance=stats.byte_variance,
            confidence=confidence,
            action=action,
            threshold=threshold,
            final_label=final_label,
            label_changed=label_changed,
            buffer_changed=buffer_changed,
            traffic_pat_var_delta=traffic_pat_var_delta,
            variance_delta=var_delta,
        ))

        prev_label = final_label
        prev_tpv = stats.mean_traffic_pat_var
        prev_var = stats.byte_variance

        # Printing
        if window_index % cfg.PRINT_EVERY == 0:
            print(f"[Window {window_index} @ row {i}] "
                  f"buf={buffer.size} "
                  f"traffic_pat_var={stats.mean_traffic_pat_var:.3f} (Δ{traffic_pat_var_delta:+.3f}) "
                  f"var={stats.byte_variance:.2e} (Δ{var_delta:+.2e}) "
                  f"conf={confidence:.2f} "
                  f"action={action} "
                  f"th={threshold:.2f} "
                  f"label={'ATTACK' if final_label else 'BENIGN'} "
                  f"label_change={'YES' if label_changed else 'no'} "
                  f"buf_change={'YES' if buffer_changed else 'no'}"
            )

            if cfg.SHOW_IO:
                meta = {"action": action, "threshold": threshold, "final_label": final_label}
                _print_component_io(cfg, row, window_df, X_scaled_df, sym_preds, meta)

    # ===== Summary / answers to your questions =====
    if not logs:
        print("No completed windows were processed. Try lowering --init-buffer or increasing num_samples.")
        return

    total_windows = len(logs)
    buf_changes = sum(1 for w in logs if w.buffer_changed)
    label_changes = sum(1 for w in logs if w.label_changed)

    # "Fluctuations" proxy: large deltas in traffic_pat_var/variance
    ent_abs = np.array([abs(w.traffic_pat_var_delta) for w in logs[1:]]) if total_windows > 1 else np.array([0.0])
    var_abs = np.array([abs(w.variance_delta) for w in logs[1:]]) if total_windows > 1 else np.array([0.0])

    ent_thresh = float(np.percentile(ent_abs, 90)) if ent_abs.size else 0.0
    var_thresh = float(np.percentile(var_abs, 90)) if var_abs.size else 0.0

    major_ent = sum(1 for w in logs[1:] if abs(w.traffic_pat_var_delta) >= ent_thresh and ent_thresh > 0)
    major_var = sum(1 for w in logs[1:] if abs(w.variance_delta) >= var_thresh and var_thresh > 0)

    print("\n==================== SUMMARY ====================")
    print(f"Dataset profile           : {cfg.DATASET}")
    print(f"RL agent                  : {cfg.RL_AGENT}")
    print(f"Flows simulated           : {len(df)}")
    print(f"Completed windows         : {total_windows}")
    print(f"Initial buffer size       : {cfg.INIT_BUFFER}")
    print(f"Final buffer size         : {logs[-1].buffer_size}")
    print(f"Buffer size changes       : {buf_changes} times")
    print(f"Buffer change frequency   : {buf_changes/total_windows:.3f} per window")
    print(f"Label changes (jitter)    : {label_changes} times")
    print(f"Label change frequency    : {label_changes/total_windows:.3f} per window")
    print(f"Did buffer changes affect labeling?")
    # crude association: % of label changes happening on windows with buffer change
    both = sum(1 for w in logs if w.label_changed and w.buffer_changed)
    print(f"  label_change AND buf_change: {both} windows ({(both/max(label_changes,1))*100:.1f}% of label changes)")

    print("Fluctuations (window-to-window)")
    print(f"  traffic_pat_var |Δ| 90th percentile threshold : {ent_thresh:.3f} -> major events: {major_ent}")
    print(f"  variance|Δ| 90th percentile threshold : {var_thresh:.2e} -> major events: {major_var}")

    print("Buffer size history (window_index -> buffer_size):")
    print("  " + ", ".join([f"{wi}:{bs}" for wi, bs in buffer.resize_history[:20]]) + (" ..." if len(buffer.resize_history) > 20 else ""))
    print("=================================================\n")
