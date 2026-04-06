class QLearningAgent:
    """Heuristic Q-style policy (discrete, reactive)."""
    def act(self, mean_traffic_pat_var, byte_variance):
        if mean_traffic_pat_var > 1.1 or byte_variance > 5e10:
            return "increase"
        if mean_traffic_pat_var < 0.5 and byte_variance < 1e10:
            return "decrease"
        return "keep"

class PPOAgent:
    """Smooth PPO-like policy (stability-biased)."""
    def act(self, mean_traffic_pat_var, byte_variance):
        score = 0.6 * mean_traffic_pat_var + 0.4 * (byte_variance / 5e10)
        if score > 1.2:
            return "increase"
        if score < 0.6:
            return "decrease"
        return "keep"
