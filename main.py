import argparse
from config import Config
from data import generate_data
from pipeline import run_ids

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive IDS (synthetic CSE/UNSW-like)")
    p.add_argument("num_samples", type=int, help="Number of flows to simulate (e.g., 10000, 50000, 100000)")
    p.add_argument("--dataset", choices=["CSE","UNSW","CIC2017"], default="CSE", help="Synthetic dataset profile")
    p.add_argument("--agent", choices=["Q","PPO"], default="PPO", help="RL controller type")
    p.add_argument("--init-buffer", type=int, default=20, help="Initial buffer size")
    p.add_argument("--min-buffer", type=int, default=10, help="Minimum buffer size")
    p.add_argument("--max-buffer", type=int, default=200, help="Maximum buffer size")
    p.add_argument("--print-every", type=int, default=1, help="Print every N completed windows")
    p.add_argument("--show-io", action="store_true", help="Print input/output at EACH component (more verbose)")
    p.add_argument("--head", type=int, default=3, help="How many rows of the window to preview in I/O prints")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        samples=args.num_samples,
        dataset=args.dataset,
        agent=args.agent,
        init_buffer=args.init_buffer,
        min_buffer=args.min_buffer,
        max_buffer=args.max_buffer,
        print_every=args.print_every,
        show_io=args.show_io,
        head=args.head,
    )
    df = generate_data(cfg)
    run_ids(df, cfg)

if __name__ == "__main__":
    main()
