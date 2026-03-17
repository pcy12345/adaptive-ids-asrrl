Run examples:

  python main.py 10000
  python main.py 50000 --dataset UNSW --agent Q
  python main.py 10000 --show-io --print-every 5 --head 5

What you get:
- Per-window print: buffer size, entropy/variance, RL action, threshold, final label, and whether label/buffer changed
- Optional detailed I/O printout for every methodology/component (--show-io)
- End summary: how often buffer changes, label jitter, and fluctuation counts
