class Config:
    def __init__(
        self,
        samples: int,
        dataset: str = "USNW",
        agent: str = "PPO",
        init_buffer: int = 20,
        min_buffer: int = 10,
        max_buffer: int = 200,
        print_every: int = 1,
        show_io: bool = False,
        head: int = 3,
    ):
        # Runtime workload
        self.SAMPLES = samples

        # Dataset options: "CSE" or "UNSW"
        self.DATASET = dataset

        # Buffer parameters
        self.INIT_BUFFER = init_buffer
        self.MIN_BUFFER = min_buffer
        self.MAX_BUFFER = max_buffer

        # RL agent: "Q" or "PPO"
        self.RL_AGENT = agent

        # Logging
        self.PRINT_EVERY = print_every
        self.SHOW_IO = show_io
        self.HEAD = head
