def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        
        # Define Spaces IMMEDIATELY so SB3 does not crash
        fleet_size = self.config['fleet']['size']
        
        # Observation space matches _get_obs(): (50 * 2) + 1 + 6 = 107
        obs_dim = (fleet_size * 2) + 1 + 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.enable_v2g = self.config.get('enable_v2g', False) 
        # Safely extract c_max; default to 50.0 if not found
        stations = self.config.get('charging stations', [])
        self.c_max = stations[0]['max port power'] if stations else 50.0
        
        # low_action = np.zeros((fleet_size, 2))
        # high_action = np.ones((fleet_size, 2))
        # if self.enable_v2g:
        #     low_action[:, 1] = -self.c_max 
        # high_action[:, 1] = self.c_max
        # self.action_space = gym.spaces.Box(low=low_action, high=high_action, shape=(fleet_size, 2), dtype=np.float32)

        low_action = numpy.zeros((len(self.fleet), 2))
        high_action = numpy.ones((len(self.fleet), 2))
        if self.enable_v2g:
            low_action[:, 1] = -1.0  # Normalized lower bound
        high_action[:, 1] = 1.0      # Normalized upper bound

        self.lagrangian_multiplier = 0.0 # Starts at 0, inflates automatically if constraints are violated
        self.reward_type = self.config.get('reward_type', 'economics').lower()
        
        self.action_space = gym.spaces.Box(low=low_action, high=high_action, shape=(len(self.fleet), 2), dtype=numpy.float32)