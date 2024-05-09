import numpy as np
from gymnasium.utils import seeding



class oneVOnePrompter():
    
    def __init__(self):
        self._np_random = None

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)