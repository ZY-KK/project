import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose: int):
        super(TensorboardCallback).__init__(verbose=verbose)


    def _on_step(self) -> bool:
        




        return True