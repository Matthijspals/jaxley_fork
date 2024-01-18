from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import vmap


class Channel:
    channel_name = None
    channel_params = None
    channel_states = None

    def __init__(self):
        self.vmaped_update_states = vmap(self.update_states, in_axes=(0, None, 0, 0))
        self.vmapped_compute_current = vmap(
            self.compute_current, in_axes=(None, 0, None)
        )

    def get_name(self):
        return self.channel_name

    def change_name(self, new_name):
        self.channel_params = {
            new_name + key[key.find("_") :]: value
            for key, value in self.channel_params.items()
        }
        self.channel_states = {
            new_name + key[key.find("_") :]: value
            for key, value in self.channel_states.items()
        }
        self.channel_name = new_name

    @staticmethod
    def update_states(
        u, dt, voltages, params
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        pass

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        pass
