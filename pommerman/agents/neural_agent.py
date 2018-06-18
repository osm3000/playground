"""
This agent just implements a generic NN, and query it everytime in order to get an action.
"""
import torch
import base_agent
import numpy as np
import random

def featurize(obs):
    """
    This function will return one big vector, having all the observeration stored.
    """
    def make_np_float(feature):
        return np.array(feature).astype(np.float32)
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

class NeuralAgent(base_agent.BaseAgent):
    def __init__(self, *args, **kwargs):
        super(NeuralAgent, self).__init__(*args, **kwargs)

        self._recently_visited_positions = [] # Record the history of the agent path
        self._recently_visited_length = 6 # Length of the history memory
        # Keep track of the previous direction to help with the enemy standoffs.
        self._prev_direction = None

        self.NN_weights = None # Will store the weights of the NN, which will operate for one generation

    def act(self, obs, action_space):
        all_obs = featurize(obs)

        # Demo Part --> the rest of this func will be filled later.
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]

        return random.choice(directions).value
