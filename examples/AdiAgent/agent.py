from random import choice
from stable_baselines3 import PPO

class DiceAdventureAgent:
    """
    Provides a uniform interface to connect agents to Dice Adventure environment.
    Developers must implement the load() and take_action() functions.
    - init():         Initialize any needed variables.
    - take_action():  Determines which action to take given a state (dict) and list of actions. Note that your
                      agent does not need to use the list of actions, it is just provided for convenience.
    """
    def __init__(self, model_filename):
        self.model = PPO.load(model_filename)

    def take_action(self, state, actions):
        """
        Given a game state and list of actions, the agent should determine which action to take and return a
        string representation of the action.
        :param state:   (dict) A 'Dice Adventure' game state
        :param actions: (list) A list of string action names
        :return:        (string) An action from the 'actions' list
        """
        print("take actions")
        print("state:")
        print(state)
        action_probs, _ = self.model.predict(state, deterministic=True)
        print("action_probs:")
        print(action_probs)
        print("_states: " )
        print(_)
        action_idx = action_probs.argmax()
        print(action_idx)
        return actions[action_idx]
