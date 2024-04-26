from game.dice_adventure import DiceAdventure
import game.env.unity_socket as unity_socket
from gymnasium import Env
import json
from json import loads
import examples.AdiAgent.rewards as rewards
from random import choice
from datetime import datetime
from gymnasium import spaces
import numpy as np
import re


class DiceAdventurePythonEnv(Env):
    """
    Implements a custom gyn environment for the Dice Adventure game.
    """
    def __init__(self,
                 player="Dwarf",
                 id_=0,
                 train_mode=False,
                 server="local",
                 state_version="full",
                 automate_players=True,
                 **kwargs):
        """
        Init function for Dice Adventure gym environment.
        :param player:      (string) The player that will be used to play the game.
        :param id_:         (int) An optional ID parameter to distinguish this environment from others.
        :param train_mode:  (bool) A helper parameter to switch between training mode and play mode. When we test agents,
                                   we will use a "play" mode, where the step function simply takes an action and returns
                                   the next state.
        :param server:      (string) Determines which game version to use. Can be one of {local, unity}.
        :param kwargs:      (dict) Additional keyword arguments to pass into Dice Adventure game. Only applies when
                                   'server' is 'local'.
        """
        self.config = loads(open("game/config/main_config.json", "r").read())
        self.player = player
        self.id = id_
        self.kwargs = kwargs

        ##################
        # STATE SETTINGS #
        ##################
        self.state_version = state_version
        self.mask_radii = {"Dwarf": self.config["OBJECT_INFO"]["OBJECT_CODES"]["1S"]["SIGHT_RANGE"],
                           "Giant": self.config["OBJECT_INFO"]["OBJECT_CODES"]["2S"]["SIGHT_RANGE"],
                           "Human": self.config["OBJECT_INFO"]["OBJECT_CODES"]["3S"]["SIGHT_RANGE"]}
        self.max_mask_radius = max(self.mask_radii.values())
        self.action_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'wait',
                           5: 'submit', 6: 'pinga', 7: 'pingb', 8: 'pingc', 9: 'pingd', 10: 'undo'}
        self.observation_object_positions = self.config["GYM_ENVIRONMENT"]["OBSERVATION"]["OBJECT_POSITIONS"]
        self.object_size_mappings = self.config["OBJECT_INFO"]["ENEMIES"]["ENEMY_SIZE_MAPPING"]
        self.pin_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.reward_codes = self.config["GYM_ENVIRONMENT"]["REWARD"]["CODES"]
        self.players = ["Dwarf", "Giant", "Human"]

        ##################
        # TRAIN SETTINGS #
        ##################
        self.train_mode = train_mode
        self.automate_players = automate_players
        self.track_metrics = False

        ###################
        # SERVER SETTINGS #
        ###################
        self.server = server
        self.unity_socket_url = self.config["GYM_ENVIRONMENT"]["UNITY"]["URL"]
        self.game = None

        if self.server == "local":
            self.game = DiceAdventure(**self.kwargs)

        ##################
        # AGENT SETTINGS #
        ##################

        self.model_type = "aggressive"

        num_actions = len(self.action_map)
        self.action_space = spaces.Discrete(num_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.mask_size = self.max_mask_radius * 2 + 1
        vector_len = (self.mask_size * self.mask_size * len(
            set(self.observation_object_positions.values())) * 4) + 6
        print("vector_len")
        print(vector_len)
        self.observation_space = spaces.Box(low=-5, high=100,
                                            shape=(vector_len,), dtype=np.float32)

    def step(self, action):
        """
        Applies the given action to the game. Determines the next observation and reward,
        whether the training should terminate, whether training should be truncated, and
        additional info.
        :param action:  (string) The action produced by the agent
        :return:        (dict, float, bool, bool, dict) See description
        """
        action = int(action)

        state = self.get_state()
        # Execute action and get next state
        game_action = self.action_map[action]
        next_state = self.execute_action(self.player, action)

        # get player info
        pstate_1 = self.get_obj_from_scene_by_type(state, self.player)
        pstate_2 = self.get_obj_from_scene_by_type(next_state, self.player)

        reward = self.get_reward(pstate_1, pstate_2, state, next_state)

        # Simulate other players
        if self.automate_players:
            self.play_others(game_action, state, next_state)
            next_state = self.get_state()

        # new_obs, reward, terminated, truncated, info
        terminated = next_state["status"] == "Done"
        if terminated:
            new_obs, info = self.reset()
        else:
            new_obs = self.get_observation(self.get_state())
            info = {}
        truncated = False
        # print(type(new_obs))

        return new_obs, reward, terminated, truncated, info

    def close(self):
        """
        close() function from standard gym environment. Not implemented.
        :return: N/A
        """
        pass

    def render(self, mode='console'):
        """
        Prints the current board state of the game. Only applies when `self.server` is 'local'.
        :param mode: (string) Determines the mode to use (not used)
        :return: N/A
        """
        if self.server == "local":
            self.game.render()

    def reset(self, **kwargs):
        """
        Resets the game. Only applies when `self.server` is 'local'.
        :param kwargs:  (dict) Additional arguments to pass into local game server
        :return:        (dict, dict) The initial state when the game is reset, An empty 'info' dict
        """
        if self.server == "local":
            self.game = DiceAdventure(**self.kwargs)
        obs = self.get_observation(self.get_state())
        return obs, {}

    def execute_action(self, player, game_action):
        """
        Executes the given action for the given player.
        :param player:      (string) The player that should take the action
        :param game_action: (string) The action to take
        :return:            (dict) The resulting state after taking the given action
        """
        if self.server == "local":
            self.game.execute_action(player, game_action)
            next_state = self.get_state()
        else:
            url = self.unity_socket_url.format(player.lower())
            next_state = unity_socket.execute_action(url, game_action)
        return next_state

    def get_state(self, player=None, version=None, server=None):
        """
        Gets the current state of the game.
        :param player: (string) The player whose perspective will be used to collect the state. Can be one of
                                {Dwarf, Giant, Human}.
        :param version: (string) The level of visibility. Can be one of {full, player, fow}
        :param server: (string) Determines whether to get state from Python version or unity version of game. Can be
                                one of {local, unity}.
        :return: (dict) The state of the game

        The state is always given from the perspective of a player and defines how much of the level the
        player can currently "see". The following state version options define how much information this function
        returns.
        - [full]:   Returns all objects and player stats. This ignores the 'player' parameter.

        - [player]: Returns all objects in the current sight range of the player. Limited information is provided about
                    other players present in the state.

        - [fow]:    Stands for Fog of War. In the Unity version of the game, you can see a visibility mask for each
                    character. Black positions have not been observed. Gray positions have been observed but are not
                    currently in the player's view. This option returns all objects in the current sight range (view) of
                    the player plus objects in positions that the player has seen before. Note that any object that can
                    move (such as monsters and other players) are only returned when they are in the player's current
                    view.
        """
        version = version if version else self.state_version
        player = player if player else self.player
        server = server if server else self.server

        if server == "local":
            state = self.game.get_state(player, version)
        else:
            url = self.unity_socket_url.format(player)
            state = unity_socket.get_state(url, version)

        return state

    ####################
    # CUSTOM FUNCTIONS #
    ####################
    def play_others(self, game_action, state, next_state):
        # Play as other players
        for p in self.players:
            if p != self.player:
                # Only force submit on other characters if case where self.player clicking submit does not
                # change the game phase (otherwise, these players will just forfeit their turns immediately)
                if game_action == "submit" \
                        and state["content"]["gameData"]["currentPhase"] == next_state["content"]["gameData"]["currentPhase"]:
                    a = game_action
                else:
                    a = choice(list(self.action_map.values()))
                # print(f"Other Player: {p}: Action: {a}")
                _ = self.execute_action(p, a)
                # next_state = self.get_state()

    def get_reward(self, p1, p2, state, next_state):
        # Get reward
        """
        Rewards:
        1. Player getting goal (0.5)
        2. Player getting to tower after getting all players have collected goals (1.0)
        3. Player winning combat (0.5) - TODO
        4. Player placing pin on object (0.1) - TODO

        Penalties:
        1. Player losing health (-0.2)
        2. Player not moving (-0.1)
        """
        if self.model_type == "aggressive":
            return self.get_reward_aggressive(p1, p2, state, next_state)
        r = 0
        reward_types = []

        # Player getting goal
        g1 = self.get_obj_from_scene_by_type(state, "shrine")
        g2 = self.get_obj_from_scene_by_type(next_state, "shrine")
        if rewards.goal_reached(g1, g2, state, next_state):
            reward_types.append(self.reward_codes["0"])
            r += 1
        # Players getting to tower after getting all goals
        if rewards.check_new_level(state, next_state):
            reward_types.append(self.reward_codes["1"])
            r += 1
        # Player winning combat
        # if self.check_combat_outcome():
        #     r += .5
        # Player placing pin on object
        # if self.check_pin_placement(p2, next_state):
        #     r += .1
        # Player losing health
        if rewards.health_lost_or_dead(p1, p2):
            reward_types.append(self.reward_codes["2"])
            r -= .2

        # Player not moving
        if not rewards.has_moved(p1, p2):
            reward_types.append(self.reward_codes["3"])
            r -= .1

        if self.track_metrics:
            # [timestep, timestamp, player, game, level, reward_type, reward]
            self.rewards_tracker.append([self.time_steps,
                                         datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                         self.player,
                                         str(self.num_games),
                                         state["content"]["gameData"]["level"],
                                         ",".join(reward_types),
                                         str(r)
                                         ])
        return r

    def get_reward_aggressive(self, p1, p2, state, next_state):
        # Get reward
        """
        Rewards:
        1. Player getting goal (0.5)
        2. Player getting to tower after getting all players have collected goals (1.0)
        3. Player winning combat (0.5) - TODO

        Penalties:
        1. Player losing health (-0.2)
        2. Player not moving (-0.1)
        """
        r = 0
        reward_types = []

        # Player getting goal
        g1 = self.get_obj_from_scene_by_type(state, "shrine")
        g2 = self.get_obj_from_scene_by_type(next_state, "shrine")
        if rewards.goal_reached(g1, g2, state, next_state):
            reward_types.append(self.reward_codes["0"])
            r += 1
        # Players getting to tower after getting all goals
        if rewards.check_new_level(state, next_state):
            reward_types.append(self.reward_codes["1"])
            r += 1
        # Player winning combat
        # if self.check_combat_outcome():
        #     r += .5
        # Player placing pin on object
        # if self.check_pin_placement(p2, next_state):
        #     r += .1
        # Player losing health

        if rewards.health_lost_or_dead(p1, p2):
            reward_types.append(self.reward_codes["2"])
            r -= .1

        # Player not moving
        if not rewards.has_moved(p1, p2):
            reward_types.append(self.reward_codes["3"])
            r -= .1
        # Added if less enemies now
        if rewards.enemy_reduced(state, next_state):
            # reward_types.append(self.reward_codes["4"])
            r += .35
        # Note: Need to figure out how to do this for each individual entity
        # TODO
        # if not rewards.closer_to_entity(p1,p2,):
        if not rewards.closer_to_entity(state, next_state, "Human", "monster"):
            # reward_types.append(self.reward_codes["4"])
            r -= -0.1
        if not rewards.closer_to_entity(state, next_state, "Dwarf", "monster"):
            # reward_types.append(self.reward_codes["4"])
            r -= -0.1
        if not rewards.closer_to_entity(state, next_state, "Giant", "monster"):
            # reward_types.append(self.reward_codes["4"])
            r -= -0.1
        if self.track_metrics:
            # [timestep, timestamp, player, game, level, reward_type, reward]
            self.rewards_tracker.append([self.time_steps,
                                         datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                         self.player,
                                         str(self.num_games),
                                         state["content"]["gameData"]["level"],
                                         ",".join(reward_types),
                                         str(r)
                                         ])
        return r


    @staticmethod
    def get_obj_from_scene_by_type(state, obj_type):
        o = None
        for ele in state["content"]["scene"]:
            if ele.get("entityType") == obj_type:
                o = ele
                break
        # if o is None:
        #     print(state)
        return o

    def get_observation(self, state, player=None):
        """
        Constructs an array observation for agent based on state. Dimensions:
        1. self.mask x self.mask (1-2)
        2. len(self.observation_type_positions) (3)
        3. 4 (4) - max number of object types is 4 [i.e., M4]
        4. six additional state variables
        Total Est.: 7x7x10x4+6= 1006
        :param state:
        :return:
        """
        if player is None:
            player = self.player
        x, y, player_info = self.parse_player_state_data(state, player)

        x_bound_upper = x + self.mask_radii[self.player]
        x_bound_lower = x - self.mask_radii[self.player]
        y_bound_upper = y + self.mask_radii[self.player]
        y_bound_lower = y - self.mask_radii[self.player]

        grid = np.zeros((self.mask_size, self.mask_size, len(set(self.observation_object_positions.values())), 4))
        for obj in state["content"]["scene"]:
            if obj["entityType"] in self.observation_object_positions and obj["x"] and obj["y"]:
                if x_bound_lower <= obj["x"] <= x_bound_upper and \
                        y_bound_lower <= obj["y"] <= y_bound_upper:
                    other_x = self.mask_radii[self.player] - (x - obj["x"])
                    other_y = self.mask_radii[self.player] - (y - obj["y"])
                    # For pins and enemies, determine which type for version
                    if obj["entityType"] == "pin":
                        version = self.pin_mapping[obj["objectCode"][1]]
                    # For enemies, determine which type for version
                    elif re.match("(monster|trap|stone)", obj["entityType"].lower()):
                        # elif obj["name"] in ["Monster", "Trap", "Stone"]:
                        version = self.object_size_mappings[obj["entityType"].split("_")[0]]
                    # All other objects have one version
                    else:
                        version = 0
                    grid[other_x][other_y][self.observation_object_positions[obj["entityType"]]][version] = 1

        return np.concatenate((np.ndarray.flatten(grid), np.ndarray.flatten(player_info)))

    @staticmethod
    def parse_player_state_data(state, player):
        # Locate player and their shrine in scene
        player_obj = None
        shrine_obj = None
        #print("hello")
        #print("state")
        #print(state)
        #print("state: content")
        #print(state["content"])
        #print("state: content: scene")
        #print(state["content"]["scene"])
        for obj in state["content"]["scene"]:
            if player_obj and shrine_obj:
                break
            if obj["entityType"] == player:
                player_obj = obj
            elif obj["entityType"] == "shrine" and obj.get("character") == player:
                shrine_obj = obj

        state_map = {
            "actionPoints": 0,
            "health": 1,
            "dead": 2,
            "reached": 3,
            "pinCursorX": 4,
            "pinCursorY": 5
        }
        value_map = {True: 1, False: 0, None: 0}
        player_info = np.zeros((len(state_map,)))

        for field in state_map:
            if field == "reached":
                data = shrine_obj[field]
            else:
                data = player_obj[field]
            # Value should come from mapping
            if data in [True, False, None]:
                player_info[state_map[field]] = value_map[data]
            # Otherwise, value is scalar
            else:
                player_info[state_map[field]] = data
        return player_obj["x"], player_obj["y"], player_info

