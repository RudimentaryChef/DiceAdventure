from stable_baselines3 import PPO
from game.env.dice_adventure_python_env import DiceAdventurePythonEnv
from time import sleep
from random import seed
from threading import Thread

PLAYERS = ["Dwarf", "Giant", "Human"]
SERVER = "unity"
PLAYER = False


def main():
    players = [p for p in PLAYERS if p != PLAYER]
    processes = [
        Thread(target=play, args=(p,))
        for p in players
    ]
    for p in processes:
        p.start()


def play(p):
    pmap = {"Dwarf": "Giant", "Giant": "Dwarf", "Human": "Human"}
    p = pmap[p]
    seed(0)
    action_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'wait',
                           5: 'submit', 6: 'pinga', 7: 'pingb', 8: 'pingc', 9: 'pingd', 10: 'undo'}
    action_map_rev = {v: k for k, v in action_map.items()}

    model_filename = "train/7/model/dice_adventure_ppo_modelchkpt-279.zip"
    model = PPO.load(model_filename)

    env = get_env(p, model_filename)
    env.load_threshold = 1
    env.model = model
    # state = env.get_state()

    obs = env.reset()[0]
    while True:
        action, _states = model.predict(obs)
        # up -> left | down -> right | left -> up | right -> down
        # action_remap = {"up": "left", "down": "right", "left": "up", "right": "down"}
        # Temporarily remap directional actions
        #action_remap = {3: 0, 4: 2, 0: 3, 2: 4}
        #if int(action) in action_remap:
        #    action = action_remap[int(action)]
        #print(f"Sending action: {action_map[int(action)]} for player: {player}")
        obs, rewards, dones, truncated, info = env.step(action)


def get_env(init_player, model_dir):
    return DiceAdventurePythonEnv(id_=0,
                             level=1,
                             player=init_player,
                             model_number=7,
                             model_dir=model_dir,
                             server="unity",
                             automate_players=True,
                             random_players=False
                             )


if __name__ == "__main__":
    main()
