from examples.AdiAgent.dice_adventure_python_env import DiceAdventurePythonEnv
from examples.AdiAgent.agent import DiceAdventureAgent

PLAYERS = ["Dwarf", "Giant", "Human"]
SERVER = "local"
ACTION_LIST = ["up", "down", "left", "right", "wait", "undo", "submit", "pinga", "pingb", "pingc", "pingd"]


def main():
    model_file = "train/7/model/dice_adventure_ppo_modelchkpt-16445.zip"
    # Load agent
    agent = DiceAdventureAgent(model_file)
    # Set up environment
    env = DiceAdventurePythonEnv(server=SERVER)
    state = env.reset()[0]

    while True:
        for p in PLAYERS:
            action = agent.take_action(state=state, actions=ACTION_LIST)
            state = env.execute_action(player=p, game_action=action)
            print(p)
            print(action)

        env.render()



if __name__ == "__main__":
    main()
