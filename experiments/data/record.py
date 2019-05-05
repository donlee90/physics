import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from mlagents.envs import UnityEnvironment

def random_action(num_agents, action_size, action_type='discrete'):
    ''' Returns a random action '''
    if action_type == 'continuous':
        action = np.random.randn(num_agents, action_size[0])
    else:
        action = np.column_stack([np.random.randint(0, action_size[i], size=(num_agents)) for i in range(len(action_size))])
    return action


def main(args):
    train_mode =True
    env = UnityEnvironment()

    # Set the default brain to work with
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    action_size = brain.vector_action_space_size
    action_type = brain.vector_action_space_type

    # Create dir for recording
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Main loop
    for episode in range(args.num_episodes):
        # Initialize episode
        brain_info = env.reset(train_mode=train_mode)[default_brain]
        done = False
        actions = []
        observations = [brain_info.vector_observations[0]]

        # Create dir for this episode
        record_path = os.path.join(args.outdir, "episode-%d" % (episode))
        if not os.path.exists(record_path):
            os.mkdir(record_path)

        # Run this episode
        while not done:
            num_agents = len(brain_info.agents)

            action = random_action(num_agents, action_size, action_type)

            brain_info = env.step(action)[default_brain]
            obs = brain_info.vector_observations

            done = brain_info.local_done[0]

            if not done:
                actions.append(action[0])
                observations.append(obs[0])

        # Record episode history
        actions = np.array(actions)
        observations = np.array(observations)

        np.savetxt(os.path.join(record_path, "actions.txt"), actions)
        np.savetxt(os.path.join(record_path, "observations.txt"), observations)
        print("Episode {}: {} actions, {} observations".format(episode,
                                                               len(actions),
                                                               len(observations)))

    # Done
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to record')
    parser.add_argument('--outdir', type=str, default='record',
                        help='Output directory')
    args = parser.parse_args()
    main(args)
