import argparse
import gym
import os
import math
import numpy as np
import universe  # register the universe environments
from scipy import ndimage
from neat import nn, population, statistics, parallel

### User Params ###

# The name of the game to solve
game_name = 'internet.SlitherIO-v0'

# Pixel location of the center screen
center_x = 270
center_y = 235
# Game screen corners
# ul_x = 20
# ul_y = 85
# lr_x = 520
# lr_y = 385
# The above bound might be too big so I just +-100 the center_x and center_y
ul_x = 170
ul_y = 135
lr_x = 370
lr_y = 335

# The snake moves to the directing of the mouse
# but to output the direction to a neural network we need to break the output to more discrete values

# the radius is the distance from the head of the snake to the mouse pointer(in pixel)
radius = 30
# This is the number of points we want around the head of the snake
# Ex: With 8 points where the mouse can be positioned around the head of the snake
# Note the distance from the point to the head is the same for all
#       *
#     *   *
#   *   s   *
#     *   *
#       *
# You can add more resolution to this if you want but it may increase learning time
resolution_points = 8
degree_per_slice = 360//resolution_points

# Available actions in the game
action_sheet = []

# We put all mouse positions in the action_sheet
for point in range(resolution_points):
    degree = point*degree_per_slice
    x_value_offset = radius * math.sin(math.radians(degree))
    x_value_offset = radius * math.cos(math.radians(degree))
    coord = universe.spaces.PointerEvent(center_x + x_value_offset, center_y + x_value_offset, 0)
    action_sheet.append(coord)

### End User Params ###


parser = argparse.ArgumentParser(description='OpenAI Gym Solver')
parser.add_argument('--max-steps', dest='max_steps', type=int, default=1000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=1,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--render', action='store_true')
parser.add_argument('--generations', type=int, default=50,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--num-cores', dest="numCores", type=int, default=4,
                    help="The number cores on your computer for parallel execution")
args = parser.parse_args()


def downsample_and_flatten(vision):
    # Each cell in the matrix has a length 3 array
    # Because each pixel has RGB values
    new_obs = np.array(vision)
    # We average the RGB values into a single value(Greyscale)
    new_obs = new_obs.mean(axis=2)
    # Then we group pixels of 5*5 blocks and average them into a single value
    # So we can further reduce the input amount
    new_obs = np.array(block_mean(new_obs, 5))
    # Next we turn the flatten the matrix into a 1-dimension array 
    new_obs = new_obs.flatten()
    return new_obs


def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res


def get_actions(outputs):
    actions = []
    for i in range(len(outputs)):
        if outputs[i] > 0.5:
            actions.append(action_sheet[i])
        else:
            actions.append(universe.spaces.PointerEvent(center_x, center_y, 0))

    return actions


def simulate_species(net, env, episodes=1, steps=5000, render=False):
    fitnesses = []
    for runs in range(episodes):
        # Input has the information about the screen and the current state of the game
        # We can get the screen pixels from it
        inputs = my_env.reset()
        cum_reward = 0.0
        for j in range(steps):
            if inputs[0] is not None:
                # Here we pass in the screen corners
                # We use the bounds because the whole screen is the browser window
                # We just want the game window
                new_obs = downsample_and_flatten(inputs[0]["vision"][ul_y:lr_y, ul_x:lr_x])
                # The new_obs will be a one dimension array
                outputs = net.serial_activate(new_obs)
            else:
                # If there is no input
                # Then just do nothing?
                # TODO: The snake will always move around even if the mouse is pointed to the head
                # IF we don't get any input maybe we can just let the snake go in a random direction
                outputs = np.zeros(len(action_sheet)).tolist()
            inputs, reward, done, _ = env.step([get_actions(outputs) for ob in inputs])
            if render:
                env.render()
            if done[0]:
                break
            cum_reward += reward[0]

        fitnesses.append(cum_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness


def train_network(env):

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, args.episodes, args.max_steps, render=args.render)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness

    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'network_config')
    pop = population.Population(config_path)
    # Load checkpoint
    if args.checkpoint:
        pop.load_checkpoint(args.checkpoint)
    # Start simulation
    pop.run(eval_fitness, args.generations)

    pop.save_checkpoint("checkpoint")

    # Log statistics.
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    import pickle
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    raw_input("Press Enter to run the best genome...")
    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net, env, 1, args.max_steps, render=True)

my_env = gym.make(game_name)
my_env.configure(remotes=1)  # automatically creates a local docker container
observation_n = my_env.reset()
if args.render:
    my_env.render()

train_network(my_env)
