# SlitherIO-AI

## Installation Guide
To run this program we need to install OpenAI gym at this [link](https://github.com/openai/gym) and OpenAI universe at this [link](https://github.com/openai/universe). We are using Python 3.5, but Python 2.7 should also work for this project. Just make sure to use `pip3` for Python 3.5 and `pip` for Python 2.7.

Next, make sure to install the proper python packages for this project.
```shell
pip3 install scipy neat-python==0.8 argparse
```

## Running the Program

There are two files that are used to solve universe environments: `network_config` and `main.py`. 

This program can be run with the following parameters:

`--max-steps`: The max number of steps to take per genome (timeout)

`--episodes`: The number of times to run a single genome. This takes the average fitness score over all episodes for one genome

`--render`: Renders the game while the algorithm is learning

`--generations`: The number of generations to evolve the network

`--checkpoint`: Uses a checkpoint to start the simulation

`--num-cores`: The number cores on your computer for parallel execution (not in `--render` mode)

Ex: 
```shell
python3 main.py --max-steps=10000 --generations=50 --render
```