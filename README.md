This is a supplementary repository for our final project submission.

Within this repository:

cmetric.py:
Takes in a states.txt file and produces a .npy file with the first and second derivate of closeness centrality measures for each state. This .npy file can be used as labels to train an aggression network.

train.py:
The main runner file for our training harness. Within train.py, tune hyperparams {num_epochs, learning_rate, etc.}. Simply run `python3 train.py`

model.py:
Defines our simply pytorch multilayer perceptron.

utils.py:
Loads relevant state/label data from local .npy files


