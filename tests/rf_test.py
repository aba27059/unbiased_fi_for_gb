import time

import numpy as np

from algorithms import CartRandomForestRegressor
from algorithms.Tree import TreeVisualizer
from tests.test_dataset_creator import create_x_y

if __name__ == '__main__':
    np.random.seed(3)
    X, y = create_x_y()
    start = time.time()
    reg = CartRandomForestRegressor()
    reg.fit(X, y)
    end = time.time()
    print(end - start)
    tree_vis = TreeVisualizer()
    tree_vis.plot(reg.trees[0].root)
    print(reg.predict(X))