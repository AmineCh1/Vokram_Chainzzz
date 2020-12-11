import numpy as np



def distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    return the euclidean distance between cities at positions x and y
    """
    return np.linalg.norm(x - y)


def mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    return the mean euclidean distance between cities at position x and y
    """
    return np.array([(x[0] + y[0]) / 2, (x[1] + y[1]) / 2])