from collections import namedtuple
import math
import numpy as np

TIME_RESOLUTION = DT = 1e-3  # sec
SPACE_RESOLUTION = 1e-3  # meters

Arena = namedtuple("Arena", ["x_height", "y_width"])


class Arena:
    def __init__(self, x_height, y_width):
        self.x_height = x_height
        self.y_width = y_width

    def does_contain(self, position):
        return (0 < position[0] < self.x_height) and (0 < position[1] < self.y_width)

    def random_position(self):
        return np.random.uniform(0, self.x_height), np.random.uniform(0, self.y_width)

    def distance(self, position):
        """
        For positions inside the arena, distance=0
        Otherwise, distance is Euclidean distance from the arena
        """
        if self.does_contain(position):
            return 0
        if 0 < position[0] < self.x_height:
            return -position[1] if position[1] < 0 else position[1] - self.y_width
        if 0 < position[1] < self.y_width:
            return -position[0] if position[0] < 0 else position[0] - self.x_height

        return min(
            distance(position, (0, 0)),
            distance(position, (0, self.y_width)),
            distance(position, (self.x_height, 0)),
            distance(position, (self.x_height, self.y_width))
        )


def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
