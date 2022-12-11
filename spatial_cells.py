import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math
import numpy as np

from arena import SPACE_RESOLUTION, Arena, distance


class SpatialCell(ABC):
    def did_spike(self, location, dt):
        return self.num_of_spikes(location, dt) > 0

    def num_of_spikes(self, location, dt):
        return np.random.poisson(self.firing_rate(location) * dt)

    @abstractmethod
    def firing_rate(self, location):
        assert NotImplementedError()


class GridCell(SpatialCell):
    def __init__(self, arena, phase, orientation, field_max_rate, field_width, field_distance):
        """
        `phase` is a 2-d position in the arena
        `orientation` is measured as a <pi angle between the x axis and the vector that connects two nearby fields
        """
        self.arena = arena
        self.phase = phase
        self.orientation = orientation
        self.field_max_rate = field_max_rate
        self.field_width = field_width
        self.field_distance = field_distance
        self.fields = []
        self.initiate_firing_fields()

    def initiate_firing_fields(self):
        approx_x_fields = self.arena.x_height / (self.field_distance * math.cos(self.orientation))
        assert approx_x_fields**2 < 75  # we're not crazy

        done = set()
        worklist = [(0, 0)]
        while len(worklist) > 0:
            field_index = worklist.pop()
            if field_index in done:
                continue
            done.add(field_index)

            # location of (1, 0) relative to (0, 0)
            direction_x = np.array([math.cos(self.orientation), math.sin(self.orientation)])
            # location of (0, 1) relative to (0, 0)
            direction_y = np.array(
                [math.cos(self.orientation + 1 / 3 * math.pi), math.sin(self.orientation + 1 / 3 * math.pi)]
            )

            field_location = (
                self.phase
                + self.field_distance * direction_x * field_index[0]
                + self.field_distance * direction_y * field_index[1]
            )
            if self.arena.distance(field_location) > 2 * self.field_width:
                continue

            self.fields.append(field_location)
            worklist.append((field_index[0] + 1, field_index[1]))
            worklist.append((field_index[0] - 1, field_index[1]))
            worklist.append((field_index[0], field_index[1] + 1))
            worklist.append((field_index[0], field_index[1] - 1))

    def plot_field_centers(self):
        plt.scatter(*(zip(*self.fields)))
        plt.axis("equal")
        plt.xlim([0, self.arena.x_height])
        plt.ylim([0, self.arena.y_width])
        plt.show()

    def plot_firing_map(self):
        x_dynamic_range = int(self.arena.x_height / SPACE_RESOLUTION)
        y_dynamic_range = int(self.arena.y_width / SPACE_RESOLUTION)

        firing_map = np.zeros([x_dynamic_range, y_dynamic_range])
        for x in range(0, x_dynamic_range):
            for y in range(0, y_dynamic_range):
                firing_map[x][y] = self.firing_rate((x * SPACE_RESOLUTION, y * SPACE_RESOLUTION))

        plt.imshow(firing_map)
        plt.show()

    def firing_rate(self, position):
        # TODO which is the better model? nearest or additive?
        return self._firing_rate_nearest(position)

    def _firing_rate_nearest(self, position):
        dist = min(distance(field, position) for field in self.fields)
        field_rate_normalization = self.field_max_rate
        return field_rate_normalization * math.exp(-0.5 * (dist / self.field_width) ** 2)

    def _firing_rate_additive(self, position):
        rate = 0
        for field in self.fields:
            dist = distance(field, position)
            field_rate_normalization = self.field_max_rate
            field_rate = field_rate_normalization * math.exp(-0.5 * (dist / self.field_width) ** 2)
            rate += field_rate
        return rate


if __name__ == "__main__":
    arena = Arena(1, 1)
    grid_cell = GridCell(arena, (0.1, 0.1), 0.3, 10, 0.08, 0.3)
    print(grid_cell.firing_rate((0.1, 0.1)))
    print(grid_cell.firing_rate((0.2, 0.2)))
    grid_cell.plot_firing_map()
