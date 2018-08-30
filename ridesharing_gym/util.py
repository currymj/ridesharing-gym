import numpy as np
import csv
def undiscounted_episode_returns(rewards):
    """
    Given a sequence of rewards, returns the sequence
    of the undiscounted returns (G_t) at each time step.
    """
    return np.cumsum(rewards[::-1])[::-1]

def discounted_episode_returns(rewards, gamma=0.999):
    """
    Given a sequence of rewards, returns the sequence
    of the discounted returns (G_t) at each time step,
    with discount rate gamma (default 0.999).
    """

    length = len(rewards)
    discounts = [gamma**x for x in range(length)]
    result = [np.dot(discounts[:length-i], rewards[i:]) for i in range(length)]
    return result


def read_requests_csv(filename, grid):
    weights = {}
    probabilities = {}
    with open(filename, 'r') as file:
        rr = csv.reader(file)
        for line in rr:
            request_type, weight, request_prob = line
            startX, startY, endX, endY = request_type.split(';')
            start_loc = grid.get_loc_from_coords([int(startX), int(startY)])
            end_loc = grid.get_loc_from_coords([int(endX), int(endY)])
            weights[(start_loc, end_loc)] = float(weight)
            probabilities[(start_loc, end_loc)] = float(request_prob)

    return (probabilities, weights)

class GridParameters:

    def __init__(self, width, length, capacity):
        self.width = width
        self.length = length
        self.grid_size = self.width * self.length
        self.capacity = capacity

    def get_loc(self, origin, action):
        """
        Return location index given a center and action.
        If exceeds boundary of the grid, returns -1.
        """
        loc = origin

        if action == 2: # N
            loc = origin - self.width
        if action == 3: # S
            loc = origin + self.width
        if action == 4: # E
            loc = origin + 1
        if action == 5: # W
            loc = origin - 1
        if loc < 0 or loc >= self.grid_size:
            loc = -1

        return loc

    def get_coord(self, location):
        """
        Computes the longitudes and latitudes of the given location.
        """
        lat = location % self.width
        long = location % self.length

        return np.array([lat, long])

    def get_loc_from_coords(self, coord_array):
        lat = coord_array[0]
        lon = coord_array[1]
        return (lon*self.width) + lat

    def get_dist(self, origin, end):
        """
        Computes the distance between two points using Euclidean dist.
        """
        a = self.get_coord(origin)
        b = self.get_coord(end)
        dist = np.linalg.norm(a-b)

        return dist
