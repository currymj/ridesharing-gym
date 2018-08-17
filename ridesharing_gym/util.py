import numpy as np
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

    def get_dist(self, origin, end):
        """
        Computes the distance between two points using Euclidean dist.
        """
        a = self.get_coord(origin)
        b = self.get_coord(end)
        dist = np.linalg.norm(a-b)

        return dist
