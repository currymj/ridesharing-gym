import numpy as np
def undiscounted_episode_returns(rewards):
    """
    Given a sequence of rewards, returns the sequence
    of the undiscounted returns (G_t) at each time step.
    """
    return np.cumsum(rewards[::-1])[::-1]

# the following method could probably be made
# much, much more efficient
def discounted_episode_returns(rewards, gamma=0.999):
    """
    Given a sequence of rewards, returns the sequence
    of the discounted returns (G_t) at each time step,
    with discount rate gamma (default 0.999).
    """

    def cumulative_discounts(length):
        res = np.zeros(length)
        res[0] = 1.0
        for i in range(1, length):
            res[i] = gamma * res[i-1]
        return res

    result = np.zeros_like(rewards)
    for i in range(len(rewards)):
        discounts = cumulative_discounts(len(rewards) - i)
        result[i] = np.sum( discounts * rewards[i:])

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
        a = get_coord(origin)
        b = get_coord(end)
        dist = np.linalg.norm(a-b)

        return dist
