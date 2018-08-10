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