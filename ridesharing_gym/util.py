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
