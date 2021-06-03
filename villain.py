class Villain():
    def __init__(self,qmaze, pos, dir):
        self.maze = qmaze.maze
        self.pos = pos
        self.row= pos[0]
        self.col = pos[1]
        if dir == "V": self.dir = (1,0)
        if dir == "H": self.dir = (0,1)
        qmaze.villains_visited[self] = [self.pos]
        # maze.best_villains_visited[self] = []

    def move(self):
        prev_pos = self.pos
        new_pos =  tuple(map(sum,zip(self.pos,self.dir)))
        if new_pos[0] < 0 or new_pos[0] >= self.maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.maze.shape[1]:
            self.dir = (-1*self.dir[0],-1*self.dir[1])
        elif self.maze[new_pos[0], new_pos[1]] == 0.0:
            self.dir = (-1*self.dir[0],-1*self.dir[1])
        self.pos = tuple(map(sum,zip(self.pos,self.dir)))
        self.row = self.pos[0]
        self.col = self.pos[1]
