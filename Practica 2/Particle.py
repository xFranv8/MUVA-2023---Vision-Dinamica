class Particle:
    def __init__(self, x: int, y: int, weight: float):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.weight = weight