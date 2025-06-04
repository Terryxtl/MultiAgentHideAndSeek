import pygame


class Obstacle:
    def __init__(self, x, y, width, height, obstacle_id="wall", color=(80, 80, 80)):
        self.id = obstacle_id
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.color = color
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def get_bounds(self):
        return self.x, self.x + self.width, self.y, self.y + self.height
