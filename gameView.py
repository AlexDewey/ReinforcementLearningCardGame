import pygame


class GameView:

    def __init__(self):
        pygame.init()
        screen = pygame.display.set_mode((640, 480))
        screen.fill((255, 255, 255))

        # White (255, 255, 255)

    def refreshBoard(self, boardGiven):
        x = 1
