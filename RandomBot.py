import random
from Dots_and_Box import DotsAndBox 
class Random_Bot():
    def __init__(self, game: DotsAndBox):
        self.game = game

    def get_move(self):
        ValidMoves = self.game.getValidMoves()
        result = random.choice(ValidMoves)
        r,c = result
        print(result)
        return r,c
       

        
