import random
class Random_Bot():
    def __init__(self, game):
        self.game = game

    def get_move(self):
        ValidMoves = self.game.getValidMoves()
        result = random.choice(ValidMoves)
        r,c = result
        print(result)
        return r,c
       

        
