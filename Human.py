
class Human():
    def __init__(self,game):
        self.game = game
    def get_move(self):
        while True:
            parts = input("Please input a coordinates of the index for the board (e.g. '1 2'):\n").split()
            if len(parts) != 2:
                print("Wrong input format! Please enter again.")
                continue
            r = int(parts[0]) 
            c = int(parts[1]) 
            if not self.game.isValid(r,c):
                print("invalid move!!!")
                continue
            return r, c
    
# test = Human(3)
# print(test.get_move())