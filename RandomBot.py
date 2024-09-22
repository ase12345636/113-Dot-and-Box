import random
from Dots_and_Box import DotsAndBox

def GreedAlg(board,m,n,ValidMoves):
    def check_box(next_board):
        box_filled = False
        for i in range(m - 1):
            for j in range(n - 1):
                box_i = 2*i + 1
                box_j = 2*j + 1
                # 檢查該方格的四條邊是否都不為 0
                if (next_board[box_i][box_j] == 8 and
                    next_board[box_i-1][box_j] != 0 and
                    next_board[box_i+1][box_j] != 0 and
                    next_board[box_i][box_j-1] != 0 and
                    next_board[box_i][box_j+1] != 0):
                    box_filled = True
        return box_filled
    for ValidMove in ValidMoves:
        r,c = ValidMove
        next_board = board
        next_board[r][c] = 1
        
        if check_box(next_board):
            next_board[r][c] = 0
            return r,c
        else:
            next_board[r][c] = 0
    

class Greedy_Bot():
    def __init__(self, game: DotsAndBox):
        self.game = game
        self.next_board = self.game.board

    def check_box(self):
        box_filled = False
        for i in range(self.game.input_m - 1):
            for j in range(self.game.input_n - 1):
                box_i = 2*i + 1
                box_j = 2*j + 1
                # 檢查該方格的四條邊是否都不為 0
                if (self.next_board[box_i][box_j] == 8 and
                    self.next_board[box_i-1][box_j] != 0 and
                    self.next_board[box_i+1][box_j] != 0 and
                    self.next_board[box_i][box_j-1] != 0 and
                    self.next_board[box_i][box_j+1] != 0):
                    box_filled = True
        return box_filled
    
    def get_move(self):
        ValidMoves = self.game.getValidMoves()
        for ValidMove in ValidMoves:
            r,c = ValidMove
            self.next_board = self.game.board
            if self.game.current_player == -1:
                self.next_board[r][c] = -1
            else:
                self.next_board[r][c] = 1
            
            if self.check_box():
                self.next_board[r][c] = 0
                return r,c
            else:
                self.next_board[r][c] = 0
        result = random.choice(ValidMoves)
        r,c = result
        return r,c
            

class Random_Bot():
    def __init__(self, game: DotsAndBox):
        self.game = game

    def get_move(self):
        ValidMoves = self.game.getValidMoves()
        result = random.choice(ValidMoves)
        r,c = result
        print(result)
        return r,c
       

        
