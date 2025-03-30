import random
from Dots_and_Box import DotsAndBox
from Dots_and_Box import ANSI_string
import numpy as np
import copy

def check_box(next_board):
    box_filled = False
    m = (len(next_board)+1)//2
    n = (len(next_board[0])+1)//2
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

def check_suicide(next_board):
    box_filled = False
    box_filled = False
    m = (len(next_board)+1)//2
    n = (len(next_board[0])+1)//2
    for i in range(m - 1):
        for j in range(n - 1):
            box_i = 2*i + 1
            box_j = 2*j + 1
            # 檢查該方格的四條邊是否都不為 0
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] == 0 and
                next_board[box_i+1][box_j] != 0 and
                next_board[box_i][box_j-1] != 0 and
                next_board[box_i][box_j+1] != 0):
                box_filled = True
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] != 0 and
                next_board[box_i+1][box_j] == 0 and
                next_board[box_i][box_j-1] != 0 and
                next_board[box_i][box_j+1] != 0):
                box_filled = True
            
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] != 0 and
                next_board[box_i+1][box_j] != 0 and
                next_board[box_i][box_j-1] == 0 and
                next_board[box_i][box_j+1] != 0):
                box_filled = True
            
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] != 0 and
                next_board[box_i+1][box_j] != 0 and
                next_board[box_i][box_j-1] != 0 and
                next_board[box_i][box_j+1] == 0):
                box_filled = True

    return box_filled
    

def GreedAlg(board,ValidMoves,verbose = False):
    for ValidMove in ValidMoves:
        r,c = ValidMove
        next_board = board
        next_board[r][c] = 1
        
        if check_box(next_board):
            if verbose:
                print(ANSI_string("greedy","green",None,True))
            next_board[r][c] = 0
            return r,c
        else:
            next_board[r][c] = 0
    
    while ValidMoves:
        r,c = random.choice(ValidMoves)
        next_board[r][c] = 1
        if check_suicide(next_board):
            ValidMoves.remove((r, c))
            next_board[r][c] = 0
        else:
            if verbose:
                print("not bad move")
            next_board[r][c] = 0
            return r,c
    if verbose:
        print(ANSI_string("bad move","red",None,True))
    return None

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
        greedy_move = GreedAlg(self.game.board,ValidMoves)
        if not greedy_move:
            greedy_move = random.choice(self.game.getValidMoves())

        r,c = greedy_move
        
        one_d_len = self.game.board_rows_nums * self.game.board_cols_nums
        
        position = r*self.game.board_cols_nums+c
        tmp=np.zeros(one_d_len)
        tmp[position] = 1.0
        board = copy.deepcopy(self.game.board)
        
        return greedy_move, [board, tmp, self.game.current_player]
            

class Random_Bot():
    def __init__(self, game: DotsAndBox):
        self.game = game

    def get_move(self):
        ValidMoves = self.game.getValidMoves()
        result = random.choice(ValidMoves)
        
        one_d_len = self.game.board_rows_nums * self.game.board_cols_nums
        r,c = result
        position = r*self.game.board_cols_nums+c
        tmp=np.zeros(one_d_len)
        tmp[position] = 1.0
        board = copy.deepcopy(self.game.board)
        print(f"Radom: {result}")
        return result, [board, tmp, self.game.current_player]
       

        
