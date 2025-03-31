import math
import copy
import random
from Dots_and_Box import DotsAndBox
from RandomBot import *
# from DeepLearning import ResnetBOT
# from arg import args_Res
# args_Res['train'] = True
# args_Res['load_model_name'] = 'Resnet_model_4x4_76.h5'

# args_select_method_bot = {
#     'load_model_name': 'Resnet_model_6x6_2.h5',
#     'train': True
# }


class AlphaBetaPlayer:
    def __init__(self, symbol, game: DotsAndBox, max_depth=3):
        self.symbol = symbol
        self.game = game
        self.max_depth = max_depth
        # self.select_meth_bot = ResnetBOT(self.game.input_m,self.game.input_n,self.game,args_Res)
        self.select_meth_bot = Greedy_Bot(self.game)
    def get_move(self):
        best_value = -math.inf
        best_move = None
        
        visited_pos = []
        valid_moves = self.game.getValidMoves()
        
        non_think_moves = ((self.game.input_m-1)*(self.game.input_n) + (self.game.input_m)*(self.game.input_n-1))//2 + 5
        
        for _ in range(len(valid_moves)):
            self.select_meth_bot.game = self.game
            move = self.select_meth_bot.get_move()[0]
            one_d_len = self.game.board_rows_nums * self.game.board_cols_nums
            
            
            if (len(valid_moves)>non_think_moves) and move:   #遊戲前期無需太多思考，直接return
                
                r,c = move
                position = r*self.game.board_cols_nums+c
                tmp=np.zeros(one_d_len)
                tmp[position] = 1.0
                board = copy.deepcopy(self.game.board)
                
                return move, [board, tmp, self.game.current_player]
            
            attempt = 0
            while move in visited_pos and attempt<12:
                move = random.choice(valid_moves)
                attempt+=1
                
            visited_pos.append(move)
            if move:
                new_game = copy.deepcopy(self.game)
                new_game.history.clear()
                new_game.make_move(*move)
                
                if self.symbol != new_game.current_player: #有輪流
                    move_value = self.alphabeta(new_game, 0, -math.inf, math.inf, False)
                else:   #還是自己
                    move_value = self.alphabeta(new_game, 0, -math.inf, math.inf, True)
                
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
        if not best_move:
            best_move =  random.choice(self.game.getValidMoves())
            print(f"random best_move {best_move} ")
        
        r,c = best_move
        position = r*self.game.board_cols_nums+c
        tmp=np.zeros(one_d_len)
        tmp[position] = 1.0
        board = copy.deepcopy(self.game.board)
        print(f"best move: {best_move}, best value: {best_value}")
        return best_move, [board, tmp, self.game.current_player]

    def evaluate(self, game):
        """ 評估函數：根據當前棋盤返回數值評估 """
        if game.GetWinner() == self.symbol:
            return 10000
        elif game.GetWinner() == -self.symbol:
            return -10000
        else:
            score_diff = game.p1_scores - game.p2_scores
            if self.symbol == 1:
                score_diff*=-1
            
            # 增加先手玩家的分數
            if self.symbol == -1:  # 先手玩家
                score_diff += 5  # 可根据情况调整加分
            
            return score_diff*100
        
        return 0  # 平局或未結束時，評分為 0
    
    def alphabeta(self, game, depth, alpha, beta, maximizing):
        # 終止條件
        if depth >= self.max_depth:
            # print(f"max depth:{depth}")
            return self.evaluate(game)  # 返回棋盤評估值
        
        winner = game.GetWinner()
        if winner is not None:
            return self.evaluate(game)  # 游戏结束，直接返回评估值
        
        
        if maximizing:
            max_eval = -math.inf
            valid_moves = game.getValidMoves()
            for move in valid_moves:
                new_game = copy.deepcopy(game)
                last_player = new_game.current_player
                new_game.make_move(*move)
                
                if last_player != new_game.current_player: #有輪流
                    eval = self.alphabeta(new_game, depth + 1, alpha, beta, False)
                else:   #還是自己
                    eval = self.alphabeta(new_game, depth + 1, alpha, beta, True)
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if beta <= alpha:
                    # print(f"Beta剪枝 beta:{beta}, alpha:{alpha}, depth: {depth}")
                    break  # Beta 剪枝
            return max_eval
        else:
            min_eval = math.inf
            valid_moves = game.getValidMoves()
            for move in valid_moves:
                new_game = copy.deepcopy(game)
                last_player = new_game.current_player
                new_game.make_move(*move)
                if last_player != new_game.current_player: #有輪流
                    eval = self.alphabeta(new_game, depth + 1, alpha, beta, True)
                else:   #還是自己
                    eval = self.alphabeta(new_game, depth + 1, alpha, beta, False)
                
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    # print(f"Alpha剪枝 beta:{beta}, alpha:{alpha}, depth: {depth}")
                    break  # Alpha 剪枝
            return min_eval