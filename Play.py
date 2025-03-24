from Human import Human
from RandomBot import Random_Bot,Greedy_Bot
from Dots_and_Box import DotsAndBox
from DeepLearning import *
from Alpha.MCTS import MCTSPlayer
from arg import *
import os

args_CNN['train'] = False
args_Conv2Plus1D['train'] = False
args_ConvLSTM['train'] = False

args_LSTM['train'] = False
args_LSTM['load_model_name'] = 'LSTM_model_4x4_18.h5'

size_m = m
size_n = n

game = DotsAndBox(size_m,size_n)
p1 = [Human(game=game), 'Human']
p2 = [Random_Bot(game=game), 'random']
p3 = [Greedy_Bot(game=game), 'greedy']
p4 = [MCTSPlayer(num_simulations=100, exploration_weight=1.5, max_depth=5), 'MCTS']
p4[0].game_state = game

p6 = [LSTM_BOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_LSTM), 'LSTM']
p7 = [Conv2Plus1D_BOT(input_size_m=size_m,input_size_n=size_n,game=game, args=args_Conv2Plus1D), 'Conv2Plus1D']

def self_play(player1, player2):
    """
    讓兩個玩家對戰，回傳比賽結果
    """
    game.NewGame()
    return game.play(player1, player2)

def record_result(file, game_num, bot1_name, bot2_name, bot1_win, bot2_win):
    """
    記錄對戰結果到檔案
    """
    file.write(f"Game {game_num}\n")
    file.write(f"{bot1_name} win: {bot1_win}\n")
    file.write(f"{bot2_name} win: {bot2_win}\n")
    file.write("-" * 76 + "\n")

def dual(n_game, bot1, bot2, bot1_name, bot2_name):
    """
    讓兩個 Bot 進行多場對戰，記錄勝負結果
    """
    print(f"{bot1_name} VS {bot2_name}".center(76))
    os.makedirs("game_record", exist_ok=True)  # 確保資料夾存在
    file_path = f'game_record/{bot1_name} VS {bot2_name}.txt'

    bot1_win, bot2_win = 0, 0

    with open(file_path, "a") as f:
        f.write(f"{bot1_name} VS {bot2_name}".center(76) + "\n")

        for i in range(1, n_game + 1):
            print(f"Game {i}")

            # 第一局
            result = self_play(bot1, bot2)
            if result == -1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
            elif result == 1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
            else:
                print('Draw!')
            # 記錄結果
            record_result(f, i, bot1_name, bot2_name, bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"{bot2_name} win: {bot2_win}")
            print("-" * 76)
            
        # 先後手交換
        for i in range(1, n_game + 1):
            print(f"Game {i}")
            result = self_play(bot2, bot1)
            if result == 1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
            elif result == -1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
            else:
                print('Draw!')

            # 記錄結果
            record_result(f, i, bot1_name, bot2_name, bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"{bot2_name} win: {bot2_win}")
            print("-" * 76)
    
def main():
    # args_Res['train'] = False
    # p5 = [ResnetBOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_Res), 'resnet']
    # game.play(p5[0], p2[0])
    for ver in range(1,20):
        # ver = 15
        args_Res['train'] = False
        args_Res['load_model_name'] = f'Resnet_model_4x4_{ver}.h5'
        p5 = [ResnetBOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_Res), 'resnet']

        # args_ConvLSTM['train'] = False
        # args_ConvLSTM['load_model_name'] = f'ConvLSTM_model_4x4_{ver}.h5'
        # p8 = [ConvLSTM_BOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_ConvLSTM), 'ConvLSTM']
        
        dual(n_game=10,
            bot1=p5[0],
            bot1_name=p5[1]+f'_{ver}',
            bot2=p3[0],
            bot2_name=p3[1])
    
    
if __name__ == "__main__":
    main()