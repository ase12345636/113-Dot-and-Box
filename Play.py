from Human import Human
from RandomBot import Random_Bot,Greedy_Bot
from Dots_and_Box import DotsAndBox
from DeepLearning6x6 import *
from Alpha.MCTS import MCTSPlayer
from Alpha.AlphaBeta import AlphaBetaPlayer
from arg import *
import os

size_m = m
size_n = n

game = DotsAndBox(size_m,size_n)
p1 = [Human(game=game), 'Human']
p2 = [Random_Bot(game=game), 'random']
p3 = [Greedy_Bot(game=game), 'greedy']
p4 = [MCTSPlayer(num_simulations=100, exploration_weight=1.5, max_depth=5,selfFirst=True), 'MCTS']
p4[0].game_state = game

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

        first_winning_rate = f"first: {round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"

        second_bot1_win = 0
        second_bot2_win = 0
        # 先後手交換
        for i in range(1, n_game + 1):
            print(f"Game {i}")
            result = self_play(bot2, bot1)
            if result == 1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
                second_bot1_win += 1
            elif result == -1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
                second_bot2_win += 1
            else:
                print('Draw!')

            # 記錄結果
            record_result(f, i, bot1_name, bot2_name, bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"{bot2_name} win: {bot2_win}")
            print("-" * 76)


        second_winning_rate = f"second: {round((second_bot1_win / (second_bot1_win + second_bot2_win)) * 100, 2)}%\n"

        print(bot1_win)
        print(bot2_win)
        if bot2_win == 0:
            winning_rate = "100%"
        else:
            print((bot1_win / (bot1_win + bot2_win)))
            winning_rate = f"{round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        print(winning_rate)

        f.write(first_winning_rate)
        f.write(second_winning_rate)
        f.write(winning_rate)

def dualAB(n_game, bot1, bot1_name,depth=3):
    """
    讓兩個 Bot 進行多場對戰，記錄勝負結果
    """
    print(f"{bot1_name} VS AlphaBeta".center(76))
    os.makedirs("game_record", exist_ok=True)  # 確保資料夾存在
    file_path = f'game_record/{bot1_name} VS AlphaBeta.txt'

    bot1_win, bot2_win = 0, 0

    with open(file_path, "a") as f:
        f.write(f"{bot1_name} VS AlphaBeta".center(76) + "\n")

        bot2 = AlphaBetaPlayer(1,game,depth)
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
            record_result(f, i, bot1_name, "AlphaBeta", bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"AlphaBeta win: {bot2_win}")
            print("-" * 76)
            
        first_winning_rate = f"first: {round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        
        second_bot1_win = 0
        second_bot2_win = 0
        bot2 = AlphaBetaPlayer(-1, game, depth)
        # 先後手交換
        for i in range(1, n_game + 1):
            print(f"Game {i}")
            result = self_play(bot2, bot1)
            if result == 1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
                second_bot1_win += 1
            elif result == -1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
                second_bot2_win += 1
            else:
                print('Draw!')

            # 記錄結果
            record_result(f, i, bot1_name, "AlphaBeta", bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"AlphaBeta win: {bot2_win}")
            print("-" * 76)
        
        
        second_winning_rate = f"second: {round((second_bot1_win / (second_bot1_win + second_bot2_win)) * 100, 2)}%\n"
        
        print(bot1_win)
        print(bot2_win)
        if bot2_win == 0:
            winning_rate = "100%"
        else:
            print((bot1_win / (bot1_win + bot2_win)))
            winning_rate = f"{round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        print(winning_rate)
        
        f.write(first_winning_rate)
        f.write(second_winning_rate)
        f.write(winning_rate)


def main():
    # args_Res['train'] = False
    # args_Res['load_model_name'] = f'Resnet_model_4x4_31.h5'
    # p5 = [ResnetBOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_Res), 'resnet']
    ab1 = AlphaBetaPlayer(-1, game, 4)
    ab2 = AlphaBetaPlayer(-1, game, 4)
    # game.play(p3[0], ab2)
    for ver in range(1,6):
        args_Res['train'] = False
        args_Res['load_model_name'] = f'Resnet_model_{size_m}x{size_n}_{ver}.h5'
        p5 = [ResnetBOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_Res), f'resnet_{size_m}x{size_n}']

        dual(n_game=50,
            bot1=p5[0],
            bot1_name=p5[1]+f'_{ver}',
            bot2=p2[0],
            bot2_name=p2[1])

    #     dualAB(n_game=20,
    #         bot1=p5[0],
    #         bot1_name=p5[1]+f'_{ver}',
    #         depth=3)



if __name__ == "__main__":
    main()