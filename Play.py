from Human import Human
from RandomBot import Random_Bot,Greedy_Bot
from Dots_and_Box import DotsAndBox
from DeepLearning import LSTM_BOT,ResnetBOT, Conv2Plus1D_BOT
from Alpha.MCTS import MCTSPlayer
from arg import m, n, args_CNN, args_Res, args_LSTM, args_ConvLSTM, args_Conv2Plus1D

args_CNN['train'] = False
args_Conv2Plus1D['train'] = False
args_ConvLSTM['train'] = False
args_LSTM['train'] = False
args_LSTM['load_model_name'] = 'LSTM_model_4x4_15.h5'
args_Res['train'] = False

def main():
    size_m = 4
    size_n = 4
    game = DotsAndBox(size_m,size_n)
    p1 = Human(game=game)
    p2 = Random_Bot(game=game)
    p3 = ResnetBOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_Res)
    
    # args_LSTM['load_model_name'] = 'LSTM_model_4x4_5.h5'
    p4 = LSTM_BOT(input_size_m=size_m,input_size_n=size_n,game=game,args=args_LSTM)
    p5 = Greedy_Bot(game=game)
    p6 = MCTSPlayer(num_simulations=100, exploration_weight=1.5, max_depth=5)
    p6.game_state = game
    p7 = Conv2Plus1D_BOT(input_size_m=size_m,input_size_n=size_n,game=game, args=args_Conv2Plus1D)
    
    
    
    game.play(player1=p2,player2=p3)
if __name__ == "__main__":
    main()