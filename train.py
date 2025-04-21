from Dots_and_Box import DotsAndBox as DaB
from DeepLearning6x6 import CNNBOT, ResnetBOT, LSTM_BOT, ConvLSTM_BOT, Conv2Plus1D_BOT
from arg import m, n, args_CNN, args_Res, args_LSTM, args_ConvLSTM, args_Conv2Plus1D
from RandomBot import *
from Alpha.AlphaBeta import AlphaBetaPlayer

game = DaB(m, n)
# bot_CNN = CNNBOT(input_size_m=m, input_size_n=n, game=game, args=args_CNN)
bot_Res = ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args_Res)
# bot_LSTM = LSTM_BOT(input_size_m=m, input_size_n=n, game=game, args=args_LSTM)
# bot_ConvLSTM = ConvLSTM_BOT(
#     input_size_m=m, input_size_n=n, game=game, args=args_ConvLSTM)
# bot_Conv2Plus1D = Conv2Plus1D_BOT(
#    input_size_m=m, input_size_n=n, game=game, args=args_Conv2Plus1D)


args_Res['train'] = True    #True:開greedy, False:關
args_Oppo = {
    'verbose': True,
    'type': 0,
    'train': False,  # 對手關閉random
    'load_model_name': None
}

# oppo_bot = ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args_Oppo)
# oppo_bot = Greedy_Bot(game)


# bot_CNN.self_play_train()
print(args_Res)
bot_Res.self_play_train(oppo=None)
# bot_Res.train_from_json()
# bot_LSTM.self_play_train()
# bot_ConvLSTM.self_play_train()
# bot_Conv2Plus1D.self_play_train()