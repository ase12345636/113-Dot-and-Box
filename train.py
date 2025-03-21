from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import CNNBOT, ResnetBOT, LSTM_BOT, ConvLSTM_BOT, Conv2Plus1D_BOT
from arg import m, n, args_CNN, args_Res, args_LSTM, args_ConvLSTM, args_Conv2Plus1D

args_LSTM['load_model_name'] = None

game = DaB(m, n)
# bot_CNN = CNNBOT(input_size_m=m, input_size_n=n, game=game, args=args_CNN)
# bot_Res = ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args_Res)
# bot_LSTM = LSTM_BOT(input_size_m=m, input_size_n=n, game=game, args=args_LSTM)
bot_ConvLSTM = ConvLSTM_BOT(
    input_size_m=m, input_size_n=n, game=game, args=args_ConvLSTM)
# bot_Conv2Plus1D = Conv2Plus1D_BOT(
#     input_size_m=m, input_size_n=n, game=game, args=args_Conv2Plus1D)

# bot_CNN.self_play_train()
# bot_Res.self_play_train()
# bot_LSTM.self_play_train()
bot_ConvLSTM.self_play_train()
# bot_Conv2Plus1D.self_play_train()

# game=OthelloGame(BOARD_SIZE)
# game.play(black=bot, white=Human())
