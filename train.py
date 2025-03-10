from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import CNNBOT, ResnetBOT, LSTM_BOT
from arg import m, n, args_CNN, args_Res, args_LSTM


game = DaB(m, n)
# bot_CNN = CNNBOT(input_size_m=m, input_size_n=n, game=game, args=args_CNN)
# bot_Res = ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args_Res)
bot_LSTM = LSTM_BOT(input_size_m=m, input_size_n=n, game=game, args=args_LSTM)

for i in range(1):
    bot_LSTM.self_play_train()

# game=OthelloGame(BOARD_SIZE)
# game.play(black=bot, white=Human())
