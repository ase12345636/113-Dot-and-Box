from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import CNNBOT, ResnetBOT, LSTM_BOT

m = 4
n = 4

batch_size = (m * (n - 1)) * 2
args = {
    'num_of_generate_data_for_train': 1,
    'epochs': 5,
    'batch_size': batch_size,
    'verbose': True,
    'seq': True
}

game = DaB(m, n)
bot_CNN = CNNBOT(input_size_m=m, input_size_n=n, game=game, args=args)
bot_Res = ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args)
bot_LSTM = LSTM_BOT(input_size_m=m, input_size_n=n, game=game, args=args)

for i in range(1):
    bot_CNN.self_play_train()

# game=OthelloGame(BOARD_SIZE)
# game.play(black=bot, white=Human())
