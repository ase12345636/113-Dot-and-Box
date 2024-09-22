from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import ResnetBOT,LSTM_BOT

m=5
n=5
game = DaB(m,n)
bot_Res = ResnetBOT(input_size_m=m,input_size_n=n,game=game)
bot_LSTM = LSTM_BOT(input_size_m=m,input_size_n=n,game=game)

batch_size = 32
args={
    'num_of_generate_data_for_train': 512,
    'epochs': 16,
    'batch_size': batch_size,
    'verbose': True,
}

bot_LSTM.self_play_train(args)

#game=OthelloGame(BOARD_SIZE)
#game.play(black=bot, white=Human())

