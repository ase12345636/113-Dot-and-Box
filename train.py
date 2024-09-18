from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import BOT

m=3
n=3
game = DaB(m,n)
bot=BOT(input_size_m=m,input_size_n=n,game=game)

batch_size = ((m*2)-1) * ((n*2)-1) * 2
args={
    'num_of_generate_data_for_train': 500,
    'epochs': 30,
    'batch_size': batch_size,
    'verbose': True,
}

for i in range(5):
    bot.self_play_train(args)

#game=OthelloGame(BOARD_SIZE)
#game.play(black=bot, white=Human())

