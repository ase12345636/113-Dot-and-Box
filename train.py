from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import BOT

game = DaB(4,4)
bot=BOT(input_size_m=4,input_size_n=4,game=game)

args={
    'num_of_generate_data_for_train': 1,
    'epochs': 1,
    'batch_size': 16,
    'verbose': True,
}

bot.self_play_train(args)

#game=OthelloGame(BOARD_SIZE)
#game.play(black=bot, white=Human())

