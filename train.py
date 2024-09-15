from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import BOT

game = DaB(5,5)
bot=BOT(input_size_m=5,input_size_n=5,game=game)
args={
    'num_of_generate_data_for_train': 10,
    'epochs': 2,
    'batch_size': 16,
    'verbose': True,
    'save_data_to_txt': False,
    'adding_dataset_from_txt': False
}

bot.self_play_train(args)

#game=OthelloGame(BOARD_SIZE)
#game.play(black=bot, white=Human())

