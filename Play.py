from Human import Human
from RandomBot import Random_Bot,Greedy_Bot
from Dots_and_Box import DotsAndBox
from DeepLearning import LSTM_BOT,ResnetBOT

def main():
    size_m = 5
    size_n = 5
    game = DotsAndBox(size_m,size_n)
    p1 = Human(game=game)
    p2 = Random_Bot(game=game)
    p3 = ResnetBOT(input_size_m=size_m,input_size_n=size_n,game=game)
    p4 = LSTM_BOT(input_size_m=size_m,input_size_n=size_n,game=game)
    p5 = Greedy_Bot(game=game)
    game.play(player1=p4,player2=p5)
if __name__ == "__main__":
    main()