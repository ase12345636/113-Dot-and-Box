from Human import Human
from RandomBot import Random_Bot
from Dots_and_Box import DotsAndBox
from DeepLearning import BOT

def main():
    size_m = 5
    size_n = 5
    game = DotsAndBox(size_m,size_n)
    p1 = Human(game=game)
    p2 = Random_Bot(game=game)
    p3 = BOT(input_size_m=size_m,input_size_n=size_n,game=game)
    game.play(player1=p3,player2=p1)
if __name__ == "__main__":
    main()  