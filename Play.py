from Human import Human
from RandomBot import Random_Bot
from Dots_and_Box import DotsAndBox

def main():
    size_m = 6
    size_n = 6
    game = DotsAndBox(size_m,size_n)
    p1 = Random_Bot(game=game)
    p2 = Random_Bot(game=game)
    game.play(player1=p1,player2=p2)
if __name__ == "__main__":
    main()  