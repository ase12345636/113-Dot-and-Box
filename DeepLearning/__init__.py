import numpy as np
from Dots_and_Box import DotsAndBox
from RandomBot import Greedy_Bot
from DeepLearning.DaB_Model import DaB_ResNet,DaB_LSTM
from RandomBot import GreedAlg

class ResnetBOT():
    def __init__(self, input_size_m, input_size_n, game):
        self.input_size_m = input_size_m * 2 - 1
        self.input_size_n = input_size_n * 2 - 1
        self.game = game
        self.model = DaB_ResNet(input_shape=(self.input_size_m, self.input_size_n))
        try:
            self.model.load_weights()
            print(f'{self.model.model_name} loaded')
        except:
            print('No model exists')
        
        self.collect_gaming_data = False
        self.history = []
    
    def get_move(self):
        board = self.preprocess_board(self.game.board)
        predict = self.model.predict(np.expand_dims(board, axis=0))
        valid_positions = self.game.getValidMoves()
        valids = np.zeros((self.input_size_m * self.input_size_n,), dtype='int')
        for pos in valid_positions:
            idx = pos[0] * self.input_size_n + pos[1]
            valids[idx] = 1
        predict *= valids
        position = np.argmax(predict)
        
        if self.collect_gaming_data:
            tmp = np.zeros_like(predict)
            tmp[position] = 1.0
            self.history.append([board, tmp, self.game.current_player])
        
        position = (position // self.input_size_n, position % self.input_size_n)
        return position
    
    def preprocess_board(self, board):
        # Convert the board to a binary representation
        processed_board = np.zeros((self.input_size_m, self.input_size_n), dtype='float32')
        for i in range(self.input_size_m):
            for j in range(self.input_size_n):
                if board[i][j] in [5, 7, 9]:  # Vertex, player 1 box, player 2 box
                    processed_board[i][j] = 1
                elif board[i][j] in [-1, 1]:  # Edges
                    processed_board[i][j] = 0.5
        return processed_board

    def self_play_train(self, args):
        self.collect_gaming_data = True
        def gen_data():
            def getSymmetries(board, pi):
                pi_board = np.reshape(pi, (self.input_size_m, self.input_size_n))
                symmetries = []
                for i in range(4):
                    for flip in [True, False]:
                        newB = np.rot90(board, i)
                        newPi = np.rot90(pi_board, i)
                        if flip:
                            newB = np.fliplr(newB)
                            newPi = np.fliplr(newPi)
                        symmetries.append((newB, list(newPi.ravel())))
                return symmetries
            # def getSymmetries(board, pi):
            #     pi_board = np.reshape(pi, (self.input_size_m, self.input_size_n))
            #     return [(board, list(pi_board.ravel()))]

            self.history = []
            self.game.NewGame()
            self.game.play(self, self)
            history = []
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b, p in sym:
                    history.append([b, p, player])
            self.history.clear()
            game_result = self.game.GetWinner()
            return [(x[0], x[1]) for x in history if (game_result == 0 or x[2] == game_result)]
        
        data = []
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print(f'Self playing {i + 1}')
            current_data = gen_data()
            data += current_data
        self.collect_gaming_data = False
        print(f"Length of data: {len(data)}")
        history = self.model.fit(data, batch_size=args['batch_size'], epochs=args['epochs'])
        self.model.save_weights()
        self.model.plot_learning_curve(history)


class LSTM_BOT():
    def __init__(self, input_size_m, input_size_n, game):
        self.input_size_m = input_size_m * 2 - 1
        self.input_size_n = input_size_n * 2 - 1
        self.game = game
        self.model = DaB_LSTM(input_shape=(self.input_size_m, self.input_size_n))
        try:
            self.model.load_weights()
            print(f'{self.model.model_name} loaded')
        except:
            print('No model exists')
        
        self.collect_gaming_data = False
        self.history = []
    
    def get_move(self):
        # pos = GreedAlg(board=self.game.board,m=self.game.input_m,n=self.game.input_n,ValidMoves=self.game.getValidMoves())
        # if pos:
        #     r,c = pos
        #     return r,c
        # else:
        #     pass
        board = self.preprocess_board(self.game.board)
        predict = self.model.predict(np.expand_dims(board, axis=0))
        valid_positions = self.game.getValidMoves()
        valids = np.zeros((self.input_size_m * self.input_size_n,), dtype='int')
        for pos in valid_positions:
            idx = pos[0] * self.input_size_n + pos[1]
            valids[idx] = 1
        predict *= valids
        position = np.argmax(predict)
        
        if self.collect_gaming_data:
            tmp = np.zeros_like(predict)
            tmp[position] = 1.0
            self.history.append([board, tmp, self.game.current_player])
        
        position = (position // self.input_size_n, position % self.input_size_n)
        return position
    
    def preprocess_board(self, board):
        # Convert the board to a binary representation
        processed_board = np.zeros((self.input_size_m, self.input_size_n), dtype='float32')
        for i in range(self.input_size_m):
            for j in range(self.input_size_n):
                if board[i][j] in [5, 7, 9]:  # Vertex, player 1 box, player 2 box
                    processed_board[i][j] = 1
                elif board[i][j] in [-1, 1]:  # Edges
                    processed_board[i][j] = 0.5
        return processed_board

    def self_play_train(self, args):
        self.collect_gaming_data = True
        def gen_data():
            def getSymmetries(board, pi):
                pi_board = np.reshape(pi, (self.input_size_m, self.input_size_n))
                return [(board, list(pi_board.ravel()))]

            self.history = []
            self.trainingBot = Greedy_Bot(game=self.game)
            
            while not self.game.isGameOver():
                print(f"Valid moves: {self.game.getValidMoves()}")
                print(f"Current player: {self.game.current_player}")
                if self.game.current_player == -1:
                    move = self.get_move()
                else:
                    move = self.trainingBot.get_move()
                    board = self.preprocess_board(self.game.board)
                    predict = self.model.predict(np.expand_dims(board, axis=0))
                    valid_positions = self.game.getValidMoves()
                    valids = np.zeros((self.input_size_m * self.input_size_n,), dtype='int')
                    for pos in valid_positions:
                        idx = pos[0] * self.input_size_n + pos[1]
                        valids[idx] = 1
                    predict *= valids
                    position = np.argmax(predict)
                    
                    if self.collect_gaming_data:
                        tmp = np.zeros_like(predict)
                        tmp[position] = 1.0
                        self.history.append([board, tmp, self.game.current_player])

                if move:
                    row, col = move
                    self.game.make_move(row, col)
                    
                                   
            self.game.print_board()
            
            history = []
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b, p in sym:
                    history.append([b, p, player])
            self.history.clear()
            game_result = self.game.GetWinner()
            return [(x[0], x[1]) for x in history if (game_result == 0 or x[2] == game_result)]
        
        data = []
        for i in range(args['num_of_generate_data_for_train']):
            self.game.NewGame()
            if i%2 == 0: 
                self.game.current_player = -1
            else:
                self.game.current_player = 1
            if args['verbose']:
                print(f'Self playing {i + 1}')
            current_data = gen_data()
            data += current_data
        self.collect_gaming_data = False
        print(f"Length of data: {len(data)}")
        history = self.model.fit(data, batch_size=args['batch_size'], epochs=args['epochs'])
        self.model.save_weights()
        self.model.plot_learning_curve(history)
