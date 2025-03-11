import numpy as np
from Dots_and_Box import DotsAndBox
from RandomBot import Greedy_Bot
from DeepLearning.DaB_Model import DaB_CNN, DaB_ResNet, DaB_LSTM
from RandomBot import GreedAlg
from einops import rearrange


class BaseBot():
    # Initiallize
    def __init__(self, input_size_m, input_size_n, game, args):
        self.input_size_m = input_size_m * 2 - 1
        self.input_size_n = input_size_n * 2 - 1
        self.total_move = input_size_m * (input_size_n-1) + \
            (input_size_m-1) * input_size_n
        self.game = game
        self.args = args

        self.collect_gaming_data = False
        self.history = []

    # Get move predicted by model
    def get_move(self):
        board = self.preprocess_board(self.game.board)

        # Type 0
        if (self.args['type'] == 0):

            # Predict move
            predict = self.model.predict(
                np.expand_dims(board, axis=0).astype(float))

        # Type 1
        elif (self.args['type'] == 1):

            # Get history of board
            board_history = np.array(None)
            for i in range(len(self.history)):
                if (board_history == np.array(None)).all():
                    board_history = rearrange(
                        np.array(self.history[i][0]), 'm n -> m n 1')

                else:
                    board_history = np.append(board_history,
                                              rearrange(
                                                  np.array(self.history[i][0]), 'm n -> m n 1'), axis=2)

            # Append current board
            if (board_history == np.array(None)).all():
                board_history = rearrange(np.array(board), 'm n -> m n 1')

            else:
                board_history = np.append(board_history,
                                          rearrange(np.array(board), 'm n -> m n 1'), axis=2)

            # Append padding board
            for i in range(self.total_move-board_history.shape[2]):
                board_history = np.append(board_history,
                                          np.full((self.input_size_m, self.input_size_n, 1), 255), axis=2)

            # Predict move
            predict = self.model.predict(np.expand_dims(
                board_history, axis=0).astype(float))

        # Type 2
        elif (self.args['type'] == 2):

            # Get history of board
            board_history = np.array(None)
            for i in range(len(self.history)):
                if (board_history == np.array(None)).all():
                    board_history = rearrange(
                        np.array(self.history[i][0]), 'm n -> 1 (m n)')

                else:
                    board_history = np.append(board_history,
                                              rearrange(np.array(self.history[i][0]), 'm n -> 1 (m n)'), axis=0)

            # Append current board
            if (board_history == np.array(None)).all():
                board_history = rearrange(np.array(board), 'm n -> 1 (m n)')

            else:
                board_history = np.append(board_history,
                                          rearrange(np.array(board), 'm n -> 1 (m n)'), axis=0)

            # Append padding board
            for i in range(self.total_move-board_history.shape[0]):
                board_history = np.append(board_history,
                                          np.full((1, self.input_size_m * self.input_size_n), 255), axis=0)

            # Predict move
            predict = self.model.predict(np.expand_dims(
                board_history, axis=0).astype(float))

        # Detect which move is valid
        valid_positions = self.game.getValidMoves()
        valids = np.zeros(
            (self.input_size_m * self.input_size_n,), dtype='int')
        for pos in valid_positions:
            idx = pos[0] * self.input_size_n + pos[1]
            valids[idx] = 1

        # Filtered invalid move and avoid invalid loop
        predict = (predict+1e-30) * valids

        # Get final prediction
        position = np.argmax(predict)

        # Append current board to history
        if self.collect_gaming_data:
            tmp = np.zeros_like(predict)
            tmp[position] = 1.0
            self.history.append([board, tmp, self.game.current_player])

        position = (position // self.input_size_n,
                    position % self.input_size_n)
        return position

    # Comment
    def preprocess_board(self, board):
        # Convert the board to a binary representation
        # processed_board = np.zeros(
        #     (self.input_size_m, self.input_size_n), dtype='float32')
        # for i in range(self.input_size_m):
        #     for j in range(self.input_size_n):
        #         if board[i][j] in [5, 7, 9]:  # Vertex, player 1 box, player 2 box
        #             processed_board[i][j] = 1
        #         elif board[i][j] in [-1, 1]:  # Edges
        #             processed_board[i][j] = 0.5
        # return processed_board

        return board

    # Training model based on history
    def self_play_train(self):

        # Allow collecting history
        self.collect_gaming_data = True

        # Generate history data
        def gen_data(type: 0):

            # Data augmentation by getting symmetries
            def getSymmetries(board, pi):
                pi_board = np.reshape(
                    pi, (self.input_size_m, self.input_size_n))
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

            # Split data and get 8 symmetries of history
            def splitSymmetries(sym):
                split_sym = []
                for i in range(8):
                    temp = []
                    j = i
                    while (j < len(sym)):
                        temp.append(sym[j])
                        j += 8

                    split_sym.append(temp)

                return split_sym

            # Initiallize history
            self.history = []

            # Get history data
            self.game.NewGame()
            self.game.play(self, self)

            # Process history data
            history = []

            # Data augmentation
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b, p in sym:
                    history.append([b, p, player])
            self.history.clear()
            game_result = self.game.GetWinner()

            # Type 0
            if (type == 0):
                return [(x[0], x[1]) for x in history if (game_result == 0 or x[2] == game_result)]

            # Type 1
            elif (type == 1):

                # Split data
                history_split_sym = splitSymmetries(history)

                # Process each split
                history_image = []
                for split in history_split_sym:

                    # Get winner's move
                    for i in range(len(split)):
                        if (game_result == 0 or split[i][2] == game_result):
                            board_temp = np.array(None)
                            probs_temp = split[i][1]

                            # Get history and current move
                            for j in range(i+1):
                                if (board_temp == np.array(None)).all():
                                    board_temp = rearrange(
                                        split[j][0], 'm n -> m n 1')

                                else:
                                    board_temp = np.append(board_temp,
                                                           rearrange(
                                                               split[j][0], 'm n -> m n 1'),
                                                           axis=2)

                            # Append padding board
                            for j in range(i+1, self.total_move):
                                board_padding = np.full(
                                    (self.input_size_m, self.input_size_n, 1), 255)
                                board_temp = np.append(board_temp,
                                                       board_padding,
                                                       axis=2)

                            history_image.append([board_temp, probs_temp])

                return [(x[0], x[1]) for x in history_image]

            # Type 2
            elif (type == 2):

                # Split data
                history_split_sym = splitSymmetries(history)

                # Process each split
                history_seq = []
                for split in history_split_sym:

                    # Get history and current move
                    for i in range(len(split)):
                        if (game_result == 0 or split[i][2] == game_result):
                            board_temp = np.array(None)
                            probs_temp = split[i][1]

                            # Get history and current move
                            for j in range(i+1):
                                if (board_temp == np.array(None)).all():
                                    board_temp = rearrange(
                                        split[j][0], 'm n -> 1 (m n)')

                                else:
                                    board_temp = np.append(board_temp,
                                                           rearrange(
                                                               split[j][0], 'm n -> 1 (m n)'),
                                                           axis=0)

                            # Append padding board
                            for j in range(i+1, self.total_move):
                                board_padding = np.full(
                                    (1, self.input_size_m*self.input_size_n), 255)
                                board_temp = np.append(board_temp,
                                                       board_padding,
                                                       axis=0)

                            history_seq.append([board_temp, probs_temp])

                return [(x[0], x[1]) for x in history_seq]

        # Generate data
        data = []
        for i in range(self.args['num_of_generate_data_for_train']):
            if self.args['verbose']:
                print(f'Self playing {i + 1}')
            current_data = gen_data(self.args['type'])
            data += current_data
        self.collect_gaming_data = False

        # Training model
        print(f"Length of data: {len(data)}")
        history = self.model.fit(
            data, batch_size=self.args['batch_size'], epochs=self.args['epochs'])
        self.model.save_weights()
        self.model.plot_learning_curve(history)


class CNNBOT(BaseBot):
    def __init__(self, input_size_m, input_size_n, game, args):
        super().__init__(input_size_m, input_size_n, game, args)

        self.model = DaB_CNN(input_shape=(
            self.input_size_m, self.input_size_n, self.total_move), args=args)
        try:
            self.model.load_weights()
            print(f'{self.model.model_name} loaded')
        except:
            print('No model exists')


class ResnetBOT(BaseBot):
    def __init__(self, input_size_m, input_size_n, game, args):
        super().__init__(input_size_m, input_size_n, game, args)

        self.model = DaB_ResNet(input_shape=(
            self.input_size_m, self.input_size_n, self.total_move), args=args)
        try:
            self.model.load_weights()
            print(f'{self.model.model_name} loaded')
        except:
            print('No model exists')


class LSTM_BOT(BaseBot):
    def __init__(self, input_size_m, input_size_n, game, args):
        super().__init__(input_size_m, input_size_n, game, args)

        self.model = DaB_LSTM(input_shape=(
            self.input_size_m, self.input_size_n, self.total_move), args=args)
        try:
            self.model.load_weights()
            print(f'{self.model.model_name} loaded')
        except:
            print('No model exists')

    # def self_play_train(self, args):
    #     self.collect_gaming_data = True

    #     def gen_data():
    #         def getSymmetries(board, pi):
    #             pi_board = np.reshape(
    #                 pi, (self.input_size_m, self.input_size_n))
    #             return [(board, list(pi_board.ravel()))]

    #         self.history = []
    #         self.trainingBot = Greedy_Bot(game=self.game)

    #         self.game.NewGame()
    #         while not self.game.isGameOver():
    #             print(f"Valid moves: {self.game.getValidMoves()}")
    #             print(f"Current player: {self.game.current_player}")
    #             if self.game.current_player == -1:
    #                 move = self.get_move()
    #             else:
    #                 move = self.trainingBot.get_move()
    #                 board = self.preprocess_board(self.game.board)
    #                 predict = self.model.predict(np.expand_dims(board, axis=0))
    #                 valid_positions = self.game.getValidMoves()
    #                 valids = np.zeros(
    #                     (self.input_size_m * self.input_size_n,), dtype='int')
    #                 for pos in valid_positions:
    #                     idx = pos[0] * self.input_size_n + pos[1]
    #                     valids[idx] = 1
    #                 predict *= valids
    #                 position = np.argmax(predict)

    #                 if self.collect_gaming_data:
    #                     tmp = np.zeros_like(predict)
    #                     tmp[position] = 1.0
    #                     self.history.append(
    #                         [board, tmp, self.game.current_player])

    #             if move:
    #                 row, col = move
    #                 self.game.make_move(row, col)

    #         self.game.print_board()

    #         self.game.NewGame()
    #         while not self.game.isGameOver():
    #             print(f"Valid moves: {self.game.getValidMoves()}")
    #             print(f"Current player: {self.game.current_player}")
    #             if self.game.current_player == 1:
    #                 move = self.get_move()
    #             else:
    #                 move = self.trainingBot.get_move()
    #                 board = self.preprocess_board(self.game.board)
    #                 predict = self.model.predict(np.expand_dims(board, axis=0))
    #                 valid_positions = self.game.getValidMoves()
    #                 valids = np.zeros(
    #                     (self.input_size_m * self.input_size_n,), dtype='int')
    #                 for pos in valid_positions:
    #                     idx = pos[0] * self.input_size_n + pos[1]
    #                     valids[idx] = 1
    #                 predict *= valids
    #                 position = np.argmax(predict)

    #                 if self.collect_gaming_data:
    #                     tmp = np.zeros_like(predict)
    #                     tmp[position] = 1.0
    #                     self.history.append(
    #                         [board, tmp, self.game.current_player])

    #             if move:
    #                 row, col = move
    #                 self.game.make_move(row, col)

    #         self.game.print_board()

    #         history = []
    #         for step, (board, probs, player) in enumerate(self.history):
    #             sym = getSymmetries(board, probs)
    #             for b, p in sym:
    #                 history.append([b, p, player])
    #         self.history.clear()
    #         game_result = self.game.GetWinner()
    #         return [(x[0], x[1]) for x in history if (game_result == 0 or x[2] == game_result)]
    #     data = []
    #     for i in range(args['num_of_generate_data_for_train']):
    #         if args['verbose']:
    #             print(f'Self playing {i + 1}')
    #         current_data = gen_data()
    #         data += current_data
    #     self.collect_gaming_data = False
    #     print(f"Length of data: {len(data)}")
    #     history = self.model.fit(
    #         data, batch_size=args['batch_size'], epochs=args['epochs'])
    #     self.model.save_weights()
    #     self.model.plot_learning_curve(history)
