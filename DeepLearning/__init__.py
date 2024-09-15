import numpy as np
from Dots_and_Box import DotsAndBox as DaB

from DeepLearning.DaB_Model import DaB_ResNet
class BOT():
    def __init__(self, input_size_m,input_size_n,game: DaB, *args, **kargs):
        self.input_size_m = input_size_m * 2 - 1
        self.input_size_n = input_size_n * 2 - 1
        self.game = game
        self.model = DaB_ResNet( input_shape=(self.input_size_m, self.input_size_n) )
        try:
            self.model.load_weights()
            print('model loaded')
        except:
            print('no model exist')
            pass
        
        self.collect_gaming_data=False
        self.history=[]
    
    def get_move(self):
        board = np.squeeze(np.expand_dims(self.game.board, axis=0)).astype('float32')
        print(f"Board shape before prediction: {board.shape}")  # 打印數據形狀
        predict = self.model.predict(np.expand_dims(board, axis=0))[0]
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
            self.history.append([np.array(self.game.board.copy()), tmp, self.game.current_player])
        
        position = (position // self.input_size_n, position % self.input_size_n)
        return position
    
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

            self.history = []
            self.game.play(self, self)
            history = []
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b, p in sym:
                    history.append([b, p, player])
            self.history.clear()
            game_result = self.game.isGameOver()
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
        
                
        data = []
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('self playing', i+1)
            current_data = gen_data()
            data+=current_data
        self.collect_gaming_data=False
        print(f"length of data: {len(data)}")
        history = self.model.fit(data, batch_size = args['batch_size'], epochs = args['epochs'])
        self.model.save_weights()
        self.model.plot_learning_curve(history)

