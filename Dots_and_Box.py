# -1 or 1 : 先手(1)or後手(-1)走的邊
# 5 : 頂點
# 8 : 未被佔領的格子
# 7 or 9 : 被先手or後手佔領的格子，7->8+(-1)，9->8+(1)
# p1 -> -1(blue)，p2 -> 1(red)
def ANSI_string(s="", color=None, background=None, bold=False):
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }

    background_colors = {
        'black': '\033[40m',
        'red': '\033[41m',
        'green': '\033[42m',
        'yellow': '\033[43m',
        'blue': '\033[44m',
        'magenta': '\033[45m',
        'cyan': '\033[46m',
        'white': '\033[47m',
        'reset': '\033[0m',
        'gray': '\033[100m',  # 新增的灰色背景
        'light_gray': '\033[47m'  # 新增的淺灰色背景
    }

    styles = {
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    color_code = colors[color] if color in colors else ''
    background_code = background_colors[background] if background in colors else ''
    bold_code = styles['bold'] if bold else ''

    return f"{color_code}{background_code}{bold_code}{s}{styles['reset']}"


class DotsAndBox():
    def initialize_board(self, m, n):
        # 初始化一個(2*m-1) * (2*n-1)的二維陣列
        board = [[0 for _ in range(2*n-1)] for _ in range(2*m-1)]
        # 填入頂點(5)和合法邊(0)，以及未被佔領的格子(8)
        for i in range(2*m-1):
            for j in range(2*n-1):
                if i % 2 == 0 and j % 2 == 0:
                    # 填入頂點 5
                    board[i][j] = 5
                elif i % 2 == 1 and j % 2 == 1:
                    # 填入格子 8
                    board[i][j] = 8
                else:
                    # 填入合法邊 0
                    board[i][j] = 0
        return board

    def __init__(self, m, n, collect_gaming_data = True):
        self.input_m = m
        self.input_n = n
        self.board_rows_nums = 2*m-1
        self.board_cols_nums = 2*n-1
        self.board = self.initialize_board(m=m, n=n)
        self.current_player = -1
        self.p1_scores = 0
        self.p2_scores = 0

        self.collect_gaming_data = collect_gaming_data
        self.history = []

    def getValidMoves(self):
        ValidMoves = []
        for i in range(self.board_rows_nums):
            for j in range(self.board_cols_nums):
                if self.board[i][j] == 0:
                    ValidMoves.append((i, j))
        return ValidMoves

    def isValid(self, r, c):
        if r >= self.board_rows_nums or c >= self.board_cols_nums:
            return False
        if self.board[r][c] != 0:
            return False
        return True

    def checkBox(self, board):
        box_filled = False
        for i in range(self.input_m - 1):
            for j in range(self.input_n - 1):
                box_i = 2*i + 1
                box_j = 2*j + 1
                # 檢查該方格的四條邊是否都不為 0
                if (board[box_i][box_j] == 8 and
                    board[box_i-1][box_j] != 0 and
                    board[box_i+1][box_j] != 0 and
                    board[box_i][box_j-1] != 0 and
                        board[box_i][box_j+1] != 0):

                    # 更新該方格的狀態
                    board[box_i][box_j] += self.current_player
                    if board[box_i][box_j] == 7:
                        self.p1_scores += 1
                    elif board[box_i][box_j] == 9:
                        self.p2_scores += 1
                    box_filled = True
        return box_filled

    def make_move(self, row, col):
        if not self.isValid(r=row, c=col):
            print("invalid move!!!")
            return

        self.board[row][col] = self.current_player

        if not self.checkBox(board=self.board):  # 沒有完成方形就換人
            self.current_player *= -1  # 沒有完成方格，換下一人

    def isGameOver(self):
        for i in range(self.input_m - 1):
            for j in range(self.input_n - 1):
                if self.board[2*i+1][2*j+1] == 8:
                    return False
        return True

    def GetWinner(self):
        if self.isGameOver():
            if self.p1_scores > self.p2_scores:
                return -1
            elif self.p1_scores < self.p2_scores:
                return 1
            else:
                return 0

    def play(self, player1, player2, verbose=True, train = False):
        if verbose:
            self.print_board()

        while not self.isGameOver():
            # print(f"Valid moves: {self.getValidMoves()}")
            print(f"Current player: {self.current_player}")

            if self.current_player == -1:
                move_data = player1.get_move()
            else:
                move_data = player2.get_move()

            if move_data[0]:    #valid_postion
                row, col = move_data[0]
                if train:
                    self.history.append(move_data[1])
                self.make_move(row, col)
                # if verbose:
                #     self.print_board()
        winner = self.GetWinner()
        if verbose:
            self.print_board()

        if (winner == 0):
            print("Tie!!!")
            return 0
        else:
            print(f"Player {winner} won!!!")
            return winner

    def NewGame(self):
        self.current_player = -1
        self.board = self.initialize_board(self.input_m, self.input_n)
        self.p1_scores = 0
        self.p2_scores = 0
        self.history.clear()

    def print_board(self):
        print(f"Player -1: {self.p1_scores}")
        print(f"Player 1: {self.p2_scores}")
        for i in range(self.board_rows_nums):
            for j in range(self.board_cols_nums):
                value = self.board[i][j]
                if value == 0:
                    print(' ', end='')
                elif value in [-1, 1]:  # 處理先手和後手邊
                    color = 'blue' if value == -1 else 'red'
                    if i % 2 == 0:  # 偶數列 -> 水平線
                        print(ANSI_string(s='-', color=color), end='')
                    else:  # 奇數列 -> 垂直線
                        print(ANSI_string(s='|', color=color), end='')
                elif value == 5:  # 頂點
                    print('o', end='')
                elif value in [7, 9]:  # 處理先手和後手佔領
                    background = 'blue' if value == 7 else 'red'
                    print(ANSI_string(s=str(value), background=background), end='')
                elif value == 8:
                    print(' ', end='')
            print()
        print("="*(self.board_cols_nums))
