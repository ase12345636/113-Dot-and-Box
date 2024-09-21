from PyQt5.QtWidgets import QMainWindow, QApplication, QComboBox
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QColor, QPen ,QFont
from PyQt5.QtCore import Qt, QTimer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from Dots_and_Box import DotsAndBox as DaB
from Human import Human
from RandomBot import Random_Bot
from DeepLearning import LSTM_BOT


class GameWindow(QMainWindow):
    def __init__(self, game: DaB):
        super(GameWindow, self).__init__()
        self.setGeometry(500, 200, 800, 600)
        self.setWindowTitle('Dots and Boxes')
        self.game = game
        self.gap = 50
        self.Dots_radius = 20
        self.BoardStart_pos = (50, 50)
        self.mouse_events_enabled = False
        
        # 設置字體樣式
        self.font = QFont("Arial", 12, QFont.Bold)  # 使用 Arial 字型，大小 14，粗體
        
        self.init_ui()

    def init_ui(self):
        self.P1_score_label = QtWidgets.QLabel(self)
        self.P1_score_label.setText(f'Player1 scores: {self.game.p1_scores}')
        self.P1_score_label.setGeometry(600, 100, 175, 25)
        self.P1_score_label.setFont(self.font)  # 設置標籤字型

        self.P2_score_label = QtWidgets.QLabel(self)
        self.P2_score_label.setText(f'Player2 scores: {self.game.p2_scores}')
        self.P2_score_label.setGeometry(600, 150, 175, 25)
        self.P2_score_label.setFont(self.font)  # 設置標籤字型

        # 下拉選單
        self.mode_combo_box = QComboBox(self)
        self.mode_combo_box.addItem("人類 VS 人類")
        self.mode_combo_box.addItem("人類 VS 隨機")
        self.mode_combo_box.addItem("人類 VS 模型")
        self.mode_combo_box.setGeometry(600, 200, 175, 25)
        self.mode_combo_box.setFont(self.font)  # 設置下拉選單字型

        self.StartButton = QtWidgets.QPushButton(self)
        self.StartButton.setText("Start!!!")
        self.StartButton.setGeometry(600, 250, 175, 25)
        self.StartButton.setFont(self.font)  # 設置按鈕字型
        self.StartButton.clicked.connect(self.OnClickStartButton)
    

    def OnClickStartButton(self):
        self.game.NewGame()
        self.P1_score_label.setText(f'Player1 scores: {self.game.p1_scores}')
        self.P2_score_label.setText(f'Player2 scores: {self.game.p2_scores}')
        self.update()

        selected_mode = self.mode_combo_box.currentText()
        self.mode = 0
        if selected_mode == "人類 VS 人類":
            self.mode = 1
        elif selected_mode == "人類 VS 隨機":
            self.mode = 2
        elif selected_mode == "人類 VS 模型":
            self.mode = 3

        if self.mode == 2:
            self.p2 = Random_Bot(self.game)
        elif self.mode == 3:
            self.p2 = LSTM_BOT(self.game.input_m, self.game.input_n, self.game)

        # Create a timer to handle the game progression
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(100)  # Check the game every 100 milliseconds

    def game_loop(self):
        if self.game.isGameOver():
            
            self.timer.stop()  # Stop the timer when the game is over
            return

        if self.mode != 1:
            if self.game.current_player == -1:  # Human player's turn
                self.mouse_events_enabled = True  # Enable mouse events
            else:  # AI player's turn
                r, c = self.p2.get_move()
                self.game.make_move(r, c)
                self.update()  # Repaint the board after AI move
                self.mouse_events_enabled = False
        else:
            self.mouse_events_enabled = True
                
    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_board(painter)
    
    # 用於繪製遊戲棋盤的方法
    def draw_board(self, painter):
        self.P1_score_label.setText(f'Player1 scores: {self.game.p1_scores}')
        self.P2_score_label.setText(f'Player2 scores: {self.game.p2_scores}')
        Dots_pen = QPen(QColor('#000000'), 3)
        Blue_solid_pen = QPen(QColor('#0000E3'), 5)  # 藍色實線，寬度為5
        Red_solid_pen = QPen(QColor('#FF0000'), 5)  # 紅色實線，寬度為5
        dash_pen = QPen(Qt.black, 3)  # 黑色線條，寬度為 3
        dash_pen.setStyle(Qt.CustomDashLine)  # 設置為自定義虛線
        dash_pen.setDashPattern([1, 3, 1, 3])  # 自定義虛線的模式：線段長度 1，空格 3，線段 1，空格 3
        for i in range(self.game.board_rows_nums):
            for j in range(self.game.board_cols_nums):
                x = self.BoardStart_pos[0] + j * self.gap
                y = self.BoardStart_pos[1] + i * self.gap
                painter.setPen(Dots_pen)
                if self.game.board[i][j] == 5:
                    painter.setPen(Dots_pen)
                    painter.drawEllipse(x-self.Dots_radius//2, y-self.Dots_radius//2, self.Dots_radius, self.Dots_radius)  # 繪製頂點
                elif self.game.board[i][j] == -1:
                    painter.setPen(Blue_solid_pen)
                    if i%2 == 0:    #偶數列，水平線
                        painter.drawLine(int(x-self.gap/2), y, int(x+self.gap/2), y)
                    else:
                        painter.drawLine(x, int(y-self.gap/2), x, int(y+self.gap/2))
                elif self.game.board[i][j] == 1:
                    painter.setPen(Red_solid_pen)
                    if i%2 == 0:    #偶數列，水平線
                        painter.drawLine(int(x-self.gap/2), y, int(x+self.gap/2), y)
                    else:
                        painter.drawLine(x, int(y-self.gap/2), x, int(y+self.gap/2))   
                elif self.game.board[i][j] == 0:
                    painter.setPen(dash_pen)  # 畫虛線(未選擇的邊)
                    if i%2 == 0:    #偶數列，水平線
                        painter.drawLine(x-self.gap//2, y, x+self.gap//2, y)  # 畫虛線(未選擇的邊)
                    else:
                        painter.drawLine(x, y-self.gap//2, x, y+self.gap//2)  # 畫虛線(未選擇的邊)
                elif self.game.board[i][j] == 7:
                    painter.setBrush(QColor('#0000E3'))
                    painter.drawRect(x-self.gap//2,y-self.gap//2,self.gap,self.gap)
                    painter.setBrush(Qt.NoBrush)  # 重置刷子
                elif self.game.board[i][j] == 9:
                    painter.setBrush(QColor('#FF0000'))
                    painter.drawRect(x-self.gap//2,y-self.gap//2,self.gap,self.gap)
                    painter.setBrush(Qt.NoBrush)
                    
    def mousePressEvent(self, event):
        if not self.mouse_events_enabled:
            return  # 如果滑鼠事件未啟用，則直接返回
        if event.button() == Qt.LeftButton:  # 確保是左鍵點擊
            x = event.x() - self.BoardStart_pos[0] + self.gap//2
            y = event.y() - self.BoardStart_pos[1] + self.gap//2
            
            # 計算點擊的行和列
            row = y // self.gap
            col = x // self.gap
            self.game.make_move(row,col)
            self.update()
            if self.mode != 1: 
                self.mouse_events_enabled = False
            
              
    
def window():
    app = QApplication(sys.argv)  # 設置系統與app的參數連結
    game = DaB(6, 6)  # 創建 DaB 類的實例
    win = GameWindow(game=game)  # 創建視窗物件
    win.show()  # 開啟視窗
    sys.exit(app.exec_())  # 確保視窗關閉是由按下視窗關閉建觸發

window()