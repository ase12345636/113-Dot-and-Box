from PyQt5.QtWidgets import QMainWindow, QApplication, QComboBox, QSlider
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
    def __init__(self):
        super(GameWindow, self).__init__()
        self.setGeometry(500, 200, 800, 600)
        self.setWindowTitle('Dots and Boxes')
        self.gap = 50
        self.Dots_radius = 50
        self.BoardStart_pos = (50, 50)
        self.mouse_events_enabled = False
        self.paint_events_enabled = False
        self.game_row = 3
        self.game_col = 3
        
        # 設置字體樣式
        self.font = QFont("Arial", 11, QFont.Bold)  # 使用 Arial 字型，大小 14，粗體
        
        self.init_ui()

    def init_ui(self):
        #玩家1分數
        self.P1_score_label = QtWidgets.QLabel(self)
        self.P1_score_label.setText(f'Player1 scores: 0')
        self.P1_score_label.setGeometry(600, 100, 175, 25)
        self.P1_score_label.setFont(self.font)  # 設置標籤字型

        #玩家2分數
        self.P2_score_label = QtWidgets.QLabel(self)
        self.P2_score_label.setText(f'Player2 scores: 0')
        self.P2_score_label.setGeometry(600, 150, 175, 25)
        self.P2_score_label.setFont(self.font)  # 設置標籤字型

        # 下拉選單
        self.mode_combo_box = QComboBox(self)
        self.mode_combo_box.addItem("人類 VS 人類")
        self.mode_combo_box.addItem("人類 VS 隨機")
        self.mode_combo_box.addItem("人類 VS 模型")
        self.mode_combo_box.setGeometry(600, 200, 175, 25)
        self.mode_combo_box.setFont(self.font)  # 設置下拉選單字型
        
        #調整遊戲行數之滑條
        self.row_slider = QSlider(Qt.Horizontal, self)
        self.row_slider.setGeometry(600,250,90,25)
        self.row_slider.setMinimum(3)  # 設定滑桿最小值
        self.row_slider.setMaximum(6)  # 設定滑桿最大值
        self.row_slider.setValue(3)  # 設定滑桿初始值
        self.row_slider.setTickPosition(QSlider.TicksBelow)  # 設置刻度線的位置
        self.row_slider.setTickInterval(1)  # 設置刻度線的間距
        self.row_slider.valueChanged.connect(self.OnRowSlide)  # 當滑桿的值變化時觸發
        
        #調整遊戲列數之滑條
        self.col_slider = QSlider(Qt.Horizontal, self)
        self.col_slider.setGeometry(600,300,90,25)
        self.col_slider.setMinimum(3)  # 設定滑桿最小值
        self.col_slider.setMaximum(6)  # 設定滑桿最大值
        self.col_slider.setValue(3)  # 設定滑桿初始值
        self.col_slider.setTickPosition(QSlider.TicksBelow)  # 設置刻度線的位置
        self.col_slider.setTickInterval(1)  # 設置刻度線的間距
        self.col_slider.valueChanged.connect(self.OnColSlide)  # 當滑桿的值變化時觸發
        
        #展示遊戲大小文字
        self.size_label = QtWidgets.QLabel(self)
        self.size_label.setText(f"{self.game_row} X {self.game_col}")
        self.size_label.setGeometry(700, 275, 90, 25)
        self.size_label.setFont(self.font)  # 設置下拉選單字型

        #開始按鈕
        self.StartButton = QtWidgets.QPushButton(self)
        self.StartButton.setText("Start!!!")
        self.StartButton.setGeometry(600, 350, 175, 25)
        self.StartButton.setFont(self.font)  # 設置按鈕字型
        self.StartButton.clicked.connect(self.OnClickStartButton)
        
        #終局文字
        self.winner_label = QtWidgets.QLabel(self)
        self.winner_label.setText("")
        self.winner_label.setGeometry(600,400,200,50)
        self.winner_label.setFont(QFont("Arial", 20, QFont.Bold))
        
    
    def OnRowSlide(self):
        self.game_row = self.row_slider.value()
        self.size_label.setText(f"{self.game_row} X {self.game_col}")
    def OnColSlide(self):
        self.game_col = self.col_slider.value()
        self.size_label.setText(f"{self.game_row} X {self.game_col}")
    

    def OnClickStartButton(self):
        #初始化遊戲
        self.game = DaB(self.game_row,self.game_col)
        self.P1_score_label.setText(f'Player1 scores: {self.game.p1_scores}')
        self.P2_score_label.setText(f'Player2 scores: {self.game.p2_scores}')
        self.winner_label.setText("")
        self.update()
        self.paint_events_enabled = True

        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()  # 停止已經存在的計時器

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

        # 每過100毫秒檢查一次遊戲並跳至game_loop更新畫面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(100)

    def game_loop(self):
        if self.game.isGameOver():
            winner = self.game.GetWinner()
            if winner == -1:
                self.winner_label.setText("Player 1")
                self.winner_label.setStyleSheet("color: #0000E3;")
            elif winner == 0:
                self.winner_label.setText("Tie")
                self.winner_label.setStyleSheet("color: #000000;")
            elif winner == 1:
                self.winner_label.setText("Player 2")
                self.winner_label.setStyleSheet("color: #FF0000;")
            self.timer.stop()
            return

        if self.mode != 1:  #非雙人類對戰
            if self.game.current_player == -1:
                self.mouse_events_enabled = True    #人類方使用滑鼠控制棋盤
            else:
                r, c = self.p2.get_move()
                self.game.make_move(r, c)
                self.update()
                self.mouse_events_enabled = False
        else:
            self.mouse_events_enabled = True
    
    #繪畫事件，用於觸發棋盤繪製    
    def paintEvent(self, event):
        if not self.paint_events_enabled:
            return
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
    win = GameWindow()  # 創建視窗物件
    win.show()  # 開啟視窗
    sys.exit(app.exec_())  # 確保視窗關閉是由按下視窗關閉建觸發

window()