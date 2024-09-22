import math
import random
from copy import deepcopy

# MCTS 節點類別，代表蒙地卡羅樹搜索中的每個節點
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state  # 節點的遊戲狀態
        self.parent = parent  # 父節點
        self.move = move  # 對應的移動
        self.children = []  # 子節點列表
        self.visits = 0  # 訪問次數
        self.score = 0  # 總得分

# MCTS 玩家類別，使用蒙地卡羅樹搜索進行決策
class MCTSPlayer:
    def __init__(self, num_simulations, exploration_weight=1.41, max_depth=20):
        self.num_simulations = num_simulations  # 蒙地卡羅模擬次數
        self.exploration_weight = exploration_weight  # 探索權重
        self.max_depth = max_depth  # 模擬的最大深度
        self.game_state = None  # 遊戲狀態

    # 獲取下一步移動
    def get_move(self):
        if not self.game_state:
            raise ValueError("Game state not set")  # 若遊戲狀態未設置，則拋出錯誤

        root = MCTSNode(deepcopy(self.game_state))  # 根節點為當前遊戲狀態的複製
        
        # 進行多次模擬
        for _ in range(self.num_simulations):
            node = self.select(root)  # 選擇節點
            simulation_result = self.simulate(node.game_state)  # 模擬遊戲結果
            self.backpropagate(node, simulation_result)  # 回傳模擬結果

        # 如果根節點沒有子節點，則隨機選擇一個有效移動
        if not root.children:
            return random.choice(self.game_state.getValidMoves())

        # 選擇訪問次數最多的子節點
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    # 節點選擇過程
    def select(self, node):
        while not node.game_state.isGameOver():
            # 若還有未探索的移動，則擴展節點
            if len(node.children) < len(node.game_state.getValidMoves()):
                return self.expand(node)
            else:
                node = self.uct_select(node)  # 否則使用 UCT 選擇節點
        return node

    # 擴展節點
    def expand(self, node):
        valid_moves = node.game_state.getValidMoves()  # 獲取有效移動
        # 找出尚未被探索的移動
        unvisited_moves = [move for move in valid_moves if not any(child.move == move for child in node.children)]
        
        if not unvisited_moves:
            return node

        move = self.choose_expansion_move(node.game_state, unvisited_moves)  # 選擇擴展的移動
        new_state = deepcopy(node.game_state)  # 複製當前遊戲狀態
        new_state.make_move(*move)  # 執行該移動
        
        new_node = MCTSNode(new_state, parent=node, move=move)  # 創建新節點
        node.children.append(new_node)  # 將新節點添加為子節點
        return new_node

    # 選擇擴展的移動
    def choose_expansion_move(self, game_state, moves):
        # 嘗試找到可以得分的移動
        for move in moves:
            temp_state = deepcopy(game_state)
            temp_state.make_move(*move)
            if temp_state.checkBox(temp_state.board):  # 如果能完成一個方框，則選擇該移動
                return move
        return random.choice(moves)  # 否則隨機選擇一個移動

    # 模擬遊戲進行
    def simulate(self, game_state):
        state = deepcopy(game_state)  # 複製遊戲狀態
        depth = 0
        while not state.isGameOver() and depth < self.max_depth:  # 直到遊戲結束或達到最大深度
            move = self.choose_simulation_move(state)  # 選擇模擬中的移動
            if move is None:
                break
            state.make_move(*move)  # 執行該移動
            depth += 1

        return self.evaluate(state)  # 回傳模擬結果的評估值

    # 選擇模擬中的移動
    def choose_simulation_move(self, game_state):
        valid_moves = game_state.getValidMoves()  # 獲取有效移動
        if not valid_moves:
            return None
        # 優先選擇可以得分的移動
        for move in valid_moves:
            temp_state = deepcopy(game_state)
            temp_state.make_move(*move)
            if temp_state.checkBox(temp_state.board):  # 如果移動能得分，則選擇該移動
                return move
        return random.choice(valid_moves)  # 否則隨機選擇一個移動

    # 評估遊戲狀態
    def evaluate(self, state):
        if state.isGameOver():  # 如果遊戲已結束
            winner = state.GetWinner()  # 獲取勝利者
            if winner is None:
                return 0  # 平局返回 0
            return winner
        
        # 根據兩名玩家的得分差評估
        score_diff = state.p1_scores - state.p2_scores
        if score_diff > 0:
            return -1  # 玩家 1 勝利
        elif score_diff < 0:
            return 1  # 玩家 2 勝利
        else:
            return 0  # 平局

    # 回傳模擬結果
    def backpropagate(self, node, result):
        while node:
            node.visits += 1  # 訪問次數增加
            if result is not None:
                node.score += result if node.game_state.current_player == result else -result  # 根據結果更新節點得分
            node = node.parent  # 繼續回傳到父節點

    # 使用 UCT (Upper Confidence Bound) 選擇節點
    def uct_select(self, node):
        log_parent_visits = math.log(node.visits)  # 計算父節點訪問次數的對數
        return max(node.children, key=lambda c: self.ucb1(c, log_parent_visits))  # 根據 UCB1 選擇子節點

    # UCB1 函數計算
    def ucb1(self, node, log_parent_visits):
        if node.visits == 0:
            return float('inf')  # 如果該節點尚未被訪問，則返回無限大
        exploitation = node.score / node.visits  # 利用值
        exploration = math.sqrt(log_parent_visits / node.visits)  # 探索值
        return exploitation + self.exploration_weight * exploration  # 返回 UCB1 值
