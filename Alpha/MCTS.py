import math
import random
from copy import deepcopy

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.score = 0

class MCTSPlayer:
    def __init__(self, num_simulations, exploration_weight=1.41, max_depth=20):
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.game_state = None

    def get_move(self):
        if not self.game_state:
            raise ValueError("Game state not set")

        root = MCTSNode(deepcopy(self.game_state))
        
        for _ in range(self.num_simulations):
            node = self.select(root)
            simulation_result = self.simulate(node.game_state)
            self.backpropagate(node, simulation_result)

        if not root.children:
            return random.choice(self.game_state.getValidMoves())

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def select(self, node):
        while not node.game_state.isGameOver():
            if len(node.children) < len(node.game_state.getValidMoves()):
                return self.expand(node)
            else:
                node = self.uct_select(node)
        return node

    def expand(self, node):
        valid_moves = node.game_state.getValidMoves()
        unvisited_moves = [move for move in valid_moves if not any(child.move == move for child in node.children)]
        
        if not unvisited_moves:
            return node

        move = self.choose_expansion_move(node.game_state, unvisited_moves)
        new_state = deepcopy(node.game_state)
        new_state.make_move(*move)
        
        new_node = MCTSNode(new_state, parent=node, move=move)
        node.children.append(new_node)
        return new_node

    def choose_expansion_move(self, game_state, moves):
        for move in moves:
            temp_state = deepcopy(game_state)
            temp_state.make_move(*move)
            if temp_state.checkBox(temp_state.board):
                return move
        return random.choice(moves)

    def simulate(self, game_state):
        state = deepcopy(game_state)
        depth = 0
        while not state.isGameOver() and depth < self.max_depth:
            move = self.choose_simulation_move(state)
            if move is None:
                break
            state.make_move(*move)
            depth += 1

        return self.evaluate(state)

    def choose_simulation_move(self, game_state):
        valid_moves = game_state.getValidMoves()
        if not valid_moves:
            return None
        for move in valid_moves:
            temp_state = deepcopy(game_state)
            temp_state.make_move(*move)
            if temp_state.checkBox(temp_state.board):
                return move
        return random.choice(valid_moves)

    def evaluate(self, state):
        if state.isGameOver():
            winner = state.GetWinner()
            if winner is None:
                return 0
            return winner
        
        # 使用簡單的啟發式評估
        score_diff = state.p1_scores - state.p2_scores
        if score_diff > 0:
            return -1  # 有利於玩家 1 (對手)
        elif score_diff < 0:
            return 1   # 有利於玩家 2 (自己)
        else:
            return 0   # 平局

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            if result is not None:
                node.score += result if node.game_state.current_player == result else -result
            node = node.parent

    def uct_select(self, node):
        log_parent_visits = math.log(node.visits)
        return max(node.children, key=lambda c: self.ucb1(c, log_parent_visits))

    def ucb1(self, node, log_parent_visits):
        if node.visits == 0:
            return float('inf')
        exploitation = node.score / node.visits
        exploration = math.sqrt(log_parent_visits / node.visits)
        return exploitation + self.exploration_weight * exploration