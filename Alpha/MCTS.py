import random
import math
from copy import deepcopy

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.score = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game_state.getValidMoves())

    def select_child(self):
        return max(self.children, key=lambda c: c.uct_value())

    def expand(self):
        valid_moves = self.game_state.getValidMoves()
        for move in valid_moves:
            if not any(child.move == move for child in self.children):
                new_state = deepcopy(self.game_state)
                new_state.make_move(*move)
                child = MCTSNode(new_state, parent=self, move=move)
                self.children.append(child)
                return child
        return None

    def update(self, result):
        self.visits += 1
        self.score += result

    def uct_value(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return self.score / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTS:
    def __init__(self, game_state, iterations=1000):
        self.root = MCTSNode(game_state)
        self.iterations = iterations

    def search(self):
        for _ in range(self.iterations):
            node = self.select(self.root)
            if not node.game_state.isGameOver():
                node = self.expand(node)
            result = self.simulate(node.game_state)
            self.backpropagate(node, result)

        return self.best_move()

    def select(self, node):
        while not node.game_state.isGameOver():
            if not node.is_fully_expanded():
                return node
            node = node.select_child()
        return node

    def expand(self, node):
        return node.expand()

    def simulate(self, game_state):
        state = deepcopy(game_state)
        while not state.isGameOver():
            moves = state.getValidMoves()
            move = random.choice(moves)
            state.make_move(*move)
        return state.GetWinner()

    def backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def best_move(self):
        return max(self.root.children, key=lambda c: c.visits).move

class MCTSPlayer:
    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.game_state = None

    def set_game_state(self, game_state):
        self.game_state = deepcopy(game_state)

    def update_game_state(self, move):
        if self.game_state:
            self.game_state.make_move(*move)

    def get_move(self):
        if not self.game_state:
            raise ValueError("Game state not set. Call set_game_state() first.")
        mcts = MCTS(self.game_state, self.iterations)
        return mcts.search()