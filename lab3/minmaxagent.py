from exceptions import AgentException
import copy
import math
import random as r

class MinMaxAgent:
    def __init__(self, my_token='o', max_depth=3):
        self.my_token = my_token
        self.max_depth = max_depth

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        best_move = self.min_max(connect4, self.max_depth)
        return best_move

    def min_max(self, connect4, depth):
        if depth == 0 or connect4.game_over:
            return None

        best_score = float('-inf')
        best_move = None

        for column in connect4.possible_drops():
            temp_connect4 = self.simulate_drop(connect4, column)
            score = self.min_value(temp_connect4, depth - 1)
            if score == 0:
                score = self.heuristic(connect4, column)
            if score > best_score:
                print(self.my_token, score)
                best_score = score
                best_move = column

        return best_move

    def max_value(self, connect4, depth):
        if depth == 0 or connect4.game_over:
            return self.check_result(connect4)

        value = float('-inf')
        for column in connect4.possible_drops():
            temp_connect4 = self.simulate_drop(connect4, column)
            value = max(value, self.min_value(temp_connect4, depth - 1))
        return value

    def min_value(self, connect4, depth):
        if depth == 0 or connect4.game_over:
            return self.check_result(connect4)

        value = float('inf')
        for column in connect4.possible_drops():
            temp_connect4 = self.simulate_drop(connect4, column)
            value = min(value, self.max_value(temp_connect4, depth - 1))
        return value

    def simulate_drop(self, connect4, column):
        temp_connect4 = copy.deepcopy(connect4)
        temp_connect4.drop_token(column)
        return temp_connect4

    def check_result(self, connect4):
        if connect4.wins == self.my_token:
            return 1
        elif connect4.wins is None:
            return 0
        else:
            return -1

    def heuristic(self, connect4, column):
        width = connect4.width
        middle = width / 2
        distance = math.fabs(middle - column)
        scaled_distance = 2 * (1 - (distance / middle)) - 1
        #return max(-0.9, min(0.9, scaled_distance))
        #return 0
        return r.uniform(-0.9, 0.9)



