import random
import time
from solver_base import NQueensBaseSolver

class NQueensSolver(NQueensBaseSolver):
    def solve_one_solution(self):
        board = [-1] * self.N
        self.steps = 0
        start_time = time.perf_counter()
        if self._backtrack(0, board):
            return board, time.perf_counter()-start_time, self.steps
        return None, time.perf_counter()-start_time, self.steps

    def _backtrack(self, col, board):
        self.steps += 1
        if col == self.N:
            return True
        rows = list(range(self.N))
        random.shuffle(rows)
        for row in rows:
            if self.is_safe(board, row, col):
                board[col] = row
                if self._backtrack(col+1, board):
                    return True
                board[col] = -1
        return False

    def is_safe(self, board, row, col):
        for c in range(col):
            r = board[c]
            if r == row or abs(r - row) == abs(c - col):
                return False
        return True
