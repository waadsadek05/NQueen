import random
import time
from solver_base import NQueensBaseSolver

class HillClimbingSolver(NQueensBaseSolver):
    def __init__(self, N, max_iterations=10000, max_restarts=50, heuristic='h2'):
        super().__init__(N)
        self.max_iterations = max_iterations
        self.max_restarts = max_restarts
        self.heuristic_name = heuristic
        
        # Set heuristic function dynamically
        if heuristic == 'h1':
            self.heuristic_func = self.h1
        else:
            self.heuristic_func = self.h2

    def solve_one_solution(self):
        self.steps = 0
        start_time = time.perf_counter()

        for restart in range(self.max_restarts):
            board = [random.randint(0, self.N-1) for _ in range(self.N)]

            for _ in range(self.max_iterations):
                self.steps += 1
                conflicts = self.heuristic_func(board)

                if conflicts == 0:
                    elapsed = time.perf_counter() - start_time
                    return board, elapsed, self.steps

                best_board = board[:]
                min_conflicts = conflicts

                for col in range(self.N):
                    original_row = board[col]
                    for row in range(self.N):
                        if row == original_row:
                            continue
                        board[col] = row
                        new_conflicts = self.heuristic_func(board)
                        if new_conflicts < min_conflicts:
                            min_conflicts = new_conflicts
                            best_board = board[:]
                    board[col] = original_row

                if min_conflicts >= conflicts:
                    break
                else:
                    board = best_board

        elapsed = time.perf_counter() - start_time
        return None, elapsed, self.steps
