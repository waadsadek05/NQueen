import random
import time
from solver_base import NQueensBaseSolver
import heapq

class BestFirstSolver(NQueensBaseSolver):
    def __init__(self, N, heuristic='h2'):
        super().__init__(N)
        self.heuristic_name = heuristic
        # Set heuristic function dynamically
        if heuristic == 'h1':
            self.heuristic_func = self.h1
        else:
            self.heuristic_func = self.h2

    def solve_one_solution(self):
        start_board = [random.randint(0, self.N-1) for _ in range(self.N)]
        self.steps = 0
        start_time = time.perf_counter()
        heap = []
        heapq.heappush(heap, (self.heuristic_func(start_board), start_board))
        visited = set()

        while heap:
            self.steps += 1
            heuristic_val, board = heapq.heappop(heap)
            if heuristic_val == 0:
                return board, time.perf_counter()-start_time, self.steps

            for col in range(self.N):
                for row in range(self.N):
                    if row != board[col]:
                        new_board = board[:]
                        new_board[col] = row
                        key = tuple(new_board)
                        if key not in visited:
                            visited.add(key)
                            heapq.heappush(heap, (self.heuristic_func(new_board), new_board))

        return None, time.perf_counter()-start_time, self.steps
