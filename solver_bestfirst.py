import random
import time
import heapq
from solver_base import NQueensBaseSolver

class BestFirstSolver(NQueensBaseSolver):
    def solve_one_solution(self):
        start_board = [random.randint(0, self.N-1) for _ in range(self.N)]
        self.steps = 0
        start_time = time.perf_counter()
        heap = []
        heapq.heappush(heap, (self.compute_conflicts(start_board), start_board))
        visited = set()

        while heap:
            self.steps += 1
            conflicts, board = heapq.heappop(heap)
            if conflicts == 0:
                return board, time.perf_counter()-start_time, self.steps

            for col in range(self.N):
                for row in range(self.N):
                    if row != board[col]:
                        new_board = board[:]
                        new_board[col] = row
                        key = tuple(new_board)
                        if key not in visited:
                            visited.add(key)
                            heapq.heappush(heap, (self.compute_conflicts(new_board), new_board))

        return None, time.perf_counter()-start_time, self.steps
