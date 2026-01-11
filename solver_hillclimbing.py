
import random
import time
from solver_base import NQueensBaseSolver

# =====================
# Hill Climbing محسّنة
# =====================
class HillClimbingSolver(NQueensBaseSolver):
    def __init__(self, N, max_iterations=10000, max_restarts=50, heuristic=None):
        super().__init__(N)
        self.max_iterations = max_iterations
        self.max_restarts = max_restarts
        # لو ما فيش heuristic محدد، استخدم heuristic محسّنة
        self.heuristic = heuristic if heuristic else self.hill_heuristic

    # =====================
    # Heuristic جديدة
    # =====================
    def hill_heuristic(self, board):
        """
        Heuristic ممتازة لـ Hill Climbing:
        - conflict على نفس الصف = 1
        - conflict على القطر = 2 (أهم)
        """
        conflicts = 0
        N = len(board)
        for i in range(N):
            for j in range(i+1, N):
                if board[i] == board[j]:            # نفس الصف
                    conflicts += 1
                if abs(board[i]-board[j]) == j-i:  # نفس القطر
                    conflicts += 2
        return conflicts

    # =====================
    # حل المشكلة
    # =====================
    def solve_one_solution(self):
        self.steps = 0
        start_time = time.perf_counter()

        for restart in range(self.max_restarts):
            # بدء board عشوائي
            board = [random.randint(0, self.N-1) for _ in range(self.N)]

            for _ in range(self.max_iterations):
                self.steps += 1
                conflicts = self.heuristic(board)

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
                        new_conflicts = self.heuristic(board)
                        if new_conflicts < min_conflicts:
                            min_conflicts = new_conflicts
                            best_board = board[:]
                    board[col] = original_row

                if min_conflicts >= conflicts:
                    # وصلنا local minimum → نعمل restart
                    break
                else:
                    board = best_board

        elapsed = time.perf_counter() - start_time
        return None, elapsed, self.steps

