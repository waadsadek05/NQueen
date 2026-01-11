from abc import ABC, abstractmethod

class NQueensBaseSolver(ABC):
    def __init__(self, N):
        self.N = N
        self.steps = 0

    @abstractmethod
    def solve_one_solution(self):
        pass

    # def compute_conflicts(self, board):
    #     conflicts = 0
    #     for i in range(self.N):
    #         for j in range(i+1, self.N):
    #             if board[i] == board[j] or abs(board[i]-board[j]) == abs(i-j):
    #                 conflicts += 1
    #     return conflicts

    def compute_conflicts(self, board):
        if board is None:
            return None

        conflicts = 0
        N = len(board)

        for c1 in range(N):
            for c2 in range(c1 + 1, N):
                r1, r2 = board[c1], board[c2]
                if r1 == r2:
                    conflicts += 1
                if abs(r1 - r2) == abs(c1 - c2):
                    conflicts += 1

        return conflicts
