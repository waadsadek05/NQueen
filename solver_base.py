import random
import time
from abc import ABC, abstractmethod  

class NQueensBaseSolver(ABC):
    def __init__(self, N):
        self.N = N
        self.steps = 0

    @abstractmethod
    def solve_one_solution(self):
        pass

    # Heuristic Function 1 (h1): Number of Conflicted Queens
    def h1(self, board):
        """Count queens involved in at least one conflict."""
        if board is None:
            return None
        
        N = len(board)
        conflicted_queens = set()
        
        for i in range(N):
            for j in range(i + 1, N):
                if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                    conflicted_queens.add(i)
                    conflicted_queens.add(j)
        
        return len(conflicted_queens)

    # Heuristic Function 2 (h2): Total Number of Conflicting Pairs
    def h2(self, board):
        """Count total conflicting pairs (existing compute_conflicts)."""
        return self.compute_conflicts(board)

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
