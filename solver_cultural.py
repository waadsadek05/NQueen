
import random
import time
from solver_base import NQueensBaseSolver

class CulturalAlgorithmSolver(NQueensBaseSolver):
    def __init__(self, N, population_size=50, max_generations=1000):
        super().__init__(N)
        self.population_size = population_size
        self.max_generations = max_generations
        self.history = []  # ✅ إضافة التاريخ

    def initialize_population(self):
        return [[random.randint(0, self.N-1) for _ in range(self.N)] for _ in range(self.population_size)]

    def update_belief_space(self, population):
        population.sort(key=self.compute_conflicts)
        return population[:len(population)//2]

    def generate_offspring(self, belief_space):
        offspring = []
        for _ in range(self.population_size):
            parent = random.choice(belief_space)
            child = parent[:]
            col = random.randint(0, self.N-1)
            child[col] = random.randint(0, self.N-1)
            offspring.append(child)
        return offspring

    def solve_one_solution(self):
        population = self.initialize_population()
        self.steps = 0
        start_time = time.perf_counter()
        self.history = []  # ✅ إعادة تهيئة التاريخ

        for generation in range(self.max_generations):
            self.steps += 1
            
            # ✅ تسجيل التاريخ في كل جيل
            population.sort(key=self.compute_conflicts)
            best_ind = population[0]
            best_conflicts = self.compute_conflicts(best_ind)
            
            self.history.append({
                "generation": generation + 1,
                "best": best_ind[:],
                "conflicts": best_conflicts,
                "population": [ind[:] for ind in population]  # ✅ حفظ السكان للعرض
            })

            if best_conflicts == 0:
                return best_ind, time.perf_counter()-start_time, self.steps

            belief_space = self.update_belief_space(population)
            population = self.generate_offspring(belief_space)

        return None, time.perf_counter()-start_time, self.steps
