import random
import time
from solver_base import NQueensBaseSolver


class CulturalAlgorithmSolver(NQueensBaseSolver):
    """
    Optimized Cultural Algorithm for N-Queens.
    Fully compatible with GUI expecting:
        board, elapsed_time, steps
    """

    def __init__(
        self, N,
        population_size=80,
        max_generations=100,
        acceptance_rate=0.4,
        mutation_rate=0.1,
        use_crossover=True,
        tournament_k=3,
        seed=None
    ):
        super().__init__(N)

        if seed is not None:
            random.seed(seed)

        self.N = N
        self.population_size = population_size
        self.max_generations = max_generations
        self.acceptance_rate = acceptance_rate
        self.mutation_rate = mutation_rate
        self.use_crossover = use_crossover
        self.tournament_k = tournament_k

        # Belief space (normative + situational)
        self.belief_space = {
            "normative": {"L": [0]*N, "U": [N-1]*N},
            "situational": {"best": None, "conflicts": float("inf")}
        }

        self.history = []
        self.steps = 0

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def initialize_population(self):
        return [
            [random.randint(0, self.N - 1) for _ in range(self.N)]
            for _ in range(self.population_size)
        ]

    def evaluate_population(self, population):
        evaluated = [(ind, self.compute_conflicts(ind)) for ind in population]
        evaluated.sort(key=lambda x: x[1])
        return evaluated

    # -------------------------------------------------------------
    # Acceptance + Belief update
    # -------------------------------------------------------------
    def select_accepted(self, evaluated_pop):
        k = max(1, int(self.acceptance_rate * len(evaluated_pop)))
        return [ind for ind, _ in evaluated_pop[:k]]

    def update_belief_space(self, accepted, evaluated):
        if not accepted:
            return

        L = []
        U = []
        for j in range(self.N):
            col_vals = [ind[j] for ind in accepted]
            L.append(min(col_vals))
            U.append(max(col_vals))
        self.belief_space["normative"]["L"] = L
        self.belief_space["normative"]["U"] = U

        best_ind, best_conf = evaluated[0]
        if best_conf < self.belief_space["situational"]["conflicts"]:
            self.belief_space["situational"]["best"] = best_ind[:]
            self.belief_space["situational"]["conflicts"] = best_conf

    # -------------------------------------------------------------
    # Influence + Variation
    # -------------------------------------------------------------
    def influence(self, parent=None):
        L = self.belief_space["normative"]["L"]
        U = self.belief_space["normative"]["U"]

        if parent is None:
            return [random.randint(L[j], U[j]) for j in range(self.N)]

        child = parent[:]
        for j in range(self.N):
            if random.random() < 0.8:
                low, high = L[j], U[j]
                if low > high:
                    low, high = 0, self.N - 1
                child[j] = random.randint(low, high)
        return child

    def mutate(self, ind):
        if random.random() >= self.mutation_rate:
            return ind

        L = self.belief_space["normative"]["L"]
        U = self.belief_space["normative"]["U"]

        col = random.randint(0, self.N - 1)
        low, high = L[col], U[col]
        if low > high:
            low, high = 0, self.N - 1
        ind[col] = random.randint(low, high)
        return ind

    def crossover(self, p1, p2):
        point = random.randint(1, self.N - 1)
        return p1[:point] + p2[point:]

    def make_offspring(self, accepted):
        offspring = []
        for _ in range(self.population_size):

            if self.use_crossover and len(accepted) >= 2 and random.random() < 0.5:
                parent1 = random.choice(accepted)
                parent2 = random.choice(accepted)
                child = self.crossover(parent1, parent2)
                child = self.influence(child)
            else:
                if accepted:
                    candidates = [
                        random.choice(accepted)
                        for _ in range(min(self.tournament_k, len(accepted)))
                    ]
                    parent = min(candidates, key=lambda ind: self.compute_conflicts(ind))
                    child = self.influence(parent)
                else:
                    child = self.influence(None)

            child = self.mutate(child)
            offspring.append(child)

        return offspring

    # -------------------------------------------------------------
    # SOLVE (main loop) — returns board, elapsed_time, steps
    # -------------------------------------------------------------
    def solve(self):
        population = self.initialize_population()
        start = time.perf_counter()
        self.steps = 0
        self.history = []

        evaluated = self.evaluate_population(population)
        accepted = self.select_accepted(evaluated)
        self.update_belief_space(accepted, evaluated)

        for gen in range(1, self.max_generations + 1):
            self.steps = gen

            evaluated = self.evaluate_population(population)
            best_ind, best_conf = evaluated[0]

            self.history.append({
                "generation": gen,
                "best": best_ind[:],
                "conflicts": best_conf,
                "belief": {
                    "L": self.belief_space["normative"]["L"][:],
                    "U": self.belief_space["normative"]["U"][:]
                }
            })

            if best_conf == 0:
                return best_ind, time.perf_counter() - start, gen

            accepted = self.select_accepted(evaluated)
            self.update_belief_space(accepted, evaluated)

            population = self.make_offspring(accepted)

        # no perfect solution — return best so far
        situ = self.belief_space["situational"]
        return situ["best"], time.perf_counter() - start, self.steps

    # -------------------------------------------------------------
    # Compatibility with GUI
    # -------------------------------------------------------------
    def solve_one_solution(self):
        return self.solve()
