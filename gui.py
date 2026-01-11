import tkinter as tk
from tkinter import ttk, messagebox
from solver_backtracking import NQueensSolver
from solver_hillclimbing import HillClimbingSolver
from solver_bestfirst import BestFirstSolver
from solver_cultural import CulturalAlgorithmSolver

import matplotlib.pyplot as plt

CELL_SIZE = 80

class NQueensGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("N-Queens Solver (OOP)")

        self.algorithms = [
            "Backtracking",
            "Hill-Climbing",
            "Best-First Search",
            "Cultural Algorithm"
        ]

        self.saved_solutions = {
            algo: {"board": None, "time": 0, "steps": 0, "conflicts": None, "solver_obj": None}
            for algo in self.algorithms
        }

        self.canvas = None
        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)

        tk.Label(top_frame, text="Board Size N:").grid(row=0, column=0)
        self.n_entry = tk.Entry(top_frame, width=5)
        self.n_entry.grid(row=0, column=1)

        tk.Label(top_frame, text="Algorithm:").grid(row=0, column=2, padx=5)
        self.algorithm_var = tk.StringVar(value=self.algorithms[0])

        self.algorithm_menu = ttk.Combobox(
            top_frame, textvariable=self.algorithm_var,
            values=self.algorithms, state="readonly"
        )
        self.algorithm_menu.grid(row=0, column=3)

        tk.Button(top_frame, text="Solve", command=self.solve_nqueens).grid(row=0, column=4, padx=5)
        tk.Button(top_frame, text="Compare", command=self.display_comparison).grid(row=0, column=5, padx=5)
        tk.Button(top_frame, text="Visualize Animation", command=self.visualize_animation).grid(row=0, column=6, padx=5)
        tk.Button(top_frame, text="Plot Graph", command=self.plot_graph).grid(row=0, column=7, padx=5)

        self.solution_frame = tk.Frame(self.root)
        self.solution_frame.pack(pady=10)

        self.time_label = tk.Label(self.root, text="")
        self.time_label.pack()
        self.steps_label = tk.Label(self.root, text="")
        self.steps_label.pack()

    def compute_conflicted_indices(self, board):
        """دالة مساعدة لحساب الملكات المتعارضة"""
        conflicts = []
        N = len(board)
        for i in range(N):
            for j in range(i+1, N):
                if board[i] == board[j] or abs(board[i]-board[j]) == abs(i-j):
                    conflicts.extend([i, j])
        return list(set(conflicts))

    def draw_board(self, board, canvas, highlight_conflicts=False):
        canvas.delete("all")
        if board is None:
            return
        N = len(board)

        conflicts = []
        if highlight_conflicts:
            conflicts = self.compute_conflicted_indices(board)

        for r in range(N):
            for c in range(N):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                color = "white" if (r + c) % 2 == 0 else "gray"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color)

                if board[c] == r:
                    fill_color = "black" if c in conflicts else "red"
                    canvas.create_text(
                        x1 + CELL_SIZE / 2,
                        y1 + CELL_SIZE / 2,
                        text="♛",
                        font=("Arial", int(CELL_SIZE / 1.5)),
                        fill=fill_color
                    )

    def display_board(self, board, N):
        global CELL_SIZE
        CELL_SIZE = min(600 // max(1, N), 80)

        if self.canvas:
            self.canvas.destroy()

        self.canvas = tk.Canvas(self.solution_frame, width=N * CELL_SIZE, height=N * CELL_SIZE)
        self.canvas.pack()

        if board:
            self.draw_board(board, self.canvas)
        else:
            self.canvas.create_text(
                N * CELL_SIZE // 2,
                N * CELL_SIZE // 2,
                text="No Solution",
                font=("Arial", 18)
            )

    def solve_nqueens(self):
        try:
            N = int(self.n_entry.get())
            if N < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a valid positive integer for N.")
            return

        algo = self.algorithm_var.get()

        if algo == "Backtracking":
            solver = NQueensSolver(N)
        elif algo == "Hill-Climbing":
            solver = HillClimbingSolver(N)
        elif algo == "Best-First Search":
            solver = BestFirstSolver(N)
        elif algo == "Cultural Algorithm":
            solver = CulturalAlgorithmSolver(N)
        else:
            messagebox.showerror("Error", "Unknown algorithm selected.")
            return

        board, elapsed, steps = solver.solve_one_solution()
        conflicts = solver.compute_conflicts(board) if board else None

        self.saved_solutions[algo] = {
            "board": board,
            "time": elapsed,
            "steps": steps,
            "conflicts": conflicts,
            "solver_obj": solver
        }

        self.display_board(board, N)
        self.time_label.config(text=f"Time: {elapsed:.6f}s")
        self.steps_label.config(text=f"Steps: {steps}")

    def display_comparison(self):
        try:
            N = int(self.n_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a valid positive integer for N.")
            return

        compare_win = tk.Toplevel(self.root)
        compare_win.title("Algorithms Comparison")

        for idx, algo in enumerate(self.algorithms):
            frame = tk.LabelFrame(compare_win, text=algo, padx=5, pady=5)
            frame.grid(row=0, column=idx, padx=5)

            global CELL_SIZE
            CELL_SIZE = min(200 // max(1, N), 80)

            canvas = tk.Canvas(frame, width=N * CELL_SIZE, height=N * CELL_SIZE)
            canvas.pack()

            data = self.saved_solutions[algo]

            if data["board"] and any(r != -1 for r in data["board"]):
                self.draw_board(data["board"], canvas)
            else:
                canvas.create_text(
                    N * CELL_SIZE // 2,
                    N * CELL_SIZE // 2,
                    text="No Solution",
                    font=("Arial", 16)
                )

            tk.Label(frame, text=f"Time: {data['time']:.6f}s").pack()
            tk.Label(frame, text=f"Steps: {data['steps']}").pack()
            tk.Label(frame, text=f"Conflicts: {data['conflicts']}").pack()

        summary_frame = tk.Frame(compare_win)
        summary_frame.grid(row=1, column=0, columnspan=len(self.algorithms), pady=10)

        headers = ["Algorithm", "Time (s)", "Steps", "Conflicts"]
        for i, h in enumerate(headers):
            tk.Label(summary_frame, text=h, width=15, borderwidth=1,
                     relief="solid").grid(row=0, column=i)

        for i, algo in enumerate(self.algorithms):
            data = self.saved_solutions[algo]
            tk.Label(summary_frame, text=algo, width=15, borderwidth=1,
                     relief="solid").grid(row=i+1, column=0)
            tk.Label(summary_frame, text=f"{data['time']:.6f}", width=15,
                     borderwidth=1, relief="solid").grid(row=i+1, column=1)
            tk.Label(summary_frame, text=data["steps"], width=15,
                     borderwidth=1, relief="solid").grid(row=i+1, column=2)
            tk.Label(summary_frame, text=data["conflicts"], width=15,
                     borderwidth=1, relief="solid").grid(row=i+1, column=3)

    def visualize_animation(self):
        data = self.saved_solutions.get("Cultural Algorithm")
        if not data:
            messagebox.showinfo("Info", "Run Cultural Algorithm first to get data.")
            return

        solver = data.get("solver_obj")
        if solver is None or not hasattr(solver, "history") or len(solver.history) == 0:
            messagebox.showinfo("Info", "No history available. Run Cultural Algorithm first (Solve).")
            return

        history = solver.history
        N = solver.N

        if N > 20:
            ok = messagebox.askyesno("Warning", f"N = {N} is large. Animation may be slow. Continue?")
            if not ok:
                return

        anim_win = tk.Toplevel(self.root)
        anim_win.title("Cultural Algorithm Animation")

        global CELL_SIZE
        CELL_SIZE = min(600 // max(1, N), 40)

        main_canvas = tk.Canvas(anim_win, width=N * CELL_SIZE, height=N * CELL_SIZE)
        main_canvas.pack(padx=10, pady=5)

        pop_canvas = tk.Canvas(anim_win, width=N * CELL_SIZE, height=N * CELL_SIZE//2)
        pop_canvas.pack(padx=10, pady=5)

        status_label = tk.Label(anim_win, text="")
        status_label.pack()

        delay_ms = 300

        def show_frame(idx=0):
            if idx >= len(history):
                status_label.config(text=f"Finished. Generations: {len(history)}")
                return
            entry = history[idx]
            board = entry["best"]
            generation = entry["generation"]
            conflicts = entry["conflicts"]
            population = entry.get("population", [])

            # main board مع highlight
            self.draw_board(board, main_canvas, highlight_conflicts=True)

            # population mini-boards
            pop_canvas.delete("all")
            pop_size = len(population)
            if pop_size > 0:
                mini_size = CELL_SIZE // 3
                for p_idx, indiv in enumerate(population):
                    offset_x = p_idx * (mini_size * N / pop_size)
                    for col, row in enumerate(indiv):
                        x1 = offset_x + col * mini_size
                        y1 = 0 + row * mini_size
                        x2 = x1 + mini_size
                        y2 = y1 + mini_size
                        color = "black" if col in self.compute_conflicted_indices(indiv) else "red"
                        pop_canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline=color)
                        pop_canvas.create_text(x1+mini_size/2, y1+mini_size/2, text="♛", fill=color, font=("Arial", int(mini_size/1.5)))

            status_label.config(text=f"Generation: {generation}  Conflicts: {conflicts}")
            anim_win.after(delay_ms, lambda: show_frame(idx + 1))

        show_frame(0)

    def plot_graph(self):
        data = self.saved_solutions.get("Cultural Algorithm")
        if not data:
            messagebox.showinfo("Info", "Run Cultural Algorithm first to get data.")
            return

        solver = data.get("solver_obj")
        if solver is None or not hasattr(solver, "history") or len(solver.history) == 0:
            messagebox.showinfo("Info", "No history available. Run Cultural Algorithm first (Solve).")
            return

        history = solver.history
        generations = [entry["generation"] for entry in history]
        conflicts = [entry["conflicts"] for entry in history]

        plt.figure(figsize=(8, 4))
        plt.plot(generations, conflicts, marker='o')
        plt.title("Cultural Algorithm — Conflicts vs Generations")
        plt.xlabel("Generation")
        plt.ylabel("Conflicts")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

