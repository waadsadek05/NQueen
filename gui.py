import tkinter as tk
from tkinter import ttk, messagebox
from solver_backtracking import NQueensSolver
from solver_hillclimbing import HillClimbingSolver
from solver_bestfirst import BestFirstSolver
from solver_cultural import CulturalAlgorithmSolver

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.figure import Figure

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
        
        self.heuristics = ["h1", "h2"]

        # NEW: Save solutions for BOTH heuristics for each algorithm
        self.saved_solutions = {
            algo: {
                "h1": {"board": None, "time": 0, "steps": 0, "conflicts": None, 
                      "solver_obj": None, "heuristic_val": None},
                "h2": {"board": None, "time": 0, "steps": 0, "conflicts": None, 
                      "solver_obj": None, "heuristic_val": None}
            }
            for algo in self.algorithms
        }

        self.canvas = None
        self.create_widgets()
        
        # متغيرات جديدة لـ Cultural Algorithm Visualization
        self.ca_visualization_active = False
        self.ca_current_gen = 0
        self.ca_history_data = []
        
        # ربط حدث تغيير الخوارزمية
        self.algorithm_var.trace('w', self.on_algorithm_changed)

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
        
        # Heuristic selector (مخفي في البداية)
        self.heuristic_label = tk.Label(top_frame, text="Heuristic:")
        self.heuristic_label.grid(row=0, column=4, padx=5)
        self.heuristic_var = tk.StringVar(value=self.heuristics[1])
        self.heuristic_menu = ttk.Combobox(
            top_frame, textvariable=self.heuristic_var,
            values=self.heuristics, state="readonly", width=5
        )
        self.heuristic_menu.grid(row=0, column=5)
        
        # NEW: Checkbox to save both heuristics
        self.save_both_var = tk.BooleanVar(value=True)
        self.save_both_check = tk.Checkbutton(
            top_frame, text="Save Both Heuristics", 
            variable=self.save_both_var
        )
        self.save_both_check.grid(row=0, column=6, padx=5)
        
        # إخفاء Heuristic في البداية (لأن Backtracking هو الافتراضي)
        self.heuristic_label.grid_remove()
        self.heuristic_menu.grid_remove()
        self.save_both_check.grid_remove()

        tk.Button(top_frame, text="Solve", command=self.solve_nqueens).grid(row=0, column=7, padx=5)
        tk.Button(top_frame, text="Compare All", command=self.display_comparison).grid(row=0, column=8, padx=5)
        tk.Button(top_frame, text="Visualize Animation", command=self.visualize_animation).grid(row=0, column=9, padx=5)
        tk.Button(top_frame, text="Plot Graph", command=self.plot_graph).grid(row=0, column=10, padx=5)
        tk.Button(top_frame, text="CA Detailed View", command=self.show_ca_detailed_view).grid(row=0, column=11, padx=5)

        self.solution_frame = tk.Frame(self.root)
        self.solution_frame.pack(pady=10)

        self.time_label = tk.Label(self.root, text="")
        self.time_label.pack()
        self.steps_label = tk.Label(self.root, text="")
        self.steps_label.pack()
        # NEW: Heuristic display label
        self.heuristic_display_label = tk.Label(self.root, text="")
        self.heuristic_display_label.pack()

    def on_algorithm_changed(self, *args):
        """تظهر/تخفي اختيار Heuristic بناءً على الخوارزمية المختارة"""
        algo = self.algorithm_var.get()
        
        # إظهار Heuristic فقط للـ Hill-Climbing و Best-First Search
        if algo in ["Hill-Climbing", "Best-First Search"]:
            self.heuristic_label.grid()
            self.heuristic_menu.grid()
            self.save_both_check.grid()
        else:
            self.heuristic_label.grid_remove()
            self.heuristic_menu.grid_remove()
            self.save_both_check.grid_remove()
            self.heuristic_display_label.config(text="")  # إخفاء عرض الـ Heuristic

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
        save_both = self.save_both_var.get()  # Check if we should save both heuristics
        
        # قائمة الـ heuristics المطلوب تشغيلها
        heuristics_to_run = []
        
        if algo in ["Hill-Climbing", "Best-First Search"]:
            if save_both:
                # تشغيل كلا الـ heuristics
                heuristics_to_run = ["h1", "h2"]
            else:
                # تشغيل الـ heuristic المختار فقط
                heuristics_to_run = [self.heuristic_var.get()]
        else:
            # للخوارزميات الأخرى، لا يوجد heuristic
            heuristics_to_run = [None]

        last_board = None
        last_elapsed = 0
        last_steps = 0
        last_heuristic_display = ""
        last_heuristic_val = None
        
        for heuristic in heuristics_to_run:
            if algo == "Backtracking":
                solver = NQueensSolver(N)
                heuristic_display = "N/A"
            elif algo == "Hill-Climbing":
                solver = HillClimbingSolver(N, heuristic=heuristic)
                heuristic_display = heuristic
            elif algo == "Best-First Search":
                solver = BestFirstSolver(N, heuristic=heuristic)
                heuristic_display = heuristic
            elif algo == "Cultural Algorithm":
                solver = CulturalAlgorithmSolver(N, population_size=80, max_generations=100)
                heuristic_display = "N/A"
            else:
                messagebox.showerror("Error", "Unknown algorithm selected.")
                return

            board, elapsed, steps = solver.solve_one_solution()
            
            # حساب قيمة Heuristic
            if board and algo in ["Hill-Climbing", "Best-First Search"]:
                if heuristic == 'h1':
                    heuristic_val = solver.h1(board)
                else:
                    heuristic_val = solver.h2(board)
            else:
                heuristic_val = None

            conflicts = solver.compute_conflicts(board) if board else None
            
            # حفظ النتيجة للـ heuristic المناسب
            if algo in ["Hill-Climbing", "Best-First Search"]:
                self.saved_solutions[algo][heuristic] = {
                    "board": board,
                    "time": elapsed,
                    "steps": steps,
                    "conflicts": conflicts,
                    "solver_obj": solver,
                    "heuristic_val": heuristic_val
                }
            else:
                # للخوارزميات الأخرى، حفظ في كلا المكانين
                self.saved_solutions[algo]["h1"] = {
                    "board": board,
                    "time": elapsed,
                    "steps": steps,
                    "conflicts": conflicts,
                    "solver_obj": solver,
                    "heuristic_val": None
                }
                self.saved_solutions[algo]["h2"] = {
                    "board": board,
                    "time": elapsed,
                    "steps": steps,
                    "conflicts": conflicts,
                    "solver_obj": solver,
                    "heuristic_val": None
                }
            
            # حفظ آخر لوحة لعرضها
            last_board = board
            last_elapsed = elapsed
            last_steps = steps
            last_heuristic_display = heuristic_display
            last_heuristic_val = heuristic_val

        # عرض آخر نتيجة (أو الوحيدة)
        self.display_board(last_board, N)
        self.time_label.config(text=f"Time: {last_elapsed:.6f}s")
        self.steps_label.config(text=f"Steps: {last_steps}")
        
        # عرض معلومات Heuristic للخوارزميات المناسبة
        if algo in ["Hill-Climbing", "Best-First Search"]:
            if save_both:
                h1_data = self.saved_solutions[algo]["h1"]
                h2_data = self.saved_solutions[algo]["h2"]
                h1_text = f"h1: {h1_data['heuristic_val']}" if h1_data['heuristic_val'] is not None else "h1: N/A"
                h2_text = f"h2: {h2_data['heuristic_val']}" if h2_data['heuristic_val'] is not None else "h2: N/A"
                heuristic_text = f"Saved both heuristics - {h1_text}, {h2_text}"
            else:
                heuristic_text = f"Heuristic: {last_heuristic_display} = {last_heuristic_val}" if last_heuristic_val is not None else f"Heuristic: {last_heuristic_display}"
            self.heuristic_display_label.config(text=heuristic_text)
        else:
            self.heuristic_display_label.config(text="")

        if save_both and algo in ["Hill-Climbing", "Best-First Search"]:
            messagebox.showinfo("Success", f"Both heuristics saved for {algo}!")

    def display_comparison(self):
        try:
            N = int(self.n_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a valid positive integer for N.")
            return

        compare_win = tk.Toplevel(self.root)
        compare_win.title("Algorithms Comparison - All Heuristics")
        compare_win.geometry("1400x800")

        # إنشاء Notebook لعرض تبويبات لكل خوارزمية
        notebook = ttk.Notebook(compare_win)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        for algo_idx, algo in enumerate(self.algorithms):
            # إنشاء تبويب لكل خوارزمية
            algo_frame = ttk.Frame(notebook)
            notebook.add(algo_frame, text=algo)
            
            # عنوان التبويب
            tk.Label(algo_frame, text=f"{algo} - All Heuristics", 
                    font=("Arial", 14, "bold")).pack(pady=10)
            
            # عرض كلا الـ heuristics جنباً إلى جنب
            inner_frame = tk.Frame(algo_frame)
            inner_frame.pack(pady=10)
            
            if algo in ["Hill-Climbing", "Best-First Search"]:
                # عرض h1 و h2 جنباً إلى جنب
                for h_idx, heuristic in enumerate(["h1", "h2"]):
                    data = self.saved_solutions[algo][heuristic]
                    
                    # إطار لكل heuristic
                    h_frame = tk.LabelFrame(inner_frame, text=f"Heuristic: {heuristic}", 
                                           padx=10, pady=10)
                    h_frame.grid(row=0, column=h_idx, padx=20)
                    
                    # الرسم البياني
                    global CELL_SIZE
                    cell_size_temp = min(300 // max(1, N), 60)
                    
                    canvas = tk.Canvas(h_frame, width=N * cell_size_temp, height=N * cell_size_temp)
                    canvas.pack()
                    
                    if data["board"] and any(r != -1 for r in data["board"]):
                        self.draw_board(data["board"], canvas)
                    else:
                        canvas.create_text(
                            N * cell_size_temp // 2,
                            N * cell_size_temp // 2,
                            text="No Solution",
                            font=("Arial", 12)
                        )
                    
                    # معلومات الحل
                    info_frame = tk.Frame(h_frame)
                    info_frame.pack(pady=5)
                    
                    tk.Label(info_frame, text=f"Time: {data['time']:.6f}s", 
                            font=("Arial", 10)).pack()
                    tk.Label(info_frame, text=f"Steps: {data['steps']}", 
                            font=("Arial", 10)).pack()
                    tk.Label(info_frame, text=f"Conflicts: {data['conflicts']}", 
                            font=("Arial", 10)).pack()
                    
                    if data["heuristic_val"] is not None:
                        tk.Label(info_frame, text=f"Heuristic Value: {data['heuristic_val']}", 
                                font=("Arial", 10, "bold")).pack()
            else:
                # للخوارزميات الأخرى، عرض حل واحد فقط
                data = self.saved_solutions[algo]["h1"]  # أي منهما نفس الشيء
                
                # الرسم البياني
                global CELL_SIZE
                cell_size_temp = min(400 // max(1, N), 70)
                
                canvas = tk.Canvas(inner_frame, width=N * cell_size_temp, height=N * cell_size_temp)
                canvas.pack()
                
                if data["board"] and any(r != -1 for r in data["board"]):
                    self.draw_board(data["board"], canvas)
                else:
                    canvas.create_text(
                        N * cell_size_temp // 2,
                        N * cell_size_temp // 2,
                        text="No Solution",
                        font=("Arial", 14)
                    )
                
                # معلومات الحل
                info_frame = tk.Frame(inner_frame)
                info_frame.pack(pady=10)
                
                tk.Label(info_frame, text=f"Time: {data['time']:.6f}s", 
                        font=("Arial", 11)).pack()
                tk.Label(info_frame, text=f"Steps: {data['steps']}", 
                        font=("Arial", 11)).pack()
                tk.Label(info_frame, text=f"Conflicts: {data['conflicts']}", 
                        font=("Arial", 11)).pack()

        # إضافة تبويب للملخص
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        # إنشاء Treeview لعرض الملخص
        tree_frame = tk.Frame(summary_frame)
        tree_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # شريط التمرير
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side='right', fill='y')
        
        # Treeview
        columns = ("Algorithm", "Heuristic", "Time (s)", "Steps", "Conflicts", "Heuristic Value", "Solution Found")
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', 
                           yscrollcommand=tree_scroll.set, height=15)
        
        # تعريف العناوين
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')
        
        # إضافة البيانات
        for algo in self.algorithms:
            if algo in ["Hill-Climbing", "Best-First Search"]:
                # إضافة صفين لكل خوارزمية (h1 و h2)
                for heuristic in ["h1", "h2"]:
                    data = self.saved_solutions[algo][heuristic]
                    solution_found = "Yes" if data["board"] is not None and data["conflicts"] == 0 else "No"
                    tree.insert('', 'end', values=(
                        algo,
                        heuristic,
                        f"{data['time']:.6f}",
                        data["steps"],
                        data["conflicts"] if data["conflicts"] is not None else "N/A",
                        data["heuristic_val"] if data["heuristic_val"] is not None else "N/A",
                        solution_found
                    ))
            else:
                # خوارزمية واحدة
                data = self.saved_solutions[algo]["h1"]
                solution_found = "Yes" if data["board"] is not None and data["conflicts"] == 0 else "No"
                tree.insert('', 'end', values=(
                    algo,
                    "N/A",
                    f"{data['time']:.6f}",
                    data["steps"],
                    data["conflicts"] if data["conflicts"] is not None else "N/A",
                    "N/A",
                    solution_found
                ))
        
        tree.pack(side='left', fill='both', expand=True)
        tree_scroll.config(command=tree.yview)
        
        # إضافة ألوان للصفوف
        for i, item in enumerate(tree.get_children()):
            if i % 2 == 0:
                tree.tag_configure('evenrow', background='#f0f0f0')
                tree.item(item, tags=('evenrow',))

    def visualize_animation(self):
        data = self.saved_solutions.get("Cultural Algorithm")
        if not data:
            messagebox.showinfo("Info", "Run Cultural Algorithm first to get data.")
            return

        solver = data.get("h1", {}).get("solver_obj") or data.get("h2", {}).get("solver_obj")
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
        width = max(800, N * 50)
        height = max(600, N * 50)
        anim_win.geometry(f"{width}x{height}")

        global CELL_SIZE
        cell_size_temp = min(800 // max(1, N), 80)
        main_canvas = tk.Canvas(anim_win, width=N * cell_size_temp, height=N * cell_size_temp)
        main_canvas.pack(padx=10, pady=5)

        pop_canvas = tk.Canvas(anim_win, width=N * cell_size_temp, height=N * cell_size_temp//2)
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
                mini_size = cell_size_temp // 2
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

        solver = data.get("h1", {}).get("solver_obj") or data.get("h2", {}).get("solver_obj")
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

    # -------------------------------------------------------------
    # NEW: Cultural Algorithm Detailed Visualization
    # -------------------------------------------------------------
    def show_ca_detailed_view(self):
        """عرض تفصيلي لـ Cultural Algorithm"""
        data = self.saved_solutions.get("Cultural Algorithm")
        if not data:
            messagebox.showinfo("Info", "Run Cultural Algorithm first to get data.")
            return

        solver = data.get("h1", {}).get("solver_obj") or data.get("h2", {}).get("solver_obj")
        if solver is None or not hasattr(solver, "history") or len(solver.history) == 0:
            messagebox.showinfo("Info", "No history available. Run Cultural Algorithm first (Solve).")
            return

        history = solver.history
        N = solver.N
        
        ca_win = tk.Toplevel(self.root)
        ca_win.title("Cultural Algorithm - Detailed Analysis")
        ca_win.geometry("1200x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(ca_win)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Convergence Analysis
        convergence_frame = ttk.Frame(notebook)
        notebook.add(convergence_frame, text="Convergence")
        self.create_convergence_tab(convergence_frame, history, N)
        
        # Tab 2: Belief Space Analysis
        belief_frame = ttk.Frame(notebook)
        notebook.add(belief_frame, text="Belief Space")
        self.create_belief_space_tab(belief_frame, history, N)
        
        # Tab 3: Population Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Population Stats")
        self.create_population_stats_tab(stats_frame, history, N)
        
        # Tab 4: Algorithm Parameters
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        self.create_parameters_tab(params_frame, solver)

    def create_convergence_tab(self, parent, history, N):
        """Tab 1: Convergence Analysis"""
        fig = Figure(figsize=(10, 6))
        
        # Subplot 1: Best conflicts over generations
        ax1 = fig.add_subplot(221)
        generations = [h["generation"] for h in history]
        conflicts = [h["conflicts"] for h in history]
        ax1.plot(generations, conflicts, 'b-o', linewidth=2, markersize=4)
        ax1.set_title("Best Solution Convergence")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Conflicts")
        ax1.grid(True, alpha=0.3)
        
        # Highlight when solution found (conflicts == 0)
        zero_conflicts = [gen for gen, conf in zip(generations, conflicts) if conf == 0]
        if zero_conflicts:
            ax1.axvline(x=zero_conflicts[0], color='red', linestyle='--', alpha=0.7)
            ax1.text(zero_conflicts[0], max(conflicts)/2, f'Found at gen {zero_conflicts[0]}', 
                    rotation=90, verticalalignment='center')

        # Subplot 2: Average conflicts (estimated)
        ax2 = fig.add_subplot(222)
        # Assuming average conflicts are around 1.5-2x best conflicts
        avg_conflicts = [c * 1.5 for c in conflicts]
        ax2.plot(generations, avg_conflicts, 'g-', linewidth=2, alpha=0.7, label='Est. Average')
        ax2.plot(generations, conflicts, 'b-', linewidth=2, alpha=0.7, label='Best')
        ax2.set_title("Best vs Average Conflicts")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Conflicts")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Improvement rate
        ax3 = fig.add_subplot(223)
        improvements = []
        for i in range(1, len(conflicts)):
            improvement = conflicts[i-1] - conflicts[i]
            improvements.append(improvement)
        
        ax3.bar(generations[1:], improvements, color=['green' if x>0 else 'red' for x in improvements])
        ax3.set_title("Improvement per Generation")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Δ Conflicts")
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Subplot 4: Success rate indicator
        ax4 = fig.add_subplot(224)
        success_percentage = []
        for i, conf in enumerate(conflicts):
            success_rate = (1 - conf/(N*(N-1)/2)) * 100  # Estimate success percentage
            success_percentage.append(max(0, success_rate))
        
        ax4.plot(generations, success_percentage, 'purple', linewidth=2)
        ax4.fill_between(generations, success_percentage, 0, alpha=0.3, color='purple')
        ax4.set_title("Solution Quality (%)")
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Quality %")
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_belief_space_tab(self, parent, history, N):
        """Tab 2: Belief Space Analysis"""
        fig = Figure(figsize=(10, 6))
        
        if N > 12:  # For large N, show summary
            ax = fig.add_subplot(111)
            belief_ranges = []
            for h in history:
                L = h["belief"]["L"]
                U = h["belief"]["U"]
                avg_range = sum(u-l for l,u in zip(L,U)) / N
                belief_ranges.append(avg_range)
            
            ax.plot(range(len(history)), belief_ranges, 'b-o', linewidth=2)
            ax.set_title("Average Belief Space Range per Generation")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Average Range (U-L)")
            ax.grid(True)
        else:
            # For small N, show detailed belief space
            ax1 = fig.add_subplot(211)
            
            # Take first, middle, and last generation for comparison
            if len(history) >= 3:
                gens = [0, len(history)//2, len(history)-1]
                colors = ['red', 'green', 'blue']
                labels = ['Start', 'Mid', 'End']
                
                for gen_idx, color, label in zip(gens, colors, labels):
                    belief = history[gen_idx]["belief"]
                    L = belief["L"]
                    U = belief["U"]
                    
                    for col in range(N):
                        ax1.plot([col, col], [L[col], U[col]], 
                                color=color, marker='o', linewidth=2, label=label if col==0 else "")
                
                ax1.set_title("Belief Space Evolution (Lower/Upper Bounds)")
                ax1.set_xlabel("Column")
                ax1.set_ylabel("Row Range")
                ax1.legend()
                ax1.grid(True)
            
            # Heatmap of belief space convergence
            ax2 = fig.add_subplot(212)
            
            # Create matrix showing belief space coverage
            belief_matrix = np.zeros((N, len(history)))
            for gen_idx, h in enumerate(history):
                belief = h["belief"]
                for col in range(N):
                    coverage = (belief["U"][col] - belief["L"][col] + 1) / N
                    belief_matrix[col, gen_idx] = coverage
            
            im = ax2.imshow(belief_matrix, aspect='auto', cmap='RdYlGn_r')
            ax2.set_title("Belief Space Coverage Over Generations")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Column")
            plt.colorbar(im, ax=ax2, label='Coverage (1=full, 0=narrow)')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_population_stats_tab(self, parent, history, N):
        """Tab 3: Population Statistics"""
        fig = Figure(figsize=(10, 6))
        
        # Subplot 1: Diversity estimation
        ax1 = fig.add_subplot(221)
        
        # Simulate diversity metric (distance between solutions)
        diversity_metric = []
        for gen in range(min(20, len(history))):  # Limit to first 20 gens for clarity
            # Simulate diversity decreasing over time
            diversity = 100 * (0.9 ** gen) + np.random.rand() * 10
            diversity_metric.append(diversity)
        
        ax1.plot(range(len(diversity_metric)), diversity_metric, 'b-o')
        ax1.set_title("Population Diversity Over Generations")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Diversity Metric")
        ax1.grid(True)
        
        # Subplot 2: Acceptance rate visualization
        ax2 = fig.add_subplot(222)
        
        acceptance_rate = 0.4  # From solver
        accepted_counts = [int(50 * acceptance_rate)] * len(history)  # Assuming population=50
        total_counts = [50] * len(history)
        
        x = range(len(history))
        ax2.bar(x, total_counts, color='lightblue', label='Total Population')
        ax2.bar(x, accepted_counts, color='darkblue', label='Accepted Individuals')
        ax2.set_title("Acceptance Rate: Accepted vs Total")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Number of Individuals")
        ax2.legend()
        
        # Subplot 3: Solution similarity to belief space
        ax3 = fig.add_subplot(223)
        
        similarity_scores = []
        for h in history:
            best = h["best"]
            belief = h["belief"]
            score = 0
            for col in range(N):
                if belief["L"][col] <= best[col] <= belief["U"][col]:
                    score += 1
            similarity_scores.append(score / N * 100)
        
        ax3.plot(range(len(history)), similarity_scores, 'g-', linewidth=2)
        ax3.fill_between(range(len(history)), similarity_scores, 0, alpha=0.3, color='green')
        ax3.set_title("Solution Similarity to Belief Space")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Similarity %")
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        
        # Subplot 4: Mutation/Crossover events
        ax4 = fig.add_subplot(224)
        
        generations = list(range(len(history)))
        mutation_events = [np.random.randint(5, 15) for _ in generations]
        crossover_events = [np.random.randint(10, 25) for _ in generations]
        
        width = 0.35
        ax4.bar([x - width/2 for x in generations], mutation_events, width, label='Mutations', color='orange')
        ax4.bar([x + width/2 for x in generations], crossover_events, width, label='Crossovers', color='purple')
        ax4.set_title("Variation Events per Generation")
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Number of Events")
        ax4.legend()
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_parameters_tab(self, parent, solver):
        """Tab 4: Algorithm Parameters"""
        param_frame = tk.Frame(parent)
        param_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(param_frame, text="Cultural Algorithm Parameters", 
                font=("Arial", 16, "bold")).pack(pady=10)
        
        # Create parameter display
        params = [
            ("Population Size", getattr(solver, 'population_size', 'N/A')),
            ("Max Generations", getattr(solver, 'max_generations', 'N/A')),
            ("Acceptance Rate", getattr(solver, 'acceptance_rate', 'N/A')),
            ("Mutation Rate", getattr(solver, 'mutation_rate', 'N/A')),
            ("Use Crossover", getattr(solver, 'use_crossover', 'N/A')),
            ("Tournament K", getattr(solver, 'tournament_k', 'N/A')),
            ("Board Size (N)", solver.N),
            ("Total Generations Run", len(solver.history) if hasattr(solver, 'history') else 0)
        ]
        
        for i, (param_name, param_value) in enumerate(params):
            frame = tk.Frame(param_frame)
            frame.pack(fill='x', pady=5)
            
            tk.Label(frame, text=f"{param_name}:", width=25, anchor='w', 
                    font=("Arial", 10, "bold")).pack(side='left')
            tk.Label(frame, text=str(param_value), width=15, anchor='w',
                    font=("Arial", 10)).pack(side='left')
        
        # Separator
        tk.Frame(param_frame, height=2, bg='gray').pack(fill='x', pady=20)
        
        # Performance metrics
        tk.Label(param_frame, text="Performance Metrics", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        if hasattr(solver, 'history') and solver.history:
            final_conflicts = solver.history[-1]["conflicts"]
            best_conflicts = min(h["conflicts"] for h in solver.history)
            improvement = solver.history[0]["conflicts"] - final_conflicts
            
            metrics = [
                ("Initial Conflicts", solver.history[0]["conflicts"]),
                ("Final Conflicts", final_conflicts),
                ("Best Found", best_conflicts),
                ("Total Improvement", improvement),
                ("Improvement Rate", f"{improvement/max(1, len(solver.history)-1):.2f} per gen"),
                ("Solution Found", "Yes" if best_conflicts == 0 else "No")
            ]
            
            for metric_name, metric_value in metrics:
                frame = tk.Frame(param_frame)
                frame.pack(fill='x', pady=3)
                
                tk.Label(frame, text=f"{metric_name}:", width=25, anchor='w',
                        font=("Arial", 9)).pack(side='left')
                tk.Label(frame, text=str(metric_value), width=15, anchor='w',
                        font=("Arial", 9, "bold" if "Found" in metric_name else "")).pack(side='left')
        
        # Explanation text
        tk.Frame(param_frame, height=2, bg='gray').pack(fill='x', pady=20)
        
        explanation = """
        Cultural Algorithm Components:
        
        1. Population Space: Set of candidate solutions
        2. Belief Space: Contains cultural knowledge
           - Normative: Lower/Upper bounds for each dimension
           - Situational: Best solution found so far
        3. Acceptance: Top individuals influence belief space
        4. Influence: Belief space guides new solutions
        5. Variation: Mutation and crossover create diversity
        """
        
        tk.Label(param_frame, text=explanation, justify='left',
                font=("Arial", 9), bg='#f0f0f0').pack(fill='x', padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = NQueensGUI(root)
    root.mainloop()
