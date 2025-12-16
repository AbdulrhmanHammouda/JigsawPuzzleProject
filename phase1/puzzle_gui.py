"""
Puzzle Solver GUI
=================
Visual demonstration of 2x2, 4x4, and 8x8 puzzle solvers.
Shows shuffled pieces alongside assembled results.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np

# Paths
BASE = r"c:\Users\Lenovo\Desktop\semester 5\image\project\phase1\results"
GT_DIR = r"c:\Users\Lenovo\Desktop\semester 5\image\project\data\correct"

PUZZLE_DIRS = {
    "2x2": os.path.join(BASE, "2x2_out"),
    "4x4": os.path.join(BASE, "4x4_out"),
    "8x8": os.path.join(BASE, "solver_8x8_ensemble_beta"),
}

SLICED_DIRS = {
    "2x2": os.path.join(BASE, "enhanced_images_sliced", "puzzle_2x2"),
    "4x4": os.path.join(BASE, "enhanced_images_sliced", "puzzle_4x4"),
    "8x8": os.path.join(BASE, "enhanced_images_sliced", "puzzle_8x8"),
}


class PuzzleSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 Jigsaw Puzzle Solver Demo")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1a1a2e")
        
        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11), padding=10)
        style.configure("TLabel", background="#1a1a2e", foreground="white", font=("Segoe UI", 11))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#00d9ff")
        
        self.current_size = "2x2"
        self.current_idx = 0
        self.puzzle_list = []
        
        self.create_widgets()
        self.load_puzzle_list()
        self.show_puzzle()
    
    def create_widgets(self):
        # Title
        title = ttk.Label(self.root, text="🧩 Jigsaw Puzzle Solver", style="Title.TLabel")
        title.pack(pady=15)
        
        # Size selector
        size_frame = tk.Frame(self.root, bg="#1a1a2e")
        size_frame.pack(pady=10)
        
        ttk.Label(size_frame, text="Puzzle Size:").pack(side=tk.LEFT, padx=5)
        
        self.size_var = tk.StringVar(value="2x2")
        for size in ["2x2", "4x4", "8x8"]:
            btn = tk.Radiobutton(
                size_frame, text=size, variable=self.size_var, value=size,
                command=self.on_size_change,
                bg="#1a1a2e", fg="white", selectcolor="#16213e",
                activebackground="#1a1a2e", activeforeground="#00d9ff",
                font=("Segoe UI", 12, "bold")
            )
            btn.pack(side=tk.LEFT, padx=10)
        
        # Image display area
        img_frame = tk.Frame(self.root, bg="#16213e", relief=tk.RIDGE, bd=2)
        img_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        # Three columns: Pieces | Assembled | Ground Truth
        self.pieces_canvas = tk.Canvas(img_frame, width=350, height=350, bg="#0f0f1a", highlightthickness=0)
        self.pieces_canvas.pack(side=tk.LEFT, padx=15, pady=15)
        
        self.assembled_canvas = tk.Canvas(img_frame, width=350, height=350, bg="#0f0f1a", highlightthickness=0)
        self.assembled_canvas.pack(side=tk.LEFT, padx=15, pady=15)
        
        self.gt_canvas = tk.Canvas(img_frame, width=350, height=350, bg="#0f0f1a", highlightthickness=0)
        self.gt_canvas.pack(side=tk.LEFT, padx=15, pady=15)
        
        # Labels
        label_frame = tk.Frame(self.root, bg="#1a1a2e")
        label_frame.pack()
        
        ttk.Label(label_frame, text="Shuffled Pieces").pack(side=tk.LEFT, padx=80)
        ttk.Label(label_frame, text="Assembled Result").pack(side=tk.LEFT, padx=80)
        ttk.Label(label_frame, text="Ground Truth").pack(side=tk.LEFT, padx=80)
        
        # Navigation
        nav_frame = tk.Frame(self.root, bg="#1a1a2e")
        nav_frame.pack(pady=15)
        
        self.prev_btn = tk.Button(
            nav_frame, text="◀ Previous", command=self.prev_puzzle,
            font=("Segoe UI", 11), bg="#16213e", fg="white",
            activebackground="#00d9ff", padx=20, pady=8
        )
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        
        self.idx_label = ttk.Label(nav_frame, text="1 / 110")
        self.idx_label.pack(side=tk.LEFT, padx=20)
        
        self.next_btn = tk.Button(
            nav_frame, text="Next ▶", command=self.next_puzzle,
            font=("Segoe UI", 11), bg="#16213e", fg="white",
            activebackground="#00d9ff", padx=20, pady=8
        )
        self.next_btn.pack(side=tk.LEFT, padx=10)
        
        # Status
        self.status_label = ttk.Label(self.root, text="", font=("Segoe UI", 12, "bold"))
        self.status_label.pack(pady=10)
    
    def load_puzzle_list(self):
        out_dir = PUZZLE_DIRS.get(self.current_size, "")
        if os.path.exists(out_dir):
            files = [f.replace("_assembled.png", "") for f in os.listdir(out_dir) if f.endswith(".png")]
            self.puzzle_list = sorted(files, key=lambda x: int(x) if x.isdigit() else x)
        else:
            self.puzzle_list = [str(i) for i in range(110)]
        self.current_idx = 0
    
    def on_size_change(self):
        self.current_size = self.size_var.get()
        self.load_puzzle_list()
        self.show_puzzle()
    
    def show_puzzle(self):
        if not self.puzzle_list:
            return
        
        puzzle_id = self.puzzle_list[self.current_idx]
        
        # Update index label
        self.idx_label.config(text=f"{self.current_idx + 1} / {len(self.puzzle_list)}")
        
        # Load and display images
        self.display_pieces(puzzle_id)
        self.display_assembled(puzzle_id)
        self.display_gt(puzzle_id)
    
    def display_pieces(self, puzzle_id):
        """Show shuffled pieces as grid"""
        sliced_dir = os.path.join(SLICED_DIRS[self.current_size], puzzle_id)
        if not os.path.exists(sliced_dir):
            self.pieces_canvas.delete("all")
            self.pieces_canvas.create_text(175, 175, text="No pieces", fill="gray", font=("Segoe UI", 14))
            return
        
        # Load pieces
        files = sorted(os.listdir(sliced_dir))[:64]
        pieces = [cv2.imread(os.path.join(sliced_dir, f)) for f in files if f.endswith(('.png', '.jpg'))]
        pieces = [p for p in pieces if p is not None]
        
        if not pieces:
            return
        
        # Create grid image
        grid_size = int(self.current_size.split('x')[0])
        ph, pw = pieces[0].shape[:2]
        grid_img = np.zeros((grid_size * ph, grid_size * pw, 3), dtype=np.uint8)
        
        for i, piece in enumerate(pieces[:grid_size*grid_size]):
            r, c = i // grid_size, i % grid_size
            grid_img[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = piece
        
        # Convert and display
        self.show_cv_image(self.pieces_canvas, grid_img)
    
    def display_assembled(self, puzzle_id):
        """Show assembled result"""
        out_dir = PUZZLE_DIRS[self.current_size]
        img_path = os.path.join(out_dir, f"{puzzle_id}_assembled.png")
        
        if not os.path.exists(img_path):
            self.assembled_canvas.delete("all")
            self.assembled_canvas.create_text(175, 175, text="Not assembled", fill="gray", font=("Segoe UI", 14))
            return
        
        img = cv2.imread(img_path)
        self.show_cv_image(self.assembled_canvas, img)
        
        # Check accuracy
        gt_path = os.path.join(GT_DIR, f"{puzzle_id}.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(GT_DIR, f"{puzzle_id}.jpg")
        
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path)
            if gt.shape != img.shape:
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]))
            mse = np.mean((img.astype(float) - gt.astype(float)) ** 2)
            
            if mse < 300:
                status = "✅ PASS"
                color = "#00ff88"
            elif mse < 1000:
                status = "⚠️ CLOSE"
                color = "#ffcc00"
            else:
                status = "❌ FAIL"
                color = "#ff4444"
            
            self.status_label.config(text=f"Puzzle #{puzzle_id}: {status} (MSE: {mse:.1f})", foreground=color)
    
    def display_gt(self, puzzle_id):
        """Show ground truth"""
        gt_path = os.path.join(GT_DIR, f"{puzzle_id}.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(GT_DIR, f"{puzzle_id}.jpg")
        
        if not os.path.exists(gt_path):
            self.gt_canvas.delete("all")
            self.gt_canvas.create_text(175, 175, text="No GT", fill="gray", font=("Segoe UI", 14))
            return
        
        img = cv2.imread(gt_path)
        self.show_cv_image(self.gt_canvas, img)
    
    def show_cv_image(self, canvas, cv_img):
        """Display OpenCV image on canvas"""
        # Resize to fit canvas
        h, w = cv_img.shape[:2]
        scale = min(340 / w, 340 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and then to PhotoImage
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)
        
        # Display
        canvas.delete("all")
        canvas.create_image(175, 175, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep reference
    
    def prev_puzzle(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_puzzle()
    
    def next_puzzle(self):
        if self.current_idx < len(self.puzzle_list) - 1:
            self.current_idx += 1
            self.show_puzzle()


def main():
    root = tk.Tk()
    app = PuzzleSolverGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
