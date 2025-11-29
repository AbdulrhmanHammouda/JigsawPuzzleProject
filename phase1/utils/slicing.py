import cv2
import os

def slice_grid(img, save_dir, grid_size):
    """
    Slice image into grid_size × grid_size tiles.
    Save each tile to save_dir.
    Returns number of saved pieces.
    """
    h, w = img.shape[:2]
    piece_h = h // grid_size
    piece_w = w // grid_size

    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * piece_h
            y2 = h if row == grid_size - 1 else (row + 1) * piece_h
            x1 = col * piece_w
            x2 = w if col == grid_size - 1 else (col + 1) * piece_w

            piece = img[y1:y2, x1:x2]
            os.makedirs(save_dir, exist_ok=True)
            piece_path = os.path.join(save_dir, f"piece_{index:03d}.png")
            cv2.imwrite(piece_path, piece)
            index += 1

    return index
