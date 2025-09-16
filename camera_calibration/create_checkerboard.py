# make_checkerboard_simple.py
# Create a printable checkerboard for OpenCV calibration.
# Edit the constants below, then run:  python create_checkerboard.py

from PIL import Image, ImageDraw

# -------------------------
# User settings
# -------------------------
ROWS  = 5          # rows 
COLS  = 5          # cols 
SQUARE_MM   = 10.0       # size of one square (mm)
BORDER_MM   = 0.0       # white margin around the board (mm)
DPI         = 300        # 300–600 recommended
OUT_PATH    = f"checker_{ROWS}x{COLS}_{int(SQUARE_MM)}mm_{DPI}dpi.png"

# -------------------------
# Implementation
# -------------------------
MM_PER_INCH = 25.4

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / MM_PER_INCH))

def make_checkerboard(rows, cols, square_mm, border_mm, dpi, out_path):

    sq_px     = mm_to_px(square_mm, dpi)
    border_px = mm_to_px(border_mm, dpi)

    board_w_px = cols * sq_px
    board_h_px = rows * sq_px
    img_w_px   = board_w_px + 2 * border_px
    img_h_px   = board_h_px + 2 * border_px

    # White canvas
    img = Image.new("RGB", (img_w_px, img_h_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    x0 = border_px
    y0 = border_px
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Draw squares (top-left is black)
    for r in range(rows):
        for c in range(cols):
            x1 = x0 + c * sq_px
            y1 = y0 + r * sq_px
            x2 = x1 + sq_px
            y2 = y1 + sq_px
            color = black if ((r + c) % 2 == 0) else white
            draw.rectangle([x1, y1, x2, y2], fill=color)

    img.save(out_path, dpi=(dpi, dpi))
    print(f"Saved: {out_path}")
    print(f"- Print at 100% scale (no 'fit to page').")
    print(f"- Each square should measure ~{square_mm} mm.")
    print(f"- Use cv2.findChessboardCorners(gray, ({cols}, {rows}))")

if __name__ == "__main__":
    make_checkerboard(
        ROWS, COLS,
        SQUARE_MM, BORDER_MM,
        DPI, OUT_PATH
    )
