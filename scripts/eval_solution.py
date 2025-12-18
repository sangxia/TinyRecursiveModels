import torch
from tqdm import tqdm
from pathlib import Path
import argparse
from PIL import Image, ImageDraw

SYMBOL_COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}

def render_grid(grid, out_path, cell_size, margin, bg_color, line_color, line_width):
    global SYMBOL_COLORS
    rows = len(grid)
    cols = len(grid[0])
    width = cols * cell_size + 2 * margin
    height = rows * cell_size + 2 * margin
    # draw background and grid lines
    img = Image.new("RGBA", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([margin, margin, width - margin, height - margin], fill=line_color)
    # draw cells
    for r in range(rows):
        for c in range(cols):
            color = SYMBOL_COLORS[grid[r][c] - 2]
            x0 = margin + c * cell_size + line_width
            y0 = margin + r * cell_size + line_width
            x1 = x0 + cell_size - line_width * 2
            y1 = y0 + cell_size - line_width * 2
            draw.rectangle([x0, y0, x1, y1], fill=color)
    img.save(out_path, format="PNG")

def eval_sample(label, pred):
    label = label.view(30, 30)
    row_range = torch.where(label[:, 0] >= 0)[0].max().item()
    col_range = torch.where(label[0, :] >= 0)[0].max().item()
    label = label[:row_range, :col_range]
    pred = pred.view(30, 30)
    pred = pred[:row_range, :col_range].to(torch.int32)
    return torch.allclose(pred, label), label.numpy().tolist(), pred.numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('saved', type=Path)
    parser.add_argument('--outimg', type=Path)
    args = parser.parse_args()

    data = torch.load(args.saved)
    # NOTE not aggregating augmentations here
    cnt = 0
    corr = 0
    for lbl, pred, pid in tqdm(zip(data['labels'], data['preds'], data['puzzle_identifiers'])):
        if pid == 0:
            break
        cnt += 1
        result, lbl_grid, pred_grid = eval_sample(lbl, pred)
        if result and args.outimg:
            render_grid(lbl_grid, args.outimg, 50, 10, '#000000', '#808080', 2)
            break
        corr += result
    print(corr, cnt, corr / cnt)

