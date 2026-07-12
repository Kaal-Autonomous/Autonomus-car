#!/usr/bin/env python3
"""
PilotNet trainer (single file)
==============================
Works directly on the folders produced by the car server:

    dataset/
      20260701_181530/
        labels.csv            index,timestamp,steering,throttle,left,center,right
        left/000000.jpg ...
        center/000000.jpg ...
        right/000000.jpg ...
      20260701_190212/
        ...

What it does, in order:
  1. REFINE  - scans every session under --data, merges them, drops bad rows
               (missing images, throttle ~ 0 i.e. car standing still),
               applies the NVIDIA 3-camera trick (left img -> steer + OFFSET,
               right img -> steer - OFFSET), balances the steering histogram
               (otherwise "drive straight" dominates and the model learns to
               output 0), and writes the merged index to dataset/merged_index.csv
               so you can inspect exactly what it trains on.
  2. TRAIN   - PilotNet CNN (Bojarski et al. 2016), MSE on steering,
               Adam + AMP on CUDA, 90/10 split, saves best-val checkpoint
               to pilotnet.pt together with the preprocessing config so the
               drive script always preprocesses identically.

Install:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python numpy

Run:
    python train_pilotnet.py --data dataset --epochs 30

Cropping - DEFAULT IS RAW (no crop):
    Full 320x240 frame is used, only resized to PilotNet's fixed 200x66
    input (that resize is unavoidable - the network input size is fixed).
    This keeps train/drive alignment simple: nothing is cut, so a bumped
    camera or different mounting can never crop into the track.

    python train_pilotnet.py --preview 8            # saves dataset/preview_preprocessing.jpg
        -> LEFT: raw frame (red lines only if cropping). RIGHT: what the net sees.
    python train_pilotnet.py --crop-top 40 --crop-bottom 0   # OPTIONAL, only if
        the top band is provably distracting the model (test raw first).

    Whatever setting you train with is saved inside pilotnet.pt and the
    drive script applies it automatically - no change needed there.
"""

import os
import csv
import glob
import random
import argparse
from collections import defaultdict

import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# tqdm gives a live per-batch progress bar so you can see training is alive
# (fallback to a no-op wrapper if tqdm isn't installed)
try:
    from tqdm import tqdm
except ImportError:
    print("tip: pip install tqdm  -> for live progress bars")
    def tqdm(x, **kw): return x

# ---------------- config (saved into the checkpoint) ----------------------
CFG = {
    "img_w": 200,          # network input width
    "img_h": 150,          # network input height - 200x150 keeps the camera's
                           # 4:3 shape EXACTLY: no squash, no distortion, just
                           # a clean downscale of the raw 320x240 frame.
                           # (classic PilotNet used 200x66, which crushes a 4:3
                           #  frame vertically; fine with a road-band crop, bad
                           #  for RAW full frames.)
    "crop_top": 0,         # RAW mode: no crop
    "crop_bottom": 0,      # RAW mode: no crop
    "side_cam_offset": 0.18,   # steering correction for left/right cameras
    "min_throttle": 0.05,  # drop samples where the car was basically parked
    "yuv": True,           # YUV colour space (as in the PilotNet paper)
}
# --------------------------------------------------------------------------


# ========================== 1. REFINE THE DATASET ==========================

def load_all_sessions(data_dir):
    """Merge every labels.csv under data_dir into one list of raw rows."""
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(data_dir, "*", "labels.csv"))):
        sdir = os.path.dirname(csv_path)
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "steering": float(r["steering"]),
                    "throttle": float(r["throttle"]),
                    "left":   os.path.join(sdir, r["left"]),
                    "center": os.path.join(sdir, r["center"]),
                    "right":  os.path.join(sdir, r["right"]),
                })
        print(f"  loaded {csv_path}")
    return rows


def build_samples(rows):
    """
    One CSV row becomes THREE training samples (the NVIDIA trick):
        center image -> steering
        left   image -> steering + OFFSET   (teaches 'steer right when too far left')
        right  image -> steering - OFFSET
    Rows where the car was not moving are dropped: steering while parked
    is meaningless noise.
    """
    off = CFG["side_cam_offset"]
    samples, dropped_still, dropped_missing = [], 0, 0
    for r in rows:
        if abs(r["throttle"]) < CFG["min_throttle"]:
            dropped_still += 1
            continue
        for cam, corr in (("center", 0.0), ("left", +off), ("right", -off)):
            p = r[cam]
            if not os.path.isfile(p):
                dropped_missing += 1
                continue
            s = float(np.clip(r["steering"] + corr, -1.0, 1.0))
            samples.append((p, s))
    print(f"  rows dropped (car still): {dropped_still}, missing images: {dropped_missing}")
    return samples


def balance(samples, n_bins=25, cap_factor=2.5):
    """
    Real driving is ~90% 'go straight', so the steering histogram has a huge
    spike at 0 and the network learns to always predict 0. Cap every bin at
    cap_factor * median bin count and randomly downsample the spike.
    """
    bins = defaultdict(list)
    for s in samples:
        b = min(n_bins - 1, int((s[1] + 1.0) / 2.0 * n_bins))
        bins[b].append(s)
    counts = sorted(len(v) for v in bins.values())
    cap = max(50, int(counts[len(counts) // 2] * cap_factor))
    out = []
    for v in bins.values():
        random.shuffle(v)
        out.extend(v[:cap])
    random.shuffle(out)
    print(f"  balance: {len(samples)} -> {len(out)} samples (bin cap {cap})")
    return out


def write_merged_index(samples, data_dir):
    path = os.path.join(data_dir, "merged_index.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "steering"])
        w.writerows(samples)
    print(f"  refined dataset index -> {path}")


# ========================== 2. PREPROCESS + AUGMENT ========================

def preprocess(img_bgr):
    """Crop -> resize to 200x66 -> YUV -> [-1, 1] float CHW. MUST match drive script."""
    h = img_bgr.shape[0]
    img = img_bgr[CFG["crop_top"]: h - CFG["crop_bottom"], :, :]
    img = cv2.resize(img, (CFG["img_w"], CFG["img_h"]), interpolation=cv2.INTER_AREA)
    if CFG["yuv"]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img.astype(np.float32) / 127.5 - 1.0
    return np.transpose(img, (2, 0, 1))          # HWC -> CHW


def augment(img_bgr, steering):
    """Applied only to training split, on the raw 320x240 frame."""
    # 1) horizontal flip: mirrored road, negated steering (free 2x data)
    if random.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
        steering = -steering
    # 2) random brightness: robustness to lighting
    if random.random() < 0.7:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.55, 1.45), 0, 255)
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 3) small horizontal shift with steering correction (~recovery data)
    if random.random() < 0.5:
        shift = random.randint(-20, 20)
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        img_bgr = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]),
                                 borderMode=cv2.BORDER_REPLICATE)
        steering = float(np.clip(steering + shift * 0.004, -1.0, 1.0))
    return img_bgr, steering


def save_preview(samples, data_dir, n):
    """Save a side-by-side image: raw frame with red crop lines (left) vs the
    exact 200x66 tensor the network receives, upscaled for viewing (right).
    Open dataset/preview_preprocessing.jpg and VERIFY the crop suits your
    camera angle before training for real."""
    picked = random.sample(samples, min(n, len(samples)))
    strips = []
    for path, steer in picked:
        raw = cv2.imread(path)
        if raw is None:
            continue
        h, w = raw.shape[:2]
        vis = raw.copy()
        y1, y2 = CFG["crop_top"], h - CFG["crop_bottom"] - 1
        cv2.line(vis, (0, y1), (w, y1), (0, 0, 255), 2)
        cv2.line(vis, (0, y2), (w, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"steer {steer:+.2f}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        net = preprocess(raw)                                   # CHW in [-1,1]
        net_img = ((np.transpose(net, (1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
        if CFG["yuv"]:
            net_img = cv2.cvtColor(net_img, cv2.COLOR_YUV2BGR)  # back to BGR just for viewing
        nh = int(w * CFG["img_h"] / CFG["img_w"])
        net_img = cv2.resize(net_img, (w, nh), interpolation=cv2.INTER_NEAREST)
        right = np.zeros((h, w, 3), np.uint8)
        right[(h - nh) // 2:(h - nh) // 2 + nh] = net_img
        cv2.putText(right, "network input 200x66", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        strips.append(np.hstack([vis, right]))
    if strips:
        out_path = os.path.join(data_dir, "preview_preprocessing.jpg")
        cv2.imwrite(out_path, np.vstack(strips))
        print(f"  preprocessing preview -> {out_path}")
        print("    LEFT: raw + red crop lines | RIGHT: what the network sees. CHECK IT.")


class DriveDataset(Dataset):
    def __init__(self, samples, train):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, steering = self.samples[i]
        img = cv2.imread(path)
        if img is None:                                  # unreadable file -> black frame, 0 steer
            img = np.zeros((240, 320, 3), np.uint8)
            steering = 0.0
        if self.train:
            img, steering = augment(img, steering)
        x = torch.from_numpy(preprocess(img))
        y = torch.tensor([steering], dtype=torch.float32)
        return x, y


# ========================== 3. THE PILOTNET MODEL ==========================

class PilotNet(nn.Module):
    """NVIDIA PilotNet conv stack, but the fully-connected layer sizes itself
    automatically for ANY input resolution (200x66, 200x150, 320x240, ...).
    The input size lives in the checkpoint config, so train and drive always
    build the exact same network."""
    def __init__(self, img_h, img_w):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
        )
        with torch.no_grad():                       # measure conv output size
            n_flat = self.conv(torch.zeros(1, 3, img_h, img_w)).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(n_flat, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ========================== 4. TRAIN LOOP ==================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="pilotnet.pt")
    ap.add_argument("--crop-top", type=int, default=0,
                    help="px removed from frame top (sky/ceiling)")
    ap.add_argument("--crop-bottom", type=int, default=0,
                    help="px removed from frame bottom (chassis)")
    ap.add_argument("--no-crop", action="store_true",
                    help="use the full raw frame (still resized to 200x66)")
    ap.add_argument("--img-w", type=int, default=200, help="network input width")
    ap.add_argument("--img-h", type=int, default=150, help="network input height (150 keeps 4:3, no distortion; 66 = classic PilotNet squash)")
    ap.add_argument("--preview", type=int, default=6,
                    help="save N preprocessing preview samples to dataset/ (0 = off)")
    args = ap.parse_args()

    if args.no_crop:
        CFG["crop_top"], CFG["crop_bottom"] = 0, 0
    else:
        CFG["crop_top"], CFG["crop_bottom"] = args.crop_top, args.crop_bottom
    CFG["img_w"], CFG["img_h"] = args.img_w, args.img_h

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    print("== refining dataset ==")
    rows = load_all_sessions(args.data)
    if not rows:
        print("No labels.csv found under", args.data); return

    # crop sanity check on a real frame from the dataset
    probe = cv2.imread(rows[0]["center"])
    if probe is not None:
        remaining = probe.shape[0] - CFG["crop_top"] - CFG["crop_bottom"]
        print(f"  frame {probe.shape[1]}x{probe.shape[0]}, crop top {CFG['crop_top']} "
              f"+ bottom {CFG['crop_bottom']} -> {remaining} px kept, resized to "
              f"{CFG['img_w']}x{CFG['img_h']}")
        if remaining < CFG["img_h"]:
            print(f"  ERROR: crop leaves only {remaining} px of a {probe.shape[0]} px "
                  f"frame - reduce --crop-top/--crop-bottom.")
            return

    samples = balance(build_samples(rows))
    write_merged_index(samples, args.data)
    if args.preview > 0:
        save_preview(samples, args.data, args.preview)

    n_val = max(1, int(len(samples) * 0.1))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    print(f"  train {len(train_samples)}  val {len(val_samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("== training on", device,
          torch.cuda.get_device_name(0) if device.type == "cuda" else "", "==")

    train_dl = DataLoader(DriveDataset(train_samples, True), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(DriveDataset(val_samples, False), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    model = PilotNet(CFG["img_h"], CFG["img_w"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    lossf = nn.MSELoss()
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train(); tr_loss, n = 0.0, 0
        tbar = tqdm(train_dl, desc=f"epoch {ep:3d}/{args.epochs} train", leave=False)
        for x, y in tbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item() * x.size(0); n += x.size(0)
            # live running loss + GPU memory in the bar
            if hasattr(tbar, "set_postfix"):
                post = {"loss": f"{tr_loss/max(n,1):.4f}"}
                if device.type == "cuda":
                    post["gpu"] = f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                tbar.set_postfix(post)
        sched.step()
        tr_loss /= max(n, 1)

        # ---- validate ----
        model.eval(); va_loss, va_mae, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"epoch {ep:3d}/{args.epochs} val  ", leave=False):
                x, y = x.to(device), y.to(device)
                p = model(x)
                va_loss += lossf(p, y).item() * x.size(0)
                va_mae += (p - y).abs().sum().item()
                n += x.size(0)
        va_loss /= max(n, 1); va_mae /= max(n, 1)

        mark = ""
        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "config": CFG}, args.out)
            mark = "  <- saved " + args.out
        print(f"epoch {ep:3d}/{args.epochs}  train MSE {tr_loss:.5f}  "
              f"val MSE {va_loss:.5f}  val MAE {va_mae:.4f}{mark}")

    print(f"done. best val MSE {best:.5f} -> {args.out}")


if __name__ == "__main__":
    main()