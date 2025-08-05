#!/usr/bin/env python3
'''
Прототип GUI-детектора агрессии: Tkinter + OpenCV + PyTorch (EfficientNet-B3).
'''
from pathlib import Path
import sys, time, threading

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import models, transforms
import pandas as pd

import kagglehub

DATA_ROOT = Path(kagglehub.dataset_download("meetnagadia/human-action-recognition-har-dataset")) / "Human Action Recognition"
TRAIN_CSV = DATA_ROOT / 'Training_set.csv'               # meta-данные
MODEL_PATH = Path('./best_efficientnet_b3.pt')           # веса модели

IMG_SIZE = (300, 300)                                    # вход модели
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ───────────────────── чтение классов из CSV ───────────────────────────
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f'Не найден файл {TRAIN_CSV}')

train_df      = pd.read_csv(TRAIN_CSV)
CLASS_NAMES   = sorted(train_df['label'].unique())
NUM_CLASSES   = len(CLASS_NAMES)
LABEL2IDX     = {lbl: i for i, lbl in enumerate(CLASS_NAMES)}
IDX2LABEL     = {i: lbl for lbl, i in LABEL2IDX.items()}

# простое правило: «агрессивные» классы — те, в названии которых
# есть ключевые слова; скорректируйте при желании
_AGGR_KEYS = ('fight', 'punch', 'kick', 'hit', 'attack', 'grab')
AGGRESSIVE_LABELS = {lbl for lbl in CLASS_NAMES
                     if any(k in lbl.lower() for k in _AGGR_KEYS)}


# ─────────────────────────── преобразования ────────────────────────────
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
TFM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ─────────────────────────── загрузка модели ───────────────────────────
def load_model() -> torch.nn.Module:
    model = models.efficientnet_b3(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_f, NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


# ──────────────────────────── GUI-приложение ───────────────────────────
class VideoApp:
    def __init__(self, root: tk.Tk, source=1):
        self.root = root
        self.root.title('Aggression detector')

        # виджеты -------------------------------------------------------
        self.lbl_video  = ttk.Label(root)
        self.lbl_video.pack()

        self.var_status = tk.StringVar(value='Загрузка модели…')
        self.lbl_status = ttk.Label(root, textvariable=self.var_status,
                                    font=('Helvetica', 14, 'bold'))
        self.lbl_status.pack(pady=4)

        # инициализация -------------------------------------------------
        self.cap   = cv2.VideoCapture(source, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError('Не удалось открыть источник видео')

        self.model = load_model()

        # рабочий поток -------------------------------------------------
        self.running = True
        self.thread  = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ────────── класификация одного кадра ──────────
    @torch.no_grad()
    def _classify(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = TFM(rgb).unsqueeze(0).to(DEVICE)
        prob   = torch.softmax(self.model(tensor), dim=1)[0]
        idx    = int(torch.argmax(prob))
        return CLASS_NAMES[idx], float(prob[idx])

    # ────────── основной цикл ──────────
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            label, conf = self._classify(frame)
            bad = label in AGGRESSIVE_LABELS

            # overlay
            txt   = f'{label}: {conf:.0%}'
            color = (0,0,255) if bad else (0,255,0)
            cv2.putText(frame, txt, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Tk-render
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.lbl_video.imgtk = img_tk
            self.lbl_video.configure(image=img_tk)

            self.var_status.set(txt)
            self.lbl_status.configure(foreground='red' if bad else 'green')

            time.sleep(0.03)        # ~30 fps

    # ────────── корректное закрытие ──────────
    def _on_close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
        self.root.destroy()


# ───────────────────────────── entrypoint ──────────────────────────────
def main():
    src = 0 if len(sys.argv) < 2 else sys.argv[1]   # camera index or URL
    root = tk.Tk()
    VideoApp(root, 1)
    root.mainloop()


if __name__ == '__main__':
    main()