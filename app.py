"""Tkinter front-end for the nonlinear UAV controller laboratory."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Sequence, Tuple

import numpy as np

from simulation import CONTROLLER_INFO, DURATION, LiveSimulation, SimulationResult, run_comparison


BG = "#07111f"
SURFACE = "#0d1b2a"
CARD = "#102338"
CARD_2 = "#132b43"
BORDER = "#1f3b55"
TEXT = "#e6f1ff"
MUTED = "#8ca3b8"
CYAN = "#22d3ee"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#fb7185"


class Chart(tk.Canvas):
    def __init__(self, master, title: str, subtitle: str = "", **kwargs):
        super().__init__(master, bg=CARD, highlightthickness=1, highlightbackground=BORDER, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.series: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
        self.x_label = ""
        self.y_label = ""
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_series(self, series, x_label="", y_label=""):
        self.series = series
        self.x_label = x_label
        self.y_label = y_label
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 100 or h < 100:
            return
        self.create_text(18, 17, text=self.title, anchor="w", fill=TEXT, font=("Helvetica", 12, "bold"))
        if self.subtitle:
            self.create_text(18, 38, text=self.subtitle, anchor="w", fill=MUTED, font=("Helvetica", 9))
        if not self.series:
            self.create_text(w / 2, h / 2, text="Test tamamlandığında grafik burada görünecek", fill=MUTED)
            return

        left, top, right, bottom = 54, 64, w - 20, h - 38
        xs = np.concatenate([np.asarray(item[0]) for item in self.series if len(item[0])])
        ys = np.concatenate([np.asarray(item[1]) for item in self.series if len(item[1])])
        if not len(xs) or not len(ys):
            return
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        xpad = max((xmax - xmin) * 0.04, 0.1)
        ypad = max((ymax - ymin) * 0.10, 0.1)
        xmin, xmax = xmin - xpad, xmax + xpad
        ymin, ymax = ymin - ypad, ymax + ypad

        def sx(value):
            return left + (value - xmin) / (xmax - xmin) * (right - left)

        def sy(value):
            return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

        for i in range(6):
            x = left + i * (right - left) / 5
            y = top + i * (bottom - top) / 5
            self.create_line(x, top, x, bottom, fill="#17314a")
            self.create_line(left, y, right, y, fill="#17314a")
            xv = xmin + i * (xmax - xmin) / 5
            yv = ymax - i * (ymax - ymin) / 5
            self.create_text(x, bottom + 14, text=f"{xv:.0f}", fill=MUTED, font=("Helvetica", 8))
            self.create_text(left - 8, y, text=f"{yv:.1f}", anchor="e", fill=MUTED, font=("Helvetica", 8))

        for x_values, y_values, color, label in self.series:
            if len(x_values) < 2:
                continue
            stride = max(1, len(x_values) // 420)
            points = []
            for xv, yv in zip(x_values[::stride], y_values[::stride]):
                points.extend((sx(float(xv)), sy(float(yv))))
            self.create_line(*points, fill=color, width=2, smooth=False)

        legend_x = right - 8
        for offset, (_x, _y, color, label) in enumerate(reversed(self.series)):
            lx = legend_x - offset * 104
            self.create_line(lx - 74, 20, lx - 58, 20, fill=color, width=3)
            self.create_text(lx - 52, 20, text=label, anchor="w", fill=MUTED, font=("Helvetica", 8))
        if self.x_label:
            self.create_text((left + right) / 2, h - 10, text=self.x_label, fill=MUTED, font=("Helvetica", 8))
        if self.y_label:
            self.create_text(10, (top + bottom) / 2, text=self.y_label, angle=90, fill=MUTED, font=("Helvetica", 8))


class Route3DChart(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=CARD, highlightthickness=1, highlightbackground=BORDER, **kwargs)
        self.result: SimulationResult | None = None
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_result(self, result):
        self.result = result
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        self.create_text(18, 17, text="3B uçuş rotası", anchor="w", fill=TEXT, font=("Helvetica", 12, "bold"))
        self.create_text(18, 38, text="X–Y düzlemi + irtifa izdüşümü", anchor="w", fill=MUTED, font=("Helvetica", 9))
        if not self.result or w < 100:
            return
        target = self.result.target_position
        uav = self.result.uav_position
        both = np.vstack((target, uav))
        x, y, z = both[:, 0], both[:, 1], both[:, 2]
        # Isometric projection.
        px = x - 0.55 * y
        py = 0.32 * x + 0.32 * y - 1.25 * z
        projected = np.column_stack((px, py))
        minv, maxv = projected.min(axis=0), projected.max(axis=0)
        pad = np.maximum((maxv - minv) * 0.08, 1.0)
        minv -= pad
        maxv += pad

        def project(points):
            xp = points[:, 0] - 0.55 * points[:, 1]
            yp = 0.32 * points[:, 0] + 0.32 * points[:, 1] - 1.25 * points[:, 2]
            sx = 24 + (xp - minv[0]) / (maxv[0] - minv[0]) * (w - 48)
            sy = h - 24 - (yp - minv[1]) / (maxv[1] - minv[1]) * (h - 90)
            return np.column_stack((sx, sy))

        for points, color, width in ((target, AMBER, 2), (uav, CONTROLLER_INFO[self.result.controller]["color"], 3)):
            screen = project(points)
            stride = max(1, len(screen) // 420)
            flat = screen[::stride].reshape(-1).tolist()
            self.create_line(*flat, fill=color, width=width, smooth=False)
        self.create_text(w - 18, 20, text="— Hedef    — İHA", anchor="e", fill=MUTED, font=("Helvetica", 8))


class ControllerMonitor(tk.Canvas):
    """Radar-like live panel for one controller."""

    def __init__(self, master, controller_name: str, **kwargs):
        super().__init__(master, bg="#091725", highlightthickness=1, highlightbackground=BORDER, **kwargs)
        self.controller_name = controller_name
        self.result: SimulationResult | None = None
        self.frame_index = 0
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_frame(self, result: SimulationResult | None, frame_index: int = 0):
        self.result = result
        self.frame_index = frame_index
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        info = CONTROLLER_INFO[self.controller_name]
        self.create_text(14, 15, text=info["short"], anchor="w", fill=info["color"], font=("Helvetica", 10, "bold"))
        if not self.result:
            self.create_text(w / 2, h / 2, text="TEST BEKLİYOR", fill=MUTED, font=("Helvetica", 9, "bold"))
            return

        index = min(self.frame_index, len(self.result.time) - 1)
        t = float(self.result.time[index])
        error = float(self.result.distance_error[index])
        self.create_text(14, 33, text=f"t={t:05.1f} sn  |  hata={error:.2f} m", anchor="w", fill=MUTED, font=("Menlo", 8))

        left, top, right, bottom = 16, 47, w - 16, h - 48
        for grid_index in range(9):
            gx = left + grid_index * (right - left) / 8
            gy = top + grid_index * (bottom - top) / 8
            self.create_line(gx, top, gx, bottom, fill="#102a40")
            self.create_line(left, gy, right, gy, fill="#102a40")

        # Fixed mission map bounds prevent distracting zoom changes while the
        # route is being generated in real time.
        mins = np.array([-27.0, -22.0])
        maxs = np.array([35.0, 22.0])

        def project(points):
            sx = left + (points[:, 0] - mins[0]) / max(maxs[0] - mins[0], 1.0) * (right - left)
            sy = bottom - (points[:, 1] - mins[1]) / max(maxs[1] - mins[1], 1.0) * (bottom - top)
            return np.column_stack((sx, sy))

        target = project(self.result.target_position[: index + 1, :2])
        uav = project(self.result.uav_position[: index + 1, :2])
        if len(target) > 1:
            stride = max(1, len(target) // 250)
            self.create_line(*target[::stride].reshape(-1).tolist(), fill="#7c6421", width=1)
            self.create_line(*uav[::stride].reshape(-1).tolist(), fill=info["color"], width=3)
        tx, ty = project(self.result.target_position[index : index + 1, :2])[0]
        ux, uy = project(self.result.uav_position[index : index + 1, :2])[0]
        self.create_oval(tx - 6, ty - 6, tx + 6, ty + 6, outline=AMBER, width=2)
        self.create_line(tx - 9, ty, tx + 9, ty, fill=AMBER)
        self.create_line(tx, ty - 9, tx, ty + 9, fill=AMBER)
        # A small UAV symbol.
        self.create_polygon(ux, uy - 9, ux - 7, uy + 7, ux, uy + 4, ux + 7, uy + 7, fill=info["color"], outline=TEXT)

        # Error sparkline in a boxed overlay, as in the supplied dashboard reference.
        box_w, box_h = min(150, w * 0.38), 44
        bx0, by0 = right - box_w, bottom - box_h
        self.create_rectangle(bx0, by0, right, bottom, fill="#0b1724", outline=BORDER)
        self.create_text(bx0 + 7, by0 + 8, text="Takip hatası", anchor="w", fill=MUTED, font=("Helvetica", 7))
        errors = self.result.distance_error[: index + 1]
        if len(errors) > 1:
            recent = errors[-160:]
            peak = max(float(np.max(recent)), 0.1)
            spark = []
            for offset, value in enumerate(recent):
                x = bx0 + 6 + offset / max(len(recent) - 1, 1) * (box_w - 12)
                y = bottom - 5 - float(value) / peak * (box_h - 18)
                spark.extend((x, y))
            self.create_line(*spark, fill=info["color"], width=1)


class PerformanceBarChart(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=CARD, highlightthickness=1, highlightbackground=BORDER, **kwargs)
        self.results: Dict[str, SimulationResult] = {}
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_results(self, results):
        self.results = results
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        self.create_text(16, 15, text="Takip oranı ve doğruluk karşılaştırması", anchor="w", fill=TEXT, font=("Helvetica", 11, "bold"))
        self.create_text(w - 16, 15, text="■ Takip oranı   ■ Doğruluk", anchor="e", fill=MUTED, font=("Helvetica", 8))
        if not self.results or w < 200:
            return
        names = list(self.results)
        left, right, top, bottom = 34, w - 18, 34, h - 24
        group_width = (right - left) / len(names)
        for index, name in enumerate(names):
            center = left + (index + 0.5) * group_width
            tracking = self.results[name].metrics["tracking_rate"]
            accuracy = self.results[name].metrics["accuracy"]
            max_height = bottom - top
            bar_width = min(24, group_width * 0.18)
            for value, color, offset in ((tracking, GREEN, -bar_width * 0.65), (accuracy, CYAN, bar_width * 0.65)):
                y = bottom - max_height * value / 100.0
                self.create_rectangle(center + offset - bar_width / 2, y, center + offset + bar_width / 2, bottom, fill=color, outline="")
                self.create_text(center + offset, y - 7, text=f"{value:.0f}", fill=TEXT, font=("Helvetica", 7, "bold"))
            self.create_text(center, h - 10, text=CONTROLLER_INFO[name]["short"], fill=MUTED, font=("Helvetica", 7))


class MetricCard(tk.Frame):
    def __init__(self, master, label, value="—", accent=CYAN):
        super().__init__(master, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        self.accent = tk.Frame(self, bg=accent, width=4)
        self.accent.pack(side="left", fill="y")
        body = tk.Frame(self, bg=CARD)
        body.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        tk.Label(body, text=label, bg=CARD, fg=MUTED, font=("Helvetica", 9)).pack(anchor="w")
        self.value = tk.Label(body, text=value, bg=CARD, fg=TEXT, font=("Helvetica", 18, "bold"))
        self.value.pack(anchor="w", pady=(2, 0))

    def set(self, text):
        self.value.configure(text=text)


class UAVControlLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AEROTRACE — Doğrusal Olmayan İHA Kontrol Laboratuvarı")
        self.geometry("1440x900")
        self.minsize(1180, 760)
        self.configure(bg=BG)
        self.results: Dict[str, SimulationResult] = {}
        self.worker_queue: queue.Queue = queue.Queue()
        self.selected_controllers = {name: tk.BooleanVar(value=True) for name in CONTROLLER_INFO}
        self.playback_speed = tk.StringVar(value="1×")
        self.result_controller = tk.StringVar(value=list(CONTROLLER_INFO)[0])
        self._setup_style()
        self._build_layout()

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, padding=(20, 10), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", CARD_2)], foreground=[("selected", TEXT)])
        style.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT, rowheight=31, borderwidth=0)
        style.configure("Treeview.Heading", background=CARD_2, foreground=TEXT, relief="flat", font=("Helvetica", 9, "bold"))
        style.map("Treeview", background=[("selected", "#18415e")])
        style.configure("TCombobox", fieldbackground=CARD_2, background=CARD_2, foreground=TEXT)

    def _build_layout(self):
        header = tk.Frame(self, bg=BG, height=72)
        header.pack(fill="x", padx=22, pady=(14, 0))
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left")
        tk.Label(brand, text="AEROTRACE", bg=BG, fg=CYAN, font=("Helvetica", 17, "bold")).pack(anchor="w")
        tk.Label(brand, text="NONLINEAR CONTROL LAB", bg=BG, fg=MUTED, font=("Helvetica", 8, "bold")).pack(anchor="w")
        badge = tk.Label(header, text="●  SİMÜLASYON HAZIR", bg="#0b2d2a", fg=GREEN, padx=12, pady=7, font=("Helvetica", 8, "bold"))
        badge.pack(side="right", pady=10)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 20))
        self._build_sidebar(body)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True, padx=(18, 0))
        self.project_tab = tk.Frame(self.notebook, bg=BG)
        self.test_tab = tk.Frame(self.notebook, bg=BG)
        self.results_tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.project_tab, text="01  PROJE")
        self.notebook.add(self.test_tab, text="02  60 SN TEST")
        self.notebook.add(self.results_tab, text="03  SONUÇLAR")
        self._build_project_tab()
        self._build_test_tab()
        self._build_results_tab()

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=SURFACE, width=270, highlightthickness=1, highlightbackground=BORDER)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="KONTROLCÜLER", bg=SURFACE, fg=MUTED, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=18, pady=(20, 10))
        for name, info in CONTROLLER_INFO.items():
            card = tk.Frame(sidebar, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.pack(fill="x", padx=14, pady=5)
            tk.Checkbutton(
                card,
                text=info["short"],
                variable=self.selected_controllers[name],
                bg=CARD,
                activebackground=CARD,
                fg=TEXT,
                activeforeground=TEXT,
                selectcolor=SURFACE,
                font=("Helvetica", 10, "bold"),
                anchor="w",
            ).pack(fill="x", padx=10, pady=(9, 2))
            tk.Label(card, text=info["description"], bg=CARD, fg=MUTED, justify="left", wraplength=220, font=("Helvetica", 8)).pack(anchor="w", padx=12, pady=(0, 10))

        tk.Label(sidebar, text="TEST AYARI", bg=SURFACE, fg=MUTED, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=18, pady=(20, 8))
        row = tk.Frame(sidebar, bg=SURFACE)
        row.pack(fill="x", padx=18)
        tk.Label(row, text="Oynatma", bg=SURFACE, fg=TEXT).pack(side="left")
        speed = ttk.Combobox(row, textvariable=self.playback_speed, values=("1×",), width=7, state="disabled")
        speed.pack(side="right")
        tk.Label(sidebar, text="Gerçek test süresi: tam 60 saniye", bg=SURFACE, fg=MUTED, font=("Helvetica", 8)).pack(anchor="w", padx=18, pady=8)
        self.start_button = tk.Button(
            sidebar,
            text="60 SN TESTİ BAŞLAT",
            command=self.start_test,
            bg=CYAN,
            fg="#04212a",
            activebackground="#67e8f9",
            relief="flat",
            cursor="hand2",
            font=("Helvetica", 10, "bold"),
            pady=12,
        )
        self.start_button.pack(side="bottom", fill="x", padx=14, pady=14)

    def _section_title(self, parent, kicker, title, body):
        tk.Label(parent, text=kicker, bg=BG, fg=CYAN, font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(18, 5))
        tk.Label(parent, text=title, bg=BG, fg=TEXT, font=("Helvetica", 24, "bold")).pack(anchor="w")
        tk.Label(parent, text=body, bg=BG, fg=MUTED, justify="left", wraplength=920, font=("Helvetica", 10)).pack(anchor="w", pady=(6, 16))

    def _build_project_tab(self):
        outer = tk.Frame(self.project_tab, bg=BG)
        outer.pack(fill="both", expand=True, padx=6)
        self._section_title(
            outer,
            "PROJE ÖZETİ",
            "Hareketli hedefi izleyen otonom İHA",
            "Amaç, aynı doğrusal olmayan İHA modelinde dört farklı kontrol yaklaşımını adil biçimde karşılaştırmaktır. "
            "İHA; karesel hava sürüklemesi, zamana/konuma bağlı rüzgâr ve eyleyici sınırı altında üç boyutlu bir hedefi takip eder.",
        )
        stats = tk.Frame(outer, bg=BG)
        stats.pack(fill="x")
        for index, (number, label) in enumerate((("4", "DOĞRUSAL OLMAYAN KONTROLCÜ"), ("60 sn", "SABİT TEST UFKU"), ("3B", "X / Y / Z HEDEF TAKİBİ"), ("20 Hz", "SİMÜLASYON ÖRNEKLEMESİ"))):
            card = tk.Frame(stats, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=index, padx=(0, 10), sticky="nsew")
            stats.grid_columnconfigure(index, weight=1)
            tk.Label(card, text=number, bg=CARD, fg=TEXT, font=("Helvetica", 20, "bold")).pack(anchor="w", padx=15, pady=(12, 2))
            tk.Label(card, text=label, bg=CARD, fg=MUTED, font=("Helvetica", 8, "bold")).pack(anchor="w", padx=15, pady=(0, 12))

        tk.Label(outer, text="SİSTEM AKIŞI", bg=BG, fg=MUTED, font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(24, 10))
        flow = tk.Frame(outer, bg=BG)
        flow.pack(fill="x")
        steps = [
            ("01", "Hedef rotası", "Hedefin konum, hız ve ivmesi üretilir."),
            ("02", "Kontrolcü", "Seçilen yöntem 3 eksenli ivme komutu üretir."),
            ("03", "İHA modeli", "Sürükleme, rüzgâr ve doygunluk uygulanır."),
            ("04", "Analiz", "RMSE, takip oranı ve doğruluk hesaplanır."),
        ]
        for index, (number, title, desc) in enumerate(steps):
            card = tk.Frame(flow, bg=CARD_2, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=index, padx=(0, 10), sticky="nsew")
            flow.grid_columnconfigure(index, weight=1)
            tk.Label(card, text=number, bg=CARD_2, fg=CYAN, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=14, pady=(13, 5))
            tk.Label(card, text=title, bg=CARD_2, fg=TEXT, font=("Helvetica", 11, "bold")).pack(anchor="w", padx=14)
            tk.Label(card, text=desc, bg=CARD_2, fg=MUTED, justify="left", wraplength=190, font=("Helvetica", 8)).pack(anchor="w", padx=14, pady=(5, 14))

        note = tk.Frame(outer, bg="#0b2b38", highlightthickness=1, highlightbackground="#155e75")
        note.pack(fill="x", pady=22)
        tk.Label(note, text="Mühendislik notu", bg="#0b2b38", fg=CYAN, font=("Helvetica", 10, "bold")).pack(anchor="w", padx=16, pady=(12, 4))
        tk.Label(
            note,
            text="Bu proje eğitim ve kontrolcü kıyaslama amaçlı 3B noktasal-kütle modelidir; gerçek uçuş öncesinde 6-DOF model, sensör gürültüsü, gecikme ve donanım emniyet katmanları eklenmelidir.",
            bg="#0b2b38",
            fg="#b9dceb",
            justify="left",
            wraplength=940,
            font=("Helvetica", 9),
        ).pack(anchor="w", padx=16, pady=(0, 12))

    def _build_test_tab(self):
        outer = tk.Frame(self.test_tab, bg=BG)
        outer.pack(fill="both", expand=True, padx=6)
        top = tk.Frame(outer, bg=BG)
        top.pack(fill="x", pady=(18, 10))
        left = tk.Frame(top, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="CANLI TEST", bg=BG, fg=CYAN, font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(left, text="60 saniyelik görev zaman çizelgesi", bg=BG, fg=TEXT, font=("Helvetica", 22, "bold")).pack(anchor="w", pady=4)
        self.test_status = tk.Label(left, text="Kontrolcüleri seçin ve testi başlatın.", bg=BG, fg=MUTED, font=("Helvetica", 10))
        self.test_status.pack(anchor="w")
        self.timer_label = tk.Label(top, text="00:00.0 / 01:00.0", bg=CARD, fg=CYAN, font=("Menlo", 16, "bold"), padx=16, pady=12)
        self.timer_label.pack(side="right")

        content = tk.Frame(outer, bg=BG)
        content.pack(fill="both", expand=True)
        monitor_grid = tk.Frame(content, bg=BG)
        monitor_grid.pack(side="left", fill="both", expand=True)
        for row in range(2):
            monitor_grid.grid_rowconfigure(row, weight=1)
        for col in range(2):
            monitor_grid.grid_columnconfigure(col, weight=1)
        self.monitor_panels = {}
        for index, name in enumerate(CONTROLLER_INFO):
            panel = ControllerMonitor(monitor_grid, name)
            panel.grid(row=index // 2, column=index % 2, sticky="nsew", padx=(0, 7), pady=(0, 7))
            self.monitor_panels[name] = panel

        control_panel = tk.Frame(content, bg=SURFACE, width=255, highlightthickness=1, highlightbackground=BORDER)
        control_panel.pack(side="right", fill="y", padx=(5, 0))
        control_panel.pack_propagate(False)
        tk.Label(control_panel, text="SİMÜLASYON KONTROL PANELİ", bg=SURFACE, fg=TEXT, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=14, pady=(14, 8))
        self.servo_status = tk.Label(control_panel, text="İHA sistemi: HAZIR", bg="#12314a", fg=CYAN, font=("Helvetica", 8, "bold"), pady=8)
        self.servo_status.pack(fill="x", padx=12)
        tk.Frame(control_panel, height=1, bg=BORDER).pack(fill="x", padx=12, pady=12)
        self.live_leader = tk.Label(control_panel, text="Lider: —", bg=SURFACE, fg=GREEN, font=("Helvetica", 10, "bold"), justify="left")
        self.live_leader.pack(anchor="w", padx=14, pady=(0, 8))
        self.telemetry_text = tk.Label(
            control_panel,
            text="Test başladığında her kontrolcü için\nX/Y/Z hatası, takip oranı ve\nkontrol enerjisi burada görünür.",
            bg=SURFACE,
            fg=MUTED,
            justify="left",
            font=("Menlo", 8),
        )
        self.telemetry_text.pack(anchor="w", padx=14)
        tk.Frame(control_panel, height=1, bg=BORDER).pack(fill="x", padx=12, pady=12)
        tk.Label(control_panel, text="Değerlendirme kriterleri", bg=SURFACE, fg=TEXT, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=14)
        tk.Label(
            control_panel,
            text="• 3B konum RMSE\n• 0,5 m içinde takip oranı\n• Doğruluk puanı\n• Maksimum hata\n• Kontrol eforu",
            bg=SURFACE,
            fg=MUTED,
            justify="left",
            font=("Helvetica", 8),
        ).pack(anchor="w", padx=14, pady=7)
        self.progress = ttk.Progressbar(outer, maximum=60.0, mode="determinate")
        self.progress.pack(fill="x", pady=(10, 0))

    def _build_results_tab(self):
        outer = tk.Frame(self.results_tab, bg=BG)
        outer.pack(fill="both", expand=True, padx=6)
        heading = tk.Frame(outer, bg=BG)
        heading.pack(fill="x", pady=(18, 10))
        tk.Label(heading, text="TEST SONUÇLARI", bg=BG, fg=CYAN, font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(heading, text="Kontrolcü performans karşılaştırması", bg=BG, fg=TEXT, font=("Helvetica", 22, "bold")).pack(anchor="w", pady=4)
        chooser = ttk.Combobox(
            heading,
            textvariable=self.result_controller,
            values=list(CONTROLLER_INFO.keys()),
            state="readonly",
            width=30,
        )
        chooser.pack(side="right", anchor="s", pady=4)
        chooser.bind("<<ComboboxSelected>>", lambda _event: self.refresh_results())

        columns = ("controller", "rmse", "tracking", "accuracy", "max", "effort")
        self.table = ttk.Treeview(outer, columns=columns, show="headings", height=4)
        headers = ("Kontrolcü", "RMSE (m)", "Takip oranı", "Doğruluk", "Maks. hata", "Enerji")
        widths = (260, 100, 110, 105, 110, 100)
        for column, header, width in zip(columns, headers, widths):
            self.table.heading(column, text=header)
            self.table.column(column, width=width, anchor="center")
        self.table.pack(fill="x", pady=(0, 10))
        self.table.bind("<<TreeviewSelect>>", self._table_select)

        self.performance_chart = PerformanceBarChart(outer, height=132)
        self.performance_chart.pack(fill="x", pady=(0, 10))

        grid = tk.Frame(outer, bg=BG)
        grid.pack(fill="both", expand=True)
        for col in range(2):
            grid.grid_columnconfigure(col, weight=1)
        for row in range(2):
            grid.grid_rowconfigure(row, weight=1)
        self.position_chart = Chart(grid, "X / Y / Z koordinat takibi", "Kesik referans yerine hedef ve İHA eksen bazında gösterilir")
        self.position_chart.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
        self.error_chart = Chart(grid, "Konum hatası", "Öklid uzaklığı — daha düşük daha iyi")
        self.error_chart.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))
        self.xy_chart = Chart(grid, "X–Y rota karşılaştırması", "Hedef ve İHA yatay düzlem rotası")
        self.xy_chart.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(5, 0))
        self.route_chart = Route3DChart(grid)
        self.route_chart.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=(5, 0))

    def start_test(self):
        names = [name for name, variable in self.selected_controllers.items() if variable.get()]
        if not names:
            messagebox.showwarning("Kontrolcü seçilmedi", "En az bir kontrolcü seçmelisiniz.")
            return
        self.start_button.configure(state="disabled", text="TEST ÇALIŞIYOR…")
        self.test_status.configure(text="Canlı fizik modeli başlatılıyor…", fg=AMBER)
        self.notebook.select(self.test_tab)
        self.progress["value"] = 0
        self.live_simulations = {name: LiveSimulation(name) for name in names}
        self.results = {name: simulation.result() for name, simulation in self.live_simulations.items()}
        self.result_controller.set(names[0])
        self.after(100, self._begin_playback)

    def _compute_worker(self, names):
        try:
            self.worker_queue.put(("ok", run_comparison(names)))
        except Exception as exc:  # UI boundary: display unexpected model errors.
            self.worker_queue.put(("error", exc))

    def _poll_worker(self):
        try:
            status, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self.after(60, self._poll_worker)
            return
        if status == "error":
            self.start_button.configure(state="normal", text="60 SN TESTİ BAŞLAT")
            self.test_status.configure(text="Test başlatılamadı.", fg=RED)
            messagebox.showerror("Simülasyon hatası", str(payload))
            return
        self.results = payload
        first = next(iter(self.results))
        self.result_controller.set(first)
        self._begin_playback()

    def _begin_playback(self):
        # Use a monotonic wall clock so the result view cannot open early even
        # if the computer renders frames faster than expected.
        test_started_at = time.monotonic()
        interval_ms = 50
        self.test_status.configure(text="Dört kontrolcü gerçek zamanlı karşılaştırılıyor — test süresi 60 saniye", fg=GREEN)
        self.servo_status.configure(text="İHA sistemi: AKTİF", fg=GREEN)

        def tick():
            elapsed = min(time.monotonic() - test_started_at, DURATION)
            current_t = float(elapsed)

            # Each controller now advances its own nonlinear plant here. This
            # is a live simulation, not playback of a precomputed trajectory.
            for simulation in self.live_simulations.values():
                simulation.step_until(elapsed)
            self.results = {
                name: simulation.result()
                for name, simulation in self.live_simulations.items()
            }
            result = next(iter(self.results.values()))
            idx = len(result.time) - 1

            if elapsed >= DURATION:
                self.timer_label.configure(text="01:00.0 / 01:00.0")
                self.progress["value"] = 60
                self.start_button.configure(state="normal", text="TESTİ TEKRARLA")
                self.test_status.configure(text="60 saniyelik görev tamamlandı. Sonuç ekranı hazır.", fg=GREEN)
                self.servo_status.configure(text="İHA sistemi: TAMAMLANDI", fg=GREEN)
                self.refresh_results()
                self.notebook.select(self.results_tab)
                self.after(250, self.show_summary_dialog)
                return
            self.timer_label.configure(text=f"00:{current_t:04.1f} / 01:00.0" if current_t < 60 else "01:00.0 / 01:00.0")
            self.progress["value"] = current_t
            ranking = []
            telemetry = []
            for name, controller_result in self.results.items():
                self.monitor_panels[name].set_frame(controller_result, idx)
                partial_distance = controller_result.distance_error[: idx + 1]
                tracking = controller_result.metrics["tracking_rate"]
                accuracy = controller_result.metrics["accuracy"]
                rmse = controller_result.metrics["rmse_total"]
                ranking.append((rmse, name))
                position_error = controller_result.error[idx]
                telemetry.append(
                    f"{CONTROLLER_INFO[name]['short'][:16]:16}\n"
                    f"  dx {position_error[0]:+5.2f}  dy {position_error[1]:+5.2f}\n"
                    f"  dz {position_error[2]:+5.2f}  takip %{tracking:4.1f}\n"
                    f"  doğruluk %{accuracy:4.1f}"
                )
            for name, panel in self.monitor_panels.items():
                if name not in self.results:
                    panel.set_frame(None)
            leader = min(ranking)[1]
            self.live_leader.configure(text=f"Anlık performans lideri:\n{CONTROLLER_INFO[leader]['short']}")
            self.telemetry_text.configure(text="\n\n".join(telemetry))
            self.after(interval_ms, tick)

        tick()

    def show_summary_dialog(self):
        if not self.results:
            return
        dialog = tk.Toplevel(self)
        dialog.title("Kontrolcü Karşılaştırma Sonucu")
        dialog.geometry("1040x480")
        dialog.minsize(900, 430)
        dialog.configure(bg=BG)
        dialog.transient(self)
        dialog.grab_set()
        tk.Label(dialog, text="İHA DOĞRUSAL OLMAYAN KONTROLCÜ KARŞILAŞTIRMA SONUCU", bg=BG, fg=TEXT, font=("Helvetica", 15, "bold")).pack(anchor="w", padx=18, pady=(18, 3))
        tk.Label(dialog, text="Test süresi: 60 saniye  |  20 Hz örnekleme  |  3B hareketli hedef ve rüzgâr bozucusu", bg=BG, fg=MUTED, font=("Helvetica", 9)).pack(anchor="w", padx=18, pady=(0, 14))

        columns = ("controller", "mean", "p95", "min", "tracking", "accuracy", "effort", "score")
        table = ttk.Treeview(dialog, columns=columns, show="headings", height=len(self.results))
        labels = ("Kontrolcü", "Ort. hata", "P95 hata", "Min. hata", "Takip", "Doğruluk", "Efor", "Skor")
        widths = (220, 95, 95, 90, 90, 90, 90, 90)
        for column, label, width in zip(columns, labels, widths):
            table.heading(column, text=label)
            table.column(column, width=width, anchor="center")

        scored = []
        for name, result in self.results.items():
            m = result.metrics
            p95 = float(np.percentile(result.distance_error, 95))
            minimum = float(np.min(result.distance_error))
            score = 0.45 * m["tracking_rate"] + 0.45 * m["accuracy"] - 2.5 * m["rmse_total"] - 0.04 * m["control_effort"]
            scored.append((score, name))
            table.insert(
                "",
                "end",
                values=(CONTROLLER_INFO[name]["short"], f"{m['mean_error']:.2f} m", f"{p95:.2f} m", f"{minimum:.2f} m", f"%{m['tracking_rate']:.1f}", f"%{m['accuracy']:.1f}", f"{m['control_effort']:.1f}", f"{score:.1f}"),
            )
        table.pack(fill="x", padx=18)
        winner = max(scored)[1]
        winner_box = tk.Frame(dialog, bg="#0b2d2a", highlightthickness=1, highlightbackground="#1e6f5c")
        winner_box.pack(fill="x", padx=18, pady=16)
        tk.Label(winner_box, text=f"Genel skor kazananı: {CONTROLLER_INFO[winner]['short']}", bg="#0b2d2a", fg=GREEN, font=("Helvetica", 13, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        tk.Label(
            winner_box,
            text="Skor; ilk 2 saniyelik geçiş bölümü çıkarıldıktan sonra hedefi 0,5 m içinde tutma oranı, doğruluk, 3B RMSE ve kontrol eforunu birlikte değerlendirir. Böylece yalnızca kısa süreli yaklaşım değil, tüm görev boyunca kararlı ve ekonomik takip ödüllendirilir.",
            bg="#0b2d2a",
            fg="#b9dceb",
            justify="left",
            wraplength=960,
            font=("Helvetica", 9),
        ).pack(anchor="w", padx=14, pady=(0, 12))
        tk.Button(dialog, text="KAPAT", command=dialog.destroy, bg="#173b59", fg=TEXT, relief="flat", padx=28, pady=8, font=("Helvetica", 9, "bold")).pack(side="right", padx=18, pady=(0, 16))

    def _table_select(self, _event):
        selected = self.table.selection()
        if selected:
            name = selected[0]
            if name in self.results:
                self.result_controller.set(name)
                self.refresh_results()

    def refresh_results(self):
        if not self.results:
            return
        for item in self.table.get_children():
            self.table.delete(item)
        best = min(result.metrics["rmse_total"] for result in self.results.values())
        for name, result in self.results.items():
            m = result.metrics
            display_name = CONTROLLER_INFO[name]["short"] + ("  ★" if abs(m["rmse_total"] - best) < 1e-9 else "")
            self.table.insert(
                "",
                "end",
                iid=name,
                values=(display_name, f"{m['rmse_total']:.2f}", f"%{m['tracking_rate']:.1f}", f"%{m['accuracy']:.1f}", f"{m['max_error']:.2f} m", f"{m['control_effort']:.1f}"),
            )
        self.performance_chart.set_results(self.results)

        name = self.result_controller.get()
        if name not in self.results:
            name = next(iter(self.results))
            self.result_controller.set(name)
        result = self.results[name]
        t = result.time
        color = CONTROLLER_INFO[name]["color"]
        # Coordinate chart shows target as muted lines and UAV as vivid lines.
        self.position_chart.set_series(
            [
                (t, result.target_position[:, 0], "#6b7f91", "Hedef-X"),
                (t, result.uav_position[:, 0], "#38bdf8", "İHA-X"),
                (t, result.target_position[:, 1], "#7c6f8f", "Hedef-Y"),
                (t, result.uav_position[:, 1], "#a78bfa", "İHA-Y"),
                (t, result.target_position[:, 2], "#8f7e5b", "Hedef-Z"),
                (t, result.uav_position[:, 2], "#fbbf24", "İHA-Z"),
            ],
            "Zaman (sn)",
            "Konum (m)",
        )
        self.error_chart.set_series(
            [
                (item.time, item.distance_error, CONTROLLER_INFO[item_name]["color"], CONTROLLER_INFO[item_name]["short"])
                for item_name, item in self.results.items()
            ],
            "Zaman (sn)",
            "Hata (m)",
        )
        xy_series = [(result.target_position[:, 0], result.target_position[:, 1], AMBER, "Hedef")]
        xy_series.extend(
            (
                item.uav_position[:, 0],
                item.uav_position[:, 1],
                CONTROLLER_INFO[item_name]["color"],
                CONTROLLER_INFO[item_name]["short"],
            )
            for item_name, item in self.results.items()
        )
        self.xy_chart.set_series(xy_series, "X (m)", "Y (m)")
        self.route_chart.set_result(result)


if __name__ == "__main__":
    app = UAVControlLab()
    app.mainloop()
