import sys
import os
import tempfile
from pathlib import Path
import multiprocessing as mp

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QFormLayout,
    QListWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from dotspotter.spotter import count_dots


# ------------------------------
# Display adjustment helper
# ------------------------------
def apply_display_adjustments(img: np.ndarray, params: dict) -> np.ndarray:
    if img is None:
        return img

    if img.dtype == np.uint16:
        img_f = img.astype(np.float32) / 65535.0
    else:
        img_f = img.astype(np.float32) / 255.0

    img_f = img_f * params["contrast"]
    img_f = img_f + params["brightness"]

    shadow_mask = img_f < 0.5
    img_f[shadow_mask] += params["shadows"] * (0.5 - img_f[shadow_mask])

    highlight_mask = img_f > 0.5
    img_f[highlight_mask] -= params["highlights"] * (img_f[highlight_mask] - 0.5)

    if params["gamma"] != 1.0:
        img_f = np.power(np.clip(img_f, 0, 1), 1.0 / params["gamma"])

    img8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
    return img8


# ------------------------------
# OpenCV → QPixmap
# ------------------------------
def cv_to_qpixmap(img: np.ndarray | None) -> QPixmap:
    if img is None:
        return QPixmap()

    if len(img.shape) == 2:
        h, w = img.shape
        bytes_per_line = w
        qimg = QImage(
            img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        )
        return QPixmap.fromImage(qimg)

    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            img_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimg)


# ------------------------------
# Histogram panel
# ------------------------------
class HistogramPanel(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(4, 3))
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

    def update_hist(self, counts: list[int]):
        self.ax.clear()
        if counts:
            self.ax.hist(
                counts,
                bins=min(20, len(counts)),
                color="steelblue",
                edgecolor="black",
            )
        self.ax.set_title("Dot count distribution")
        self.ax.set_xlabel("Count")
        self.ax.set_ylabel("Frequency")
        self.figure.tight_layout()
        self.draw()


# ------------------------------
# Preview worker
# ------------------------------
class PreviewWorker(QThread):
    finished = pyqtSignal(dict, object)

    def __init__(self, img_path: str, params: dict, temp_dir: Path):
        super().__init__()
        self.img_path = Path(img_path)
        self.params = params
        self.temp_dir = temp_dir

    def run(self):
        filename = self.img_path.name
        img_dir = self.img_path.parent

        args = (
            filename,
            img_dir,
            self.temp_dir,
            True,
            self.params["dot_size"],
            self.params["sensitivity"],
            self.params["preprocess"],
            not self.params["no_mask"],
            self.params["min_artifact"],
        )

        result = count_dots(args)

        qc_path = self.temp_dir / f"spottedQC_{filename}"
        processed_img = None
        if qc_path.exists():
            processed_img = cv2.imread(str(qc_path), cv2.IMREAD_ANYCOLOR)

        self.finished.emit(result, processed_img)


# ------------------------------
# Batch helper
# ------------------------------
def process_one(args):
    img_path, params, output_dir = args
    img_path = Path(img_path)
    filename = img_path.name
    img_dir = img_path.parent
    save_path = Path(output_dir)

    tup = (
        filename,
        img_dir,
        save_path,
        False,
        params["dot_size"],
        params["sensitivity"],
        params["preprocess"],
        not params["no_mask"],
        params["min_artifact"],
    )

    result = count_dots(tup)
    if result is None:
        return str(img_path), 0
    return str(img_path), result["count"]


# ------------------------------
# Main GUI
# ------------------------------
class DotspotterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dotspotter GUI")

        self.selected_files: list[str] = []
        self.batch_results: dict[str, int] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dotspotter_gui_"))

        self.zoom_factor = 1.0
        self.raw_img: np.ndarray | None = None
        self.proc_img: np.ndarray | None = None

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        left = QVBoxLayout()

        # File controls
        self.load_image_btn = QPushButton("Load single image")
        self.load_image_btn.clicked.connect(self.load_single_image)
        left.addWidget(self.load_image_btn)

        self.load_dir_btn = QPushButton("Load directory")
        self.load_dir_btn.clicked.connect(self.load_directory)
        left.addWidget(self.load_dir_btn)

        self.file_list = QListWidget()
        left.addWidget(self.file_list)

        # Parameters
        form = QFormLayout()

        self.dot_size = QDoubleSpinBox()
        self.dot_size.setRange(0.1, 20)
        self.dot_size.setValue(1.5)
        form.addRow("Dot size (px):", self.dot_size)

        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.1, 10)
        self.sensitivity.setValue(1.0)
        form.addRow("Sensitivity:", self.sensitivity)

        self.preprocess = QDoubleSpinBox()
        self.preprocess.setRange(0.1, 3.0)
        self.preprocess.setSingleStep(0.1)
        self.preprocess.setValue(1.0)
        form.addRow("Preprocess strength:", self.preprocess)

        self.no_mask = QCheckBox("Disable artefact masking")
        form.addRow(self.no_mask)

        self.min_artifact = QSpinBox()
        self.min_artifact.setRange(0, 1000000)
        self.min_artifact.setValue(5000)
        form.addRow("Min artefact area (px):", self.min_artifact)

        # Display adjustments
        self.brightness = QDoubleSpinBox()
        self.brightness.setRange(-1.0, 1.0)
        self.brightness.setSingleStep(0.05)
        self.brightness.setValue(0.0)
        form.addRow("Brightness:", self.brightness)

        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.1, 5.0)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(1.0)
        form.addRow("Contrast:", self.contrast)

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 5.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(1.0)
        form.addRow("Gamma:", self.gamma)

        self.shadows = QDoubleSpinBox()
        self.shadows.setRange(0.0, 1.0)
        self.shadows.setSingleStep(0.05)
        self.shadows.setValue(0.0)
        form.addRow("Shadows lift:", self.shadows)

        self.highlights = QDoubleSpinBox()
        self.highlights.setRange(0.0, 1.0)
        self.highlights.setSingleStep(0.05)
        self.highlights.setValue(0.0)
        form.addRow("Highlights compress:", self.highlights)

        for w in [
            self.brightness,
            self.contrast,
            self.gamma,
            self.shadows,
            self.highlights,
        ]:
            w.valueChanged.connect(self.refresh_display)

        left.addLayout(form)

        # Zoom buttons
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_out_btn = QPushButton("Zoom –")
        self.zoom_reset_btn = QPushButton("Reset zoom")

        self.zoom_in_btn.clicked.connect(lambda: self.change_zoom(1.1))
        self.zoom_out_btn.clicked.connect(lambda: self.change_zoom(1 / 1.1))
        self.zoom_reset_btn.clicked.connect(lambda: self.set_zoom(1.0))

        left.addWidget(self.zoom_in_btn)
        left.addWidget(self.zoom_out_btn)
        left.addWidget(self.zoom_reset_btn)

        # Action buttons
        self.preview_btn = QPushButton("Preview selected image")
        self.preview_btn.clicked.connect(self.preview_selected)
        left.addWidget(self.preview_btn)

        self.batch_btn = QPushButton("Process all images (batch)")
        self.batch_btn.clicked.connect(self.run_batch)
        left.addWidget(self.batch_btn)

        # Right side tabs
        self.tabs = QTabWidget()

        # Raw image tab
        self.raw_label = QLabel()
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.raw_scroll = QScrollArea()
        self.raw_scroll.setWidget(self.raw_label)
        self.raw_scroll.setWidgetResizable(True)

        self.tabs.addTab(self.raw_scroll, "Raw image")

        # Processed image tab
        self.proc_label = QLabel()
        self.proc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.proc_scroll = QScrollArea()
        self.proc_scroll.setWidget(self.proc_label)
        self.proc_scroll.setWidgetResizable(True)

        self.tabs.addTab(self.proc_scroll, "Processed image")

        # Summary tab
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()

        self.hist_panel = HistogramPanel()
        summary_layout.addWidget(self.hist_panel)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Image", "Count"])
        summary_layout.addWidget(self.summary_table)

        summary_widget.setLayout(summary_layout)
        self.tabs.addTab(summary_widget, "Summary")

        # CONNECT SIGNAL AFTER ALL TABS EXIST
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Main layout
        main = QHBoxLayout()
        main.addLayout(left, 1)
        main.addWidget(self.tabs, 3)
        self.setLayout(main)

    # ---------- Core helpers ----------
    def get_params(self) -> dict:
        return {
            "dot_size": self.dot_size.value(),
            "sensitivity": self.sensitivity.value(),
            "preprocess": self.preprocess.value(),
            "no_mask": self.no_mask.isChecked(),
            "min_artifact": self.min_artifact.value(),
            "brightness": self.brightness.value(),
            "contrast": self.contrast.value(),
            "gamma": self.gamma.value(),
            "shadows": self.shadows.value(),
            "highlights": self.highlights.value(),
        }

    # ---------- FIXED ZOOM BEHAVIOUR ----------
    def render_image(self, img: np.ndarray | None, label: QLabel):
        if img is None:
            label.clear()
            return

        params = self.get_params()
        adj = apply_display_adjustments(img, params)
        pix = cv_to_qpixmap(adj)

        # --- FIX: zoom based on image size, not label size ---
        h0, w0 = adj.shape[:2]
        w = int(w0 * self.zoom_factor)
        h = int(h0 * self.zoom_factor)

        label.setPixmap(
            pix.scaled(
                w,
                h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def refresh_display(self):
        if self.raw_img is not None:
            self.render_image(self.raw_img, self.raw_label)
        if self.proc_img is not None:
            self.render_image(self.proc_img, self.proc_label)

    def change_zoom(self, factor):
        self.zoom_factor *= factor
        self.refresh_display()

    def set_zoom(self, value):
        self.zoom_factor = value
        self.refresh_display()

    def sync_scroll_from_raw(self):
        self.proc_scroll.horizontalScrollBar().setValue(
            self.raw_scroll.horizontalScrollBar().value()
        )
        self.proc_scroll.verticalScrollBar().setValue(
            self.raw_scroll.verticalScrollBar().value()
        )

    def sync_scroll_from_proc(self):
        self.raw_scroll.horizontalScrollBar().setValue(
            self.proc_scroll.horizontalScrollBar().value()
        )
        self.raw_scroll.verticalScrollBar().setValue(
            self.proc_scroll.verticalScrollBar().value()
        )

    def on_tab_changed(self, idx: int):
        if idx == 0:
            self.sync_scroll_from_proc()
        elif idx == 1:
            self.sync_scroll_from_raw()

    # ---------- Mouse wheel zoom ----------
    def wheelEvent(self, event):
        if self.tabs.currentWidget() in [self.raw_scroll, self.proc_scroll]:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1
            self.refresh_display()
        else:
            super().wheelEvent(event)

    # ---------- File loading ----------
    def load_single_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not path:
            return
        self.selected_files = [path]
        self.file_list.clear()
        self.file_list.addItem(path)
        self.show_raw_image(path)

    def load_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select directory")
        if not directory:
            return

        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = [
            str(Path(directory) / f)
            for f in sorted(os.listdir(directory))
            if f.lower().endswith(exts)
        ]

        self.selected_files = files
        self.file_list.clear()
        self.file_list.addItems(files)

        if files:
            self.show_raw_image(files[0])

    # ---------- Display raw ----------
    def show_raw_image(self, path: str):
        self.raw_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        self.render_image(self.raw_img, self.raw_label)

    # ---------- Preview ----------
    def preview_selected(self):
        if not self.selected_files:
            return

        row = self.file_list.currentRow()
        if row < 0:
            row = 0

        img_path = self.selected_files[row]
        self.show_raw_image(img_path)

        params = self.get_params()
        self.preview_btn.setEnabled(False)
        self.setWindowTitle("dotspotter GUI — previewing...")

        self.preview_worker = PreviewWorker(img_path, params, self.temp_dir)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.start()

    def on_preview_finished(self, result: dict | None, processed_img):
        self.preview_btn.setEnabled(True)

        if result is None:
            self.setWindowTitle("dotspotter GUI — preview failed")
            return

        count = result["count"]

        if processed_img is not None:
            self.proc_img = processed_img
            self.render_image(self.proc_img, self.proc_label)

        self.setWindowTitle(f"dotspotter GUI — preview count: {count}")

    # ---------- Batch ----------
    def run_batch(self):
        if not self.selected_files:
            return

        params = self.get_params()
        self.batch_btn.setEnabled(False)
        self.setWindowTitle("dotspotter GUI — batch processing...")

        output_dir = self.temp_dir

        args_list = [(p, params, output_dir) for p in self.selected_files]

        def _run():
            with mp.Pool(mp.cpu_count()) as pool:
                for img_path, count in pool.imap_unordered(process_one, args_list):
                    self.batch_results[img_path] = count
                    self.update_summary()

            self.batch_btn.setEnabled(True)
            self.setWindowTitle("dotspotter GUI — batch complete")

        self.batch_thread = QThread()
        self.batch_thread.run = _run
        self.batch_thread.start()

    def update_summary(self):
        counts = list(self.batch_results.values())
        self.hist_panel.update_hist(counts)

        items = sorted(self.batch_results.items(), key=lambda x: x[0])
        self.summary_table.setRowCount(len(items))

        for row, (img, count) in enumerate(items):
            self.summary_table.setItem(row, 0, QTableWidgetItem(Path(img).name))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(count)))


# ------------------------------
# Entry point
# ------------------------------
def main():
    mp.freeze_support()
    app = QApplication(sys.argv)
    gui = DotspotterGUI()
    gui.resize(1400, 900)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()