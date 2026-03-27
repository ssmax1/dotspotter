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
    return str(img_path), result


class DotspotterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dotspotter")

        self.selected_files: list[str] = []
        self.batch_results: dict[str, int] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dotspotter_gui_"))

        self.zoom_factor = 1.0
        self.raw_img: np.ndarray | None = None
        self.proc_img: np.ndarray | None = None

        self._scaled_w: int | None = None
        self._scaled_h: int | None = None

        self._syncing = False

        self._build_ui()

    def _build_ui(self):
        left = QVBoxLayout()

        self.load_image_btn = QPushButton("Load single image")
        self.load_image_btn.clicked.connect(self.load_single_image)
        left.addWidget(self.load_image_btn)

        self.load_dir_btn = QPushButton("Load directory")
        self.load_dir_btn.clicked.connect(self.load_directory)
        left.addWidget(self.load_dir_btn)

        self.file_list = QListWidget()
        left.addWidget(self.file_list)

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

        self.brightness = QDoubleSpinBox()
        self.brightness.setRange(-1.0, 1.0)
        self.brightness.setSingleStep(0.05)
        self.brightness.setValue(0.5)
        form.addRow("Brightness:", self.brightness)

        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.1, 5.0)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(5.0)
        form.addRow("Contrast:", self.contrast)

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 5.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(0.4)
        form.addRow("Gamma:", self.gamma)

        self.shadows = QDoubleSpinBox()
        self.shadows.setRange(0.0, 1.0)
        self.shadows.setSingleStep(0.05)
        self.shadows.setValue(0.0)
        form.addRow("Shadows lift:", self.shadows)

        self.highlights = QDoubleSpinBox()
        self.highlights.setRange(0.0, 1.0)
        self.highlights.setSingleStep(0.05)
        self.highlights.setValue(0.25)
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

        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_out_btn = QPushButton("Zoom –")
        self.zoom_reset_btn = QPushButton("Reset zoom")

        self.zoom_in_btn.clicked.connect(lambda: self.change_zoom(1.1))
        self.zoom_out_btn.clicked.connect(lambda: self.change_zoom(1 / 1.1))
        self.zoom_reset_btn.clicked.connect(lambda: self.set_zoom(1.0))

        left.addWidget(self.zoom_in_btn)
        left.addWidget(self.zoom_out_btn)
        left.addWidget(self.zoom_reset_btn)

        self.preview_btn = QPushButton("Process selected image")
        self.preview_btn.clicked.connect(self.preview_selected)
        left.addWidget(self.preview_btn)

        self.batch_btn = QPushButton("Process all images (batch)")
        self.batch_btn.clicked.connect(self.run_batch)
        left.addWidget(self.batch_btn)
        
        self.export_btn = QPushButton("Export summary to CSV")
        self.export_btn.clicked.connect(self.export_summary)
        left.addWidget(self.export_btn)

        self.tabs = QTabWidget()

        self.raw_label = QLabel()
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.raw_label.setScaledContents(False)

        self.raw_scroll = QScrollArea()
        self.raw_scroll.setWidget(self.raw_label)
        self.raw_scroll.setWidgetResizable(True)

        self.tabs.addTab(self.raw_scroll, "Raw image")

        self.proc_label = QLabel()
        self.proc_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.proc_label.setScaledContents(False)

        self.proc_scroll = QScrollArea()
        self.proc_scroll.setWidget(self.proc_label)
        self.proc_scroll.setWidgetResizable(True)

        self.tabs.addTab(self.proc_scroll, "Processed image")

        summary_widget = QWidget()
        summary_layout = QVBoxLayout()

        self.hist_panel = HistogramPanel()
        summary_layout.addWidget(self.hist_panel)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels([
            "Image",
            "Count",
            "Masking Used",
            "Percent Masked",
            "Estimated Total"
        ])

        self.summary_table.setSortingEnabled(True)

        summary_layout.addWidget(self.summary_table)

        summary_widget.setLayout(summary_layout)
        self.tabs.addTab(summary_widget, "Summary")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.raw_scroll.horizontalScrollBar().valueChanged.connect(
            lambda _: self.sync_scroll_from_raw()
        )
        self.raw_scroll.verticalScrollBar().valueChanged.connect(
            lambda _: self.sync_scroll_from_raw()
        )

        self.proc_scroll.horizontalScrollBar().valueChanged.connect(
            lambda _: self.sync_scroll_from_proc()
        )
        self.proc_scroll.verticalScrollBar().valueChanged.connect(
            lambda _: self.sync_scroll_from_proc()
        )

        main = QHBoxLayout()
        main.addLayout(left, 1)
        main.addWidget(self.tabs, 3)
        self.setLayout(main)

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

    def render_image(self, img: np.ndarray | None, label: QLabel, is_raw: bool):
        if img is None:
            label.clear()
            return

        params = self.get_params()
        adj = apply_display_adjustments(img, params)
        pix = cv_to_qpixmap(adj)

        h0, w0 = adj.shape[:2]

        if is_raw:
            self._scaled_w = int(w0 * self.zoom_factor)
            self._scaled_h = int(h0 * self.zoom_factor)
        else:
            if self._scaled_w is None or self._scaled_h is None:
                self._scaled_w = int(w0 * self.zoom_factor)
                self._scaled_h = int(h0 * self.zoom_factor)

        w = self._scaled_w
        h = self._scaled_h

        label.setPixmap(
            pix.scaled(
                w,
                h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _get_reference_scrollbars(self):
        if self.tabs.currentIndex() == 1:
            area = self.proc_scroll
        else:
            area = self.raw_scroll
        return area.horizontalScrollBar(), area.verticalScrollBar()

    def _get_scroll_fractions(self):
        hbar, vbar = self._get_reference_scrollbars()
        fx = hbar.value() / hbar.maximum() if hbar.maximum() > 0 else 0.0
        fy = vbar.value() / vbar.maximum() if vbar.maximum() > 0 else 0.0
        return fx, fy

    def _apply_scroll_fractions(self, fx: float, fy: float):
        self._syncing = True
        for area in (self.raw_scroll, self.proc_scroll):
            hbar = area.horizontalScrollBar()
            vbar = area.verticalScrollBar()
            if hbar.maximum() > 0:
                hbar.setValue(int(fx * hbar.maximum()))
            else:
                hbar.setValue(0)
            if vbar.maximum() > 0:
                vbar.setValue(int(fy * vbar.maximum()))
            else:
                vbar.setValue(0)
        self._syncing = False

    def refresh_display(self):
        fx, fy = self._get_scroll_fractions()

        if self.raw_img is not None:
            self.render_image(self.raw_img, self.raw_label, is_raw=True)
        if self.proc_img is not None:
            self.render_image(self.proc_img, self.proc_label, is_raw=False)

        self._apply_scroll_fractions(fx, fy)

    def change_zoom(self, factor):
        self.zoom_factor *= factor
        self.zoom_factor = max(1, min(self.zoom_factor, 8.0))
        self.refresh_display()

    def set_zoom(self, value):
        self.zoom_factor = value
        self.zoom_factor = max(1, min(self.zoom_factor, 8.0))
        self.refresh_display()

    def sync_scroll_from_raw(self):
        if self._syncing:
            return
        self._syncing = True
        self.proc_scroll.horizontalScrollBar().setValue(
            self.raw_scroll.horizontalScrollBar().value()
        )
        self.proc_scroll.verticalScrollBar().setValue(
            self.raw_scroll.verticalScrollBar().value()
        )
        self._syncing = False

    def sync_scroll_from_proc(self):
        if self._syncing:
            return
        self._syncing = True
        self.raw_scroll.horizontalScrollBar().setValue(
            self.proc_scroll.horizontalScrollBar().value()
        )
        self.raw_scroll.verticalScrollBar().setValue(
            self.proc_scroll.verticalScrollBar().value()
        )
        self._syncing = False

    def on_tab_changed(self, idx: int):
        if idx == 0:
            self.sync_scroll_from_proc()
        elif idx == 1:
            self.sync_scroll_from_raw()

    def wheelEvent(self, event):
        if self.tabs.currentWidget() in [self.raw_scroll, self.proc_scroll]:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1
            
            self.zoom_factor = max(1.0, min(self.zoom_factor, 8.0))

            self.refresh_display()
        else:
            super().wheelEvent(event)

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

    def show_raw_image(self, path: str):
        self.raw_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        self.render_image(self.raw_img, self.raw_label, is_raw=True)
        self._apply_scroll_fractions(0.0, 0.0)

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
        self.setWindowTitle("dotspotter — previewing...")

        self.preview_worker = PreviewWorker(img_path, params, self.temp_dir)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.start()

    def on_preview_finished(self, result: dict | None, processed_img):
        self.preview_btn.setEnabled(True)

        if result is None:
            self.setWindowTitle("dotspotter — preview failed")
            return

        count = result["count"]

        if processed_img is not None:
            self.proc_img = processed_img
            fx, fy = self._get_scroll_fractions()
            self.render_image(self.proc_img, self.proc_label, is_raw=False)
            self._apply_scroll_fractions(fx, fy)

        self.setWindowTitle(f"dotspotter — dot counts: {count}")

    def run_batch(self):
        if not self.selected_files:
            return

        params = self.get_params()
        self.batch_btn.setEnabled(False)
        self.setWindowTitle("dotspotter — batch processing...")

        output_dir = self.temp_dir

        args_list = [(p, params, output_dir) for p in self.selected_files]

        def _run():
            with mp.Pool(mp.cpu_count()) as pool:
                for img_path, result in pool.imap_unordered(process_one, args_list):
                    self.batch_results[img_path] = result
                    self.update_summary()

            self.batch_btn.setEnabled(True)
            self.setWindowTitle("dotspotter — batch complete")

        self.batch_thread = QThread()
        self.batch_thread.run = _run
        self.batch_thread.start()

    def export_summary(self):
        if not self.batch_results:
            return

        # choose save location
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save summary CSV",
            "dotspotter_summary.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        params = self.get_params()

        with open(path, "w") as f:
            # write header
            f.write("dotspotter summary export\n")
            f.write(f"dot_size,{params['dot_size']}\n")
            f.write(f"sensitivity,{params['sensitivity']}\n")
            f.write(f"preprocess_strength,{params['preprocess']}\n")
            f.write(f"masking_enabled,{not params['no_mask']}\n")
            f.write(f"min_artifact_area,{params['min_artifact']}\n")
            f.write("\n")

            # table header
            f.write("image,count,masking_used,percent_masked,estimated_total\n")

            # table rows
            for img, result in sorted(self.batch_results.items()):
                f.write(
                    f"{Path(img).name},"
                    f"{result['count']},"
                    f"{result['masking_used']},"
                    f"{result['percent_masked']},"
                    f"{result['estimated_total_count']}\n"
                )

    def update_summary(self):
        counts = [result["count"] for result in self.batch_results.values()]
        self.hist_panel.update_hist(counts)

        items = sorted(self.batch_results.items(), key=lambda x: x[0])
        self.summary_table.setRowCount(len(items))

        for row, (img, result) in enumerate(items):
            self.summary_table.setItem(row, 0, QTableWidgetItem(Path(img).name))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(result["count"])))
            self.summary_table.setItem(row, 2, QTableWidgetItem(result["masking_used"]))
            self.summary_table.setItem(row, 3, QTableWidgetItem(f"{result['percent_masked']:.2f}%"))
            self.summary_table.setItem(row, 4, QTableWidgetItem(str(result["estimated_total_count"])))
        
        self.summary_table.resizeColumnsToContents()
        self.summary_table.horizontalHeader().setStretchLastSection(True)

def main():
    mp.freeze_support()
    app = QApplication(sys.argv)
    gui = DotspotterGUI()
    gui.resize(1400, 900)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()