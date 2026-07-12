"""
AFC&gtr Calculator Dialog
=========================
A pop-up panel for selecting experiment time & voltage files,
plotting a 2D spectrogram with voltage event markers.
"""

import sys
import os
import json
import logging as _log
_log.getLogger('matplotlib.ticker').setLevel(_log.WARNING)

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
from matplotlib import gridspec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from rionid.nonparams_est import NONPARAMS_EST

from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QSplitter,
    QFrame, QCheckBox, QDesktopWidget, QPlainTextEdit, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QCursor
from PyQt5.QtWidgets import QMenu

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
COMMON_FONT_SIZE = 12


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas embedded in Qt."""

    def __init__(self, parent=None, width=8, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(400)

    def clear(self):
        self.fig.clear()


class AFCCalculatorDialog(QDialog):
    """AFC&gtr Calculator main dialog.

    Panel contains:
    - Experiment time file selection
    - Voltage file selection
    - 2D spectrogram plot with voltage event overlays
    """

    # Emitted when the user clicks "Load Data & Plot" — carries current file paths
    paths_changed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AFC&gtr Calculator")
        self.setMinimumSize(900, 700)

        # ── Data cache ──
        self.time_data = None           # time array (1D) -> Y axis
        self.freq_data = None           # freq array (1D) -> X axis
        self.spectrogram_data = None    # 2D spectrogram matrix
        self.voltage_data = None        # voltage timeline array (1D)
        self.voltage_time = None        # voltage time axis
        self._time_offset = 0.0         # original min(time) before normalisation
        self._projections = None        # list of (time_label, freq_1d_array) for projection panel
        self._projections_baseline = None  # estimated baseline per projection
        self._projections_clean = None  # baseline-removed projections (same shape)
        self._projections_peaks = None  # list of peak indices per projection
        self._projections_peak_masks = None  # list of bool arrays: True=keep peak
        self._projections_peak_data = None  # list of dicts with freqs/heights/widths/areas
        self._projections_har_fit = None  # list of (f0, offset, residuals) per projection
        self._proj_voltages = None      # voltage values for projection labels
        self._proj_nframes = None       # number of spectrogram frames per projection

        # ── Threshold profile ──
        self.threshold_profile_freqs = None  # 1D array of frequency values
        self.threshold_profile_vals = None  # 1D array of threshold values
        self._thresh_profile_path = ""       # path to the current threshold CSV
        self._thresh_click_mode = False
        self._thresh_cid = None
        self._proj_axes = []  # cached list of projection Axes for lightweight redraw
        self._font_scale = 1.0  # font size multiplier
        self._locked_ax_positions = []  # list of [x0, y0, w, h] per axis in figure coords
        self._last_res_params = None  # (A0, Q, f_sys) from last resonance fit
        self._config_path = os.path.join(os.getcwd(), "afc_calculator_config.json")

        self._init_ui()

        # ── Auto-load threshold CSV from cwd if it exists ──
        default_thresh = os.path.join(os.getcwd(), "height_thresh.csv")
        if os.path.exists(default_thresh):
            self._load_threshold_profile(default_thresh)

        # ── Load saved config ──
        self._load_config()

        # ── Maximise to near-full-screen ──
        screen = QDesktopWidget().screenGeometry(-1)
        self.setGeometry(10, 10, int(screen.width() * 0.92), int(screen.height() * 0.88))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI construction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _init_ui(self):
        self._clear_debug_log()
        main_layout = QVBoxLayout(self)

        # ── File selection area ──
        file_frame = QFrame()
        file_frame.setFrameShape(QFrame.StyledPanel)
        file_layout = QVBoxLayout(file_frame)

        # Experiment time file
        h1 = QHBoxLayout()
        self.time_label = QLabel("Experiment Time File:")
        self.time_label.setMinimumWidth(130)
        self.time_edit = QLineEdit()
        self.time_edit.setPlaceholderText("Select file with time & frequency data...")
        self.time_browse_btn = QPushButton("Browse")
        self.time_browse_btn.clicked.connect(self._browse_time_file)
        h1.addWidget(self.time_label)
        h1.addWidget(self.time_edit, 1)
        h1.addWidget(self.time_browse_btn)
        # "📊 Load Data & Plot" button — simplified style for row placement
        self.load_plot_btn = QPushButton("📊 Load Data & Plot")
        self.load_plot_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 3px;
                padding: 2px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.load_plot_btn.clicked.connect(self._load_and_plot)
        h1.addWidget(self.load_plot_btn)
        file_layout.addLayout(h1)

        # Voltage file
        h2 = QHBoxLayout()
        self.voltage_label = QLabel("Voltage File:")
        self.voltage_label.setMinimumWidth(130)
        self.voltage_edit = QLineEdit()
        self.voltage_edit.setPlaceholderText("Select voltage data file...")
        self.voltage_browse_btn = QPushButton("Browse")
        self.voltage_browse_btn.clicked.connect(self._browse_voltage_file)
        self.voltage_edit_btn = QPushButton("Edit")
        self.voltage_edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 3px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.voltage_edit_btn.clicked.connect(self._edit_voltage_file)
        h2.addWidget(self.voltage_label)
        h2.addWidget(self.voltage_edit, 1)
        h2.addWidget(self.voltage_browse_btn)
        h2.addWidget(self.voltage_edit_btn)
        # "Update Voltage Only" button — simplified style for row placement
        self.update_volt_btn = QPushButton("Update Voltage Only")
        self.update_volt_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.update_volt_btn.clicked.connect(self._update_voltage_only)
        h2.addWidget(self.update_volt_btn)
        file_layout.addLayout(h2)

        # (Load & Plot / Update Voltage Only moved into rows above)
        main_layout.addWidget(file_frame)

        # ── Threshold profile row ──
        thresh_frame = QFrame()
        thresh_frame.setFrameShape(QFrame.StyledPanel)
        thresh_layout = QHBoxLayout(thresh_frame)
        thresh_layout.setContentsMargins(4, 2, 4, 2)

        thresh_layout.addWidget(QLabel("Threshold:"))
        self.thresh_path_edit = QLineEdit()
        self.thresh_path_edit.setPlaceholderText("height_thresh.csv path...")
        self.thresh_path_edit.setFont(QFont("", 9))
        thresh_layout.addWidget(self.thresh_path_edit, 1)

        self.thresh_browse_btn = QPushButton("Browse")
        self.thresh_browse_btn.setFont(QFont("", 9))
        self.thresh_browse_btn.clicked.connect(self._on_thresh_browse)
        thresh_layout.addWidget(self.thresh_browse_btn)

        self.thresh_click_btn = QPushButton("Start Click Threshold")
        self.thresh_click_btn.setFont(QFont("", 9))
        self.thresh_click_btn.clicked.connect(self._on_toggle_threshold_click)
        thresh_layout.addWidget(self.thresh_click_btn)

        self.thresh_clear_btn = QPushButton("Clear")
        self.thresh_clear_btn.setFont(QFont("", 9))
        self.thresh_clear_btn.clicked.connect(self._on_thresh_clear)
        thresh_layout.addWidget(self.thresh_clear_btn)

        self.thresh_update_btn = QPushButton("Update")
        self.thresh_update_btn.setFont(QFont("", 9))
        self.thresh_update_btn.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 3px;")
        self.thresh_update_btn.clicked.connect(self._on_thresh_update)
        thresh_layout.addWidget(self.thresh_update_btn)

        # ── Baseline removal controls ──
        thresh_layout.addSpacing(8)
        thresh_layout.addWidget(QLabel("l:"))
        self.baseline_l_edit = QLineEdit("100")
        self.baseline_l_edit.setMaximumWidth(35)
        thresh_layout.addWidget(self.baseline_l_edit)
        thresh_layout.addWidget(QLabel("ratio:"))
        self.baseline_ratio_edit = QLineEdit("0.001")
        self.baseline_ratio_edit.setMaximumWidth(40)
        thresh_layout.addWidget(self.baseline_ratio_edit)
        self.remove_baseline_btn = QPushButton("Remove Baseline")
        self.remove_baseline_btn.setFont(QFont("", 9))
        self.remove_baseline_btn.setStyleSheet("background-color: #FF5722; color: white; border-radius: 3px;")
        self.remove_baseline_btn.clicked.connect(self._remove_baseline_projections)
        thresh_layout.addWidget(self.remove_baseline_btn)

        main_layout.addWidget(thresh_frame)

        # ── Axis range controls ──
        range_frame = QFrame()
        range_frame.setFrameShape(QFrame.StyledPanel)
        range_layout = QHBoxLayout(range_frame)

        range_layout.addWidget(QLabel("X range:"))
        self.xmin_edit = QLineEdit()
        self.xmin_edit.setPlaceholderText("auto")
        self.xmin_edit.setMaximumWidth(80)
        self.xmax_edit = QLineEdit()
        self.xmax_edit.setPlaceholderText("auto")
        self.xmax_edit.setMaximumWidth(80)
        range_layout.addWidget(self.xmin_edit)
        range_layout.addWidget(QLabel("–"))
        range_layout.addWidget(self.xmax_edit)

        range_layout.addSpacing(10)

        range_layout.addWidget(QLabel("Y range:"))
        self.ymin_edit = QLineEdit()
        self.ymin_edit.setPlaceholderText("0")
        self.ymin_edit.setMaximumWidth(80)
        self.ymax_edit = QLineEdit()
        self.ymax_edit.setPlaceholderText("auto")
        self.ymax_edit.setMaximumWidth(80)
        range_layout.addWidget(self.ymin_edit)
        range_layout.addWidget(QLabel("–"))
        range_layout.addWidget(self.ymax_edit)

        range_layout.addSpacing(10)

        range_layout.addSpacing(5)
        self.reset_range_btn = QPushButton("Reset")
        self.reset_range_btn.clicked.connect(self._reset_range)
        range_layout.addWidget(self.reset_range_btn)

        # Connect editingFinished to redraw
        for w in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            w.editingFinished.connect(self._on_range_changed)

        # ── Projection controls ──
        range_layout.addSpacing(10)
        range_layout.addWidget(QLabel("Proj:"))
        self.proj_offset_edit = QLineEdit("5")
        self.proj_offset_edit.setMaximumWidth(40)
        range_layout.addWidget(self.proj_offset_edit)
        range_layout.addWidget(QLabel("+"))
        self.proj_dt_edit = QLineEdit("20")
        self.proj_dt_edit.setMaximumWidth(40)
        range_layout.addWidget(self.proj_dt_edit)
        range_layout.addWidget(QLabel("s"))
        range_layout.addSpacing(5)
        range_layout.addWidget(QLabel("Split:"))
        self.split_ratio_edit = QLineEdit("65")
        self.split_ratio_edit.setMaximumWidth(35)
        self.split_ratio_edit.setToolTip("Left panel width % (rest goes to projections)")
        self.split_ratio_edit.editingFinished.connect(self._on_range_changed)
        range_layout.addWidget(self.split_ratio_edit)
        range_layout.addWidget(QLabel("%"))
        self.proj_btn = QPushButton("Project")
        self.proj_btn.clicked.connect(self._project_and_redraw)
        range_layout.addWidget(self.proj_btn)
        range_layout.addSpacing(5)
        range_layout.addWidget(QLabel("View:"))
        self.proj_combo = QComboBox()
        self.proj_combo.setMinimumWidth(80)
        self.proj_combo.currentIndexChanged.connect(self._on_proj_combo_changed)
        range_layout.addWidget(self.proj_combo)

        self.find_peaks_btn = QPushButton("Find Peaks")
        self.find_peaks_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 2px 8px; border-radius: 3px;")
        self.find_peaks_btn.clicked.connect(self._find_peaks_projections)
        range_layout.addWidget(self.find_peaks_btn)
        range_layout.addWidget(QLabel("dist(MHz):"))
        self.peak_dist_edit = QLineEdit("0.003")
        self.peak_dist_edit.setMaximumWidth(40)
        range_layout.addWidget(self.peak_dist_edit)
        range_layout.addWidget(QLabel("width:"))
        self.bg_width_edit = QLineEdit("0.001")
        self.bg_width_edit.setMaximumWidth(40)
        self.bg_width_edit.setToolTip("Integration width for both peak and background (MHz)")
        range_layout.addWidget(self.bg_width_edit)
        range_layout.addWidget(QLabel("gap:"))
        self.bg_gap_edit = QLineEdit("0.001")
        self.bg_gap_edit.setMaximumWidth(30)
        self.bg_gap_edit.setToolTip("Gap between peak interval and bg interval (MHz)")
        range_layout.addWidget(self.bg_gap_edit)

        self.fit_har_btn = QPushButton("Fit Har")
        self.fit_har_btn.setStyleSheet("background-color: #E91E63; color: white; padding: 2px 8px; border-radius: 3px;")
        self.fit_har_btn.clicked.connect(self._fit_harmonics)
        range_layout.addWidget(self.fit_har_btn)
        range_layout.addWidget(QLabel("h_off:"))
        self.har_offset_edit = QLineEdit("0")
        self.har_offset_edit.setMaximumWidth(30)
        self.har_offset_edit.setToolTip("Global harmonic number offset (added to all projections)")
        range_layout.addWidget(self.har_offset_edit)
        range_layout.addWidget(QLabel("per:"))
        self.har_per_offset_edit = QLineEdit()
        self.har_per_offset_edit.setPlaceholderText("e.g. 0:3 2:-1")
        self.har_per_offset_edit.setMaximumWidth(90)
        self.har_per_offset_edit.setToolTip("Per-projection offset: proj_idx:offset  (space separated)")
        range_layout.addWidget(self.har_per_offset_edit)
        self.norm_area_btn = QPushButton("Plot Norm Area")
        self.norm_area_btn.setStyleSheet("background-color: #00BCD4; color: white; padding: 2px 8px; border-radius: 3px;")
        self.norm_area_btn.clicked.connect(self._plot_normalized_areas)
        range_layout.addWidget(self.norm_area_btn)
        self.norm_ref_combo = QComboBox()
        self.norm_ref_combo.addItems(["Ref: 1st peak", "Ref: last peak", "Ref: Max Area"])
        self.norm_ref_combo.setMinimumWidth(100)
        self.norm_ref_combo.setToolTip("Reference peak for normalization within each projection")
        range_layout.addWidget(self.norm_ref_combo)
        range_layout.addSpacing(5)
        range_layout.addWidget(QLabel("Har#:"))
        self.har_vs_time_edit = QLineEdit()
        self.har_vs_time_edit.setMaximumWidth(40)
        self.har_vs_time_edit.setPlaceholderText("731")
        range_layout.addWidget(self.har_vs_time_edit)
        self.har_vs_time_btn = QPushButton("Area vs Time")
        self.har_vs_time_btn.setStyleSheet("background-color: #009688; color: white; padding: 2px 8px; border-radius: 3px;")
        self.har_vs_time_btn.clicked.connect(self._plot_harmonic_vs_time)
        range_layout.addWidget(self.har_vs_time_btn)
        self.sc_fit_btn = QPushButton("Self-consistent")
        self.sc_fit_btn.setStyleSheet("background-color: #673AB7; color: white; padding: 2px 8px; border-radius: 3px;")
        self.sc_fit_btn.clicked.connect(self._self_consistent_fit)
        range_layout.addWidget(self.sc_fit_btn)

        main_layout.addWidget(range_frame)

        # ── Display controls ──
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addStretch()
        self.logz_checkbox = QCheckBox("Log Z")
        self.logz_checkbox.setChecked(False)
        self.logz_checkbox.toggled.connect(self._on_logz_toggled)
        ctrl_layout.addWidget(self.logz_checkbox)
        self.show_thresh_checkbox = QCheckBox("Show Threshold")
        self.show_thresh_checkbox.setChecked(True)
        self.show_thresh_checkbox.toggled.connect(self._on_show_thresh_toggled)
        ctrl_layout.addWidget(self.show_thresh_checkbox)
        self.show_proj_checkbox = QCheckBox("Show Projections")
        self.show_proj_checkbox.setChecked(False)
        self.show_proj_checkbox.toggled.connect(self._on_show_proj_toggled)
        ctrl_layout.addWidget(self.show_proj_checkbox)
        ctrl_layout.addSpacing(8)
        ctrl_layout.addWidget(QLabel("Font:"))
        self.font_scale_edit = QLineEdit("1.0")
        self.font_scale_edit.setMaximumWidth(40)
        ctrl_layout.addWidget(self.font_scale_edit)
        self.font_scale_btn = QPushButton("Scale")
        self.font_scale_btn.setStyleSheet("background-color: #607D8B; color: white; border-radius: 3px; padding: 1px 6px;")
        self.font_scale_btn.clicked.connect(self._on_font_scale)
        ctrl_layout.addWidget(self.font_scale_btn)
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)

        # ── Plot area ──
        self.canvas = MatplotlibCanvas(self, width=8, height=7, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, 1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Config persistence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_config(self):
        """Save all panel parameters to JSON config file."""
        try:
            cfg = {}
            # File paths
            cfg['time_file'] = self.time_edit.text().strip()
            cfg['voltage_file'] = self.voltage_edit.text().strip()
            cfg['threshold_path'] = self.thresh_path_edit.text().strip()

            # Range controls
            for k in ('xmin_edit','xmax_edit','ymin_edit','ymax_edit',
                      'proj_offset_edit','proj_dt_edit','split_ratio_edit',
                      'peak_dist_edit','bg_width_edit','bg_gap_edit',
                      'baseline_l_edit','baseline_ratio_edit',
                      'font_scale_edit','har_offset_edit', 'har_vs_time_edit'):
                w = getattr(self, k, None)
                if w is not None:
                    cfg[k] = w.text()

            cfg['har_per_offset'] = self.har_per_offset_edit.text().strip()
            cfg['norm_ref'] = self.norm_ref_combo.currentText() if hasattr(self, 'norm_ref_combo') else "Ref: 1st peak"

            # Checkboxes
            cfg['log_z'] = self.logz_checkbox.isChecked() if hasattr(self, 'logz_checkbox') else False
            cfg['show_thresh'] = self.show_thresh_checkbox.isChecked() if hasattr(self, 'show_thresh_checkbox') else True
            cfg['show_proj'] = self.show_proj_checkbox.isChecked() if hasattr(self, 'show_proj_checkbox') else False

            # Resonance params
            if self._last_res_params is not None:
                cfg['A0'], cfg['Q'], cfg['f_sys'] = self._last_res_params

            with open(self._config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def _load_config(self):
        """Load panel parameters from JSON config file."""
        if not os.path.exists(self._config_path):
            return
        try:
            with open(self._config_path, 'r') as f:
                cfg = json.load(f)

            # File paths
            for k, attr in [('time_file','time_edit'),('voltage_file','voltage_edit'),
                            ('threshold_path','thresh_path_edit')]:
                if k in cfg:
                    w = getattr(self, attr, None)
                    if w is not None:
                        w.setText(cfg[k])

            # Numeric text fields
            for k in ('xmin_edit','xmax_edit','ymin_edit','ymax_edit',
                      'proj_offset_edit','proj_dt_edit','split_ratio_edit',
                      'peak_dist_edit','bg_width_edit','bg_gap_edit',
                      'baseline_l_edit','baseline_ratio_edit',
                      'font_scale_edit','har_offset_edit', 'har_vs_time_edit'):
                if k in cfg:
                    w = getattr(self, k, None)
                    if w is not None:
                        w.setText(cfg[k])

            if 'har_per_offset' in cfg:
                self.har_per_offset_edit.setText(cfg['har_per_offset'])
            if 'norm_ref' in cfg:
                idx = self.norm_ref_combo.findText(cfg['norm_ref'])
                if idx >= 0:
                    self.norm_ref_combo.setCurrentIndex(idx)

            # Checkboxes
            if 'log_z' in cfg:
                self.logz_checkbox.setChecked(cfg['log_z'])
            if 'show_thresh' in cfg:
                self.show_thresh_checkbox.setChecked(cfg['show_thresh'])
            if 'show_proj' in cfg:
                self.show_proj_checkbox.setChecked(cfg['show_proj'])

            # Font scale
            if 'font_scale_edit' in cfg:
                try:
                    self._font_scale = float(cfg['font_scale_edit'])
                except ValueError:
                    pass

            # Resonance params
            if all(k in cfg for k in ('A0','Q','f_sys')):
                self._last_res_params = (cfg['A0'], cfg['Q'], cfg['f_sys'])
        except Exception:
            pass


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # File dialogs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # File dialogs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _browse_time_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Experiment Time File", "",
            "Supported (*.npz *.csv *.txt *.bin_time *.bin_amp *.root);;All Files (*)"
        )
        if path:
            self.time_edit.setText(path)

    def _browse_voltage_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Voltage File", "",
            "Supported (*.npz *.csv *.txt);;All Files (*)"
        )
        if path:
            self.voltage_edit.setText(path)

    def _edit_voltage_file(self):
        """Open an in-app text editor dialog to view/edit the voltage file."""
        path = self.voltage_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Warning", "No voltage file selected — browse or type a path first")
            return
        try:
            with open(path, 'r') as f:
                content = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file:\n{str(e)}")
            return

        basename = path.replace('\\', '/').split('/')[-1]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit: {basename}")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        text_edit = QPlainTextEdit()
        text_edit.setPlainText(content)
        text_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(text_edit)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 4px 16px; border-radius: 3px;")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("padding: 4px 16px; border-radius: 3px;")

        def on_save():
            try:
                with open(path, 'w') as f:
                    f.write(text_edit.toPlainText())
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(dialog, "Error", f"Could not save file:\n{str(e)}")

        save_btn.clicked.connect(on_save)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        dialog.exec_()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Threshold profile
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_thresh_browse(self):
        """Browse and load a threshold profile CSV."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Threshold Profile", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self.thresh_path_edit.setText(path)
        self._load_threshold_profile(path)
        self._redraw_projections()

    def _load_threshold_profile(self, path):
        """Load threshold profile from CSV (freq, value columns)."""
        try:
            data = np.genfromtxt(path, delimiter=',', autostrip=True)
            if data.ndim == 2 and data.shape[1] >= 2:
                self.threshold_profile_freqs = data[:, 0]
                self.threshold_profile_vals = data[:, 1]
            else:
                data = np.genfromtxt(path, autostrip=True)
                if data.ndim == 2 and data.shape[1] >= 2:
                    self.threshold_profile_freqs = data[:, 0]
                    self.threshold_profile_vals = data[:, 1]
                else:
                    raise ValueError("Need 2 columns")
            self._thresh_profile_path = path
            self.thresh_path_edit.setText(path)
            print(f"✅ Threshold profile loaded: {len(self.threshold_profile_freqs)} points")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load threshold profile:\n{e}")
            self.threshold_profile_freqs = None
            self.threshold_profile_vals = None

    def _auto_save_threshold(self):
        """Save current threshold profile to CSV (creates if needed)."""
        if self.threshold_profile_freqs is None or self.threshold_profile_vals is None:
            return
        path = self._thresh_profile_path
        if not path:
            import os
            path = os.path.join(os.getcwd(), "height_thresh.csv")
            self._thresh_profile_path = path
            self.thresh_path_edit.setText(path)
        try:
            np.savetxt(path, np.column_stack((self.threshold_profile_freqs,
                                              self.threshold_profile_vals)),
                       delimiter=',', fmt='%.6f,%.6e',
                       header='Frequency(MHz),Amplitude', comments='')
            print(f"✅ Threshold profile saved to {path}")
        except Exception as e:
            print(f"⚠️ Could not save threshold profile: {e}")

    def _on_toggle_threshold_click(self):
        """Toggle click-to-set-threshold mode on the plot."""
        # Disconnect any previous connection first
        if hasattr(self, '_thresh_cid') and self._thresh_cid is not None:
            self.canvas.mpl_disconnect(self._thresh_cid)
            self._thresh_cid = None

        self._thresh_click_mode = not self._thresh_click_mode
        if self._thresh_click_mode:
            self.thresh_click_btn.setText("Click Threshold: ON")
            self.thresh_click_btn.setStyleSheet("background-color: #90EE90;")
            self._thresh_cid = self.canvas.mpl_connect(
                'button_press_event', self._on_thresh_plot_click)
        else:
            self.thresh_click_btn.setText("Start Click Threshold")
            self.thresh_click_btn.setStyleSheet("")
            self._auto_save_threshold()

    def _on_thresh_plot_click(self, event):
        """Handle click on the spectrogram plot to add threshold points."""
        if event.inaxes is None:
            return
        # Get frequency and amplitude at click position
        freq_click = event.xdata
        amp_click = event.ydata

        # Append to threshold profile
        if self.threshold_profile_freqs is None:
            self.threshold_profile_freqs = np.array([freq_click])
            self.threshold_profile_vals = np.array([amp_click])
        else:
            self.threshold_profile_freqs = np.append(self.threshold_profile_freqs, freq_click)
            self.threshold_profile_vals = np.append(self.threshold_profile_vals, amp_click)

        # Sort by frequency
        order = np.argsort(self.threshold_profile_freqs)
        self.threshold_profile_freqs = self.threshold_profile_freqs[order]
        self.threshold_profile_vals = self.threshold_profile_vals[order]

        print(f"✅ Threshold point added: f={freq_click:.2f} MHz, amp={amp_click:.4f}")
        self._auto_save_threshold()
        self._redraw_projections()

    def _on_thresh_update(self):
        """Re-read threshold CSV from the path and redraw projections."""
        path = self.thresh_path_edit.text().strip()
        if path:
            self._load_threshold_profile(path)
        self._redraw_projections()
        self._save_config()

    def _get_threshold_curve(self):
        """Interpolate threshold profile onto self.freq_data grid.

        Returns a 1D array of threshold values, or None if no profile is loaded.
        """
        if self.threshold_profile_freqs is None or self.freq_data is None:
            return None
        return np.interp(self.freq_data, self.threshold_profile_freqs, self.threshold_profile_vals,
                         left=self.threshold_profile_vals[0], right=self.threshold_profile_vals[-1])

    def _on_thresh_clear(self):
        """Clear the threshold profile."""
        self.threshold_profile_freqs = None
        self.threshold_profile_vals = None
        self._thresh_profile_path = ""
        self.thresh_path_edit.clear()
        print("✅ Threshold profile cleared")
        self._redraw_projections()
        self._save_config()

    def _remove_baseline_projections(self):
        """Remove baseline from all projections using BrPLS."""
        if self._projections is None or len(self._projections) == 0:
            QMessageBox.warning(self, "Warning", "No projections computed yet — click Project first")
            return
        try:
            l_val = float(self.baseline_l_edit.text())
            ratio_val = float(self.baseline_ratio_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid l or ratio value")
            return

        freq = self.freq_data
        proj_list_bl = []
        proj_list_clean = []

        for i, proj in enumerate(self._projections):
            try:
                baseline = NONPARAMS_EST(proj).pls('BrPLS', l=l_val, ratio=ratio_val)
                clean = proj - baseline
                proj_list_bl.append(baseline)
                proj_list_clean.append(clean)
            except Exception as e:
                self._write_debug(f"Baseline removal failed for projection {i}: {e}")
                proj_list_bl.append(np.zeros_like(proj))
                proj_list_clean.append(proj)

        self._projections_baseline = np.array(proj_list_bl)
        self._projections_clean = np.array(proj_list_clean)
        # Reset peaks since data changed
        self._projections_peaks = None
        self._projections_peak_data = None
        self._projections_peak_masks = None
        self._projections_har_fit = None
        print(f"✅ Baseline removed from {len(proj_list_clean)} projections (l={l_val}, ratio={ratio_val})")
        self._redraw_projections()
        self._save_config()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Data loading
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _load_and_plot(self):
        time_path = self.time_edit.text().strip()
        volt_path = self.voltage_edit.text().strip()

        if not time_path:
            QMessageBox.warning(self, "Warning", "Please select an experiment time file first")
            return

        try:
            self._write_debug(f"DEBUG: _load_and_plot: time_path='{time_path}', volt_path='{volt_path}'")
            self._load_time_file(time_path)
            if volt_path:
                self._load_voltage_file(volt_path)
                # Auto-compute projections when both spectrogram and voltage exist
                if self.spectrogram_data is not None:
                    self._compute_projections()
            self._draw_plots()
            # Emit signal so the parent can save paths immediately
            self.paths_changed.emit(time_path, volt_path)
            self._save_config()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Data load or plot failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_voltage_only(self):
        """Re-read the voltage file and redraw markers without re-loading the spectrogram."""
        volt_path = self.voltage_edit.text().strip()
        if not volt_path:
            QMessageBox.warning(self, "Warning", "Please select a voltage file first")
            return
        if self.spectrogram_data is None and self.freq_data is None:
            QMessageBox.warning(self, "Warning", "No spectrogram data loaded — load a time file first")
            return
        try:
            self._load_voltage_file(volt_path)
            if self.spectrogram_data is not None:
                self._compute_projections()
            self._draw_plots()
            self.paths_changed.emit(self.time_edit.text().strip(), volt_path)
            self._save_config()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Voltage update failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _on_logz_toggled(self, checked):
        if self.spectrogram_data is not None or self.freq_data is not None:
            self._draw_plots()

    def _on_show_thresh_toggled(self, checked):
        """Redraw when Show Threshold toggles."""
        if self._projections is not None:
            self._redraw_projections()

    def _on_show_proj_toggled(self, checked):
        """Toggle projection panel visibility."""
        if self._projections is not None and len(self._projections) > 0:
            self._draw_plots()

    def _load_time_file(self, path):
        """Load experiment time file, trying various formats."""
        # ── .npz ──
        if path.lower().endswith('.npz'):
            data = np.load(path)
            time_keys = ['time', 't', 'times', 'timestamp', 'timestamps', 'Time', 'T']
            freq_keys = ['freq', 'frequency', 'f', 'frequencies', 'Freq', 'Frequency']
            spec_keys = ['spectrogram', 'spec', 'Sxx', 'data', 'z', 'Z']

            time_arr = None
            freq_arr = None
            spec_arr = None

            for k in time_keys:
                if k in data:
                    time_arr = data[k]
                    break
            for k in freq_keys:
                if k in data:
                    freq_arr = data[k]
                    break
            for k in spec_keys:
                if k in data:
                    spec_arr = data[k]
                    break

            # Fallback: use positional order
            if time_arr is None or freq_arr is None or spec_arr is None:
                arrays = [data[k] for k in data.files]
                if len(arrays) >= 3:
                    if time_arr is None:
                        time_arr = arrays[0].ravel() if arrays[0].ndim <= 1 else arrays[0]
                    if freq_arr is None:
                        freq_arr = arrays[1].ravel() if arrays[1].ndim <= 1 else arrays[1]
                    if spec_arr is None:
                        spec_arr = arrays[2] if arrays[2].ndim == 2 else arrays[2]
                elif len(arrays) == 2:
                    if time_arr is None and freq_arr is None:
                        time_arr = arrays[0].ravel() if arrays[0].ndim <= 1 else arrays[0]
                        freq_arr = arrays[1].ravel() if arrays[1].ndim <= 1 else arrays[1]

            self.time_data = time_arr
            self.freq_data = freq_arr
            self.spectrogram_data = spec_arr
            data.close()

        # ── .csv / .txt ──
        elif path.lower().endswith(('.csv', '.txt')):
            try:
                arr = np.loadtxt(path)
            except Exception:
                arr = np.genfromtxt(path, delimiter=',', missing_values='', filling_values=0)

            if arr.ndim == 2:
                if arr.shape[1] >= 3:
                    self.time_data = arr[:, 0]
                    self.freq_data = arr[:, 1]
                    self.spectrogram_data = arr[:, 2:]
                elif arr.shape[1] == 2:
                    self.time_data = arr[:, 0]
                    self.freq_data = arr[:, 1]
            elif arr.ndim == 1:
                self.time_data = arr

        # ── .bin_time / .bin_amp (Puyuan format) ──
        elif path.lower().endswith(('.bin_time', '.bin_amp')):
            try:
                raw = np.fromfile(path, dtype=np.float64)
            except Exception:
                raw = np.fromfile(path, dtype=np.float32)
            self.time_data = raw

        # ── .root / cached .npz ──
        elif path.lower().endswith('.root'):
            # Check if a cached .npz exists in cwd
            basename = os.path.splitext(os.path.basename(path))[0]
            cache_path = os.path.join(os.getcwd(), basename + "_spectrogram.npz")
            if os.path.exists(cache_path):
                self._write_debug(f"DEBUG: loading cached npz: {cache_path}")
                data = np.load(cache_path)
                self.time_data = data['time_data']
                self.freq_data = data['freq_data']
                self.spectrogram_data = data['spectrogram_data']
                self._time_offset = float(data['time_offset'])
                data.close()
                print(f"✅ Loaded cached spectrogram from {cache_path}")
            else:
                self._load_root_file(path)

        if self.time_data is None:
            raise ValueError("Cannot recognise time/frequency data — check file format")

        # Ensure 1D
        if self.time_data is not None and self.time_data.ndim > 1:
            self.time_data = self.time_data.ravel()
        if self.freq_data is not None and self.freq_data.ndim > 1:
            self.freq_data = self.freq_data.ravel()

        print(f"✅ Experiment time file loaded:")
        if self.time_data is not None:
            print(f"   Time data: shape={self.time_data.shape}")
        if self.freq_data is not None:
            print(f"   Freq data: shape={self.freq_data.shape}")
        if self.spectrogram_data is not None:
            print(f"   Spectrogram: shape={self.spectrogram_data.shape}")

    def _write_debug(self, msg: str):
        """Write debug message to afc_debug.log in current directory."""
        try:
            with open("afc_debug.log", "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def _clear_debug_log(self):
        """Clear the debug log file."""
        try:
            with open("afc_debug.log", "w") as f:
                f.write("=== AFC Calculator Debug Log ===\n")
        except Exception:
            pass

    def _load_voltage_file(self, path):
        """Load voltage file — manual line-by-line parsing."""
        self._write_debug(f"DEBUG: _load_voltage_file called with path = '{path}'")

        # ── .npz ──
        if path.lower().endswith('.npz'):
            data = np.load(path)
            volt_keys = ['voltage', 'volt', 'v', 'V', 'Voltage', 'data']
            time_keys = ['time', 't', 'times', 'Time']
            volt_arr = None; time_arr = None
            for k in volt_keys:
                if k in data: volt_arr = data[k]; break
            for k in time_keys:
                if k in data: time_arr = data[k]; break
            if volt_arr is None: volt_arr = data[data.files[0]]
            self.voltage_data = volt_arr.ravel() if volt_arr is not None else None
            self.voltage_time = time_arr.ravel() if time_arr is not None else None
            data.close()
            return

        # ── Text file: manual line-by-line ──
        try:
            with open(path, 'r') as f:
                raw = f.read()
        except Exception as e:
            self._write_debug(f"DEBUG: cannot open file: {e}")
            return

        lines = raw.splitlines()
        self._write_debug(f"DEBUG: read {len(lines)} lines")
        if lines:
            self._write_debug(f"DEBUG: line 0 = {repr(lines[0])}")
        if len(lines) > 1:
            self._write_debug(f"DEBUG: line 1 = {repr(lines[1])}")
        if len(lines) > 2:
            self._write_debug(f"DEBUG: line 2 = {repr(lines[2])}")

        times = []
        voltages = []
        header_skipped = False

        for line in lines:
            s = line.strip()
            if not s:
                continue
            # Try tab, then comma, then whitespace split
            if '\t' in s:
                parts = s.split('\t')
            elif ',' in s:
                parts = s.split(',')
            else:
                parts = s.split()
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not header_skipped:
                try:
                    float(a); float(b)
                except ValueError:
                    header_skipped = True
                    continue
                header_skipped = True
            try:
                times.append(float(a))
                voltages.append(float(b))
            except ValueError:
                continue

        self._write_debug(f"DEBUG: parsed {len(times)} data rows")
        if times:
            self._write_debug(f"DEBUG: times = {times}")
            self.voltage_time = np.array(times)
            self.voltage_data = np.array(voltages)
        else:
            self._write_debug("DEBUG: FAILED to parse any numeric data")

        # ── Debug output ──
        if self.voltage_time is not None:
            self._write_debug(f"DEBUG: parsed voltage_time (first 10) = {self.voltage_time[:min(10, len(self.voltage_time))]}")
            self._write_debug(f"DEBUG: parsed voltage_data (first 10) = {self.voltage_data[:min(10, len(self.voltage_data))]}")
        elif self.voltage_data is not None:
            self._write_debug(f"DEBUG: parsed voltage_data only, shape = {self.voltage_data.shape}")
        else:
            self._write_debug(f"DEBUG: FAILED to parse voltage file")

        if self.voltage_data is not None:
            print(f"✅ Voltage file loaded: shape={self.voltage_data.shape}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ROOT file reading
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _load_root_file(self, path):
        """Extract TH2F data from a ROOT file (TCanvas → TH2).

        ROOT convention: X axis = frequency, Y axis = time.
        Time axis is normalised so that it starts at 0.
        """
        try:
            import ROOT
        except ImportError:
            raise ImportError("ROOT (PyROOT) is required to read .root files")

        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            raise ValueError(f"Cannot open ROOT file: {path}")

        # Find first TCanvas
        canvas = None
        for key in f.GetListOfKeys():
            if key.GetClassName() in ('TCanvas', 'TCanvasE', 'TCanvasImp'):
                canvas = f.Get(key.GetName())
                break

        if not canvas:
            for key in f.GetListOfKeys():
                if 'TH2' in key.GetClassName():
                    hist = f.Get(key.GetName())
                    break
            else:
                f.Close()
                raise ValueError("No TCanvas or TH2 found in ROOT file")
        else:
            hist = None
            for obj in canvas.GetListOfPrimitives():
                if 'TH2' in obj.ClassName():
                    hist = obj
                    break
            if not hist:
                f.Close()
                raise ValueError("No TH2 found in TCanvas")

        # ── Extract raw data ──
        npx = hist.GetNbinsX()          # frequency bins
        npy = hist.GetNbinsY()          # time bins

        # bin centre arrays
        x_centers = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, npx + 1)])
        y_centers = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, npy + 1)])

        # Full matrix: GetBinContent(ix, iy)
        # We want spec[row, col] = spec[time_idx, freq_idx]
        spec = np.zeros((npy, npx), dtype=np.float64)
        for i in range(1, npx + 1):
            for j in range(1, npy + 1):
                spec[j - 1, i - 1] = hist.GetBinContent(i, j)

        f.Close()

        # ── Assign: X = frequency, Y = time ──
        self.freq_data = x_centers
        self.time_data = y_centers

        # ── Normalise time to start at 0 ──
        self._time_offset = 0.0
        if len(self.time_data) > 0:
            self._time_offset = float(self.time_data.min())
            self.time_data = self.time_data - self._time_offset

        self.spectrogram_data = spec  # shape (time_bins, freq_bins)

        print(f"✅ ROOT file loaded:")
        print(f"   TH2: {hist.GetName()} ({hist.GetTitle()})")
        print(f"   Frequency: {npx} bins, "
              f"range [{self.freq_data[0]:.4e}–{self.freq_data[-1]:.4e}] Hz "
              f"({self.freq_data[0]/1e6:.4f}–{self.freq_data[-1]/1e6:.4f} MHz)")
        print(f"   Time:      {npy} bins, "
              f"range [{self.time_data[0]:.4e}–{self.time_data[-1]:.4e}] s")
        print(f"   Matrix: {self.spectrogram_data.shape}")

        # ── Save as cached .npz in current directory ──
        try:
            base_in = os.path.splitext(os.path.basename(path))[0]
            cache_path = os.path.join(os.getcwd(), base_in + "_spectrogram.npz")
            np.savez(cache_path,
                     time_data=self.time_data,
                     freq_data=self.freq_data,
                     spectrogram_data=self.spectrogram_data,
                     time_offset=self._time_offset)
            print(f"✅ Saved cached spectrogram to {cache_path}")
        except Exception as e:
            print(f"⚠️ Could not save cached .npz: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Range controls
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_figsize_changed(self):
        """Resize the figure canvas based on width×height text boxes."""
        try:
            w = float(self.fig_width_edit.text())
            h = float(self.fig_height_edit.text())
        except ValueError:
            return
        w = max(3, min(w, 40))
        h = max(2, min(h, 30))
        self.canvas.fig.set_size_inches(w, h, forward=True)
        self.canvas.draw_idle()

    def _on_font_scale(self):
        """Scale all plot font sizes proportionally, then redraw."""
        try:
            s = float(self.font_scale_edit.text())
        except ValueError:
            return
        s = max(0.3, min(s, 5.0))
        self._font_scale = s
        base = 10 * s
        matplotlib.rcParams.update({
            'font.size':               base,
            'axes.labelsize':          base,
            'axes.titlesize':          base * 1.2,
            'xtick.labelsize':         base * 0.9,
            'ytick.labelsize':         base * 0.9,
            'legend.fontsize':         base * 0.85,
            'figure.titlesize':        base * 1.2,
        })
        # Update constrained_layout padding to leave room for larger labels
        self.canvas.fig.set_constrained_layout_pads(h_pad=0.04 * s, hspace=0)
        self._save_config()
        if self.spectrogram_data is not None or self.freq_data is not None:
            self._draw_plots()

    def _reset_range(self):
        """Clear all range inputs and redraw."""
        for w in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            w.clear()
        if self.spectrogram_data is not None or self.freq_data is not None:
            self._draw_plots()

    def _on_range_changed(self):
        """Redraw when any range input is edited."""
        if self.spectrogram_data is not None or self.freq_data is not None:
            self._draw_plots()

    def _apply_axis_limits(self, ax):
        """Apply user-specified axis limits from the range inputs."""
        try:
            xmin = float(self.xmin_edit.text()) if self.xmin_edit.text().strip() else None
        except ValueError:
            xmin = None
        try:
            xmax = float(self.xmax_edit.text()) if self.xmax_edit.text().strip() else None
        except ValueError:
            xmax = None
        try:
            ymin = float(self.ymin_edit.text()) if self.ymin_edit.text().strip() else None
        except ValueError:
            ymin = None
        try:
            ymax = float(self.ymax_edit.text()) if self.ymax_edit.text().strip() else None
        except ValueError:
            ymax = None

        if xmin is not None or xmax is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Projection computation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_projections(self, offset_s=None, dt_s=None):
        """For each voltage event, extract frequency projection in [t+offset, t+offset+dt].

        Stores results in self._projections and self._proj_voltages.
        Resets baseline-removed and peak-finding caches.
        """
        self._projections = None
        self._projections_baseline = None
        self._projections_clean = None
        self._projections_peaks = None
        self._projections_peak_data = None
        self._proj_voltages = None
        self._proj_nframes = None
        self._projections_frame_data = None

        if self.spectrogram_data is None or self.time_data is None:
            return
        if self.voltage_time is None or len(self.voltage_time) == 0:
            return

        try:
            offset_s = float(self.proj_offset_edit.text()) if offset_s is None else offset_s
            dt_s = float(self.proj_dt_edit.text()) if dt_s is None else dt_s
        except (ValueError, AttributeError):
            offset_s, dt_s = 5.0, 20.0

        t_arr = self.time_data
        freq = self.freq_data  # in MHz
        spec = self.spectrogram_data  # shape (time_bins, freq_bins)

        proj_list = []
        volt_list = []
        nframe_list = []
        frame_data_list = []

        for i, vt in enumerate(self.voltage_time):
            t_start = vt + offset_s
            t_end = t_start + dt_s
            idx = np.where((t_arr >= t_start) & (t_arr <= t_end))[0]
            if len(idx) < 2:
                # Not enough time bins — skip
                self._write_debug(f"DEBUG project: vt={vt}, t_start={t_start}, t_end={t_end} → only {len(idx)} bins, skip")
                continue
            # Sum over time bins → 1D frequency spectrum
            proj = spec[idx, :].sum(axis=0)
            proj_list.append(proj)
            nframe_list.append(len(idx))
            frame_data_list.append(spec[idx, :].copy())  # store per-frame data for area statistics
            volt_val = self.voltage_data[i] if self.voltage_data is not None and i < len(self.voltage_data) else 0
            volt_list.append(volt_val)
            self._write_debug(f"DEBUG project: vt={vt:.1f}, range=[{t_start:.1f},{t_end:.1f}], idx={len(idx)} bins, sum={proj.sum():.1f}")

        if proj_list:
            self._projections = np.array(proj_list)
            self._proj_voltages = np.array(volt_list)
            self._proj_nframes = np.array(nframe_list)
            self._projections_frame_data = frame_data_list
            self._write_debug(f"DEBUG project: {len(proj_list)} projections, each {len(proj_list[0])} freq bins")
            # Populate projection combo box
            self.proj_combo.blockSignals(True)
            self.proj_combo.clear()
            self.proj_combo.addItem("All")
            for i in range(len(proj_list)):
                label = f"#{i}"
                if self.voltage_time is not None and i < len(self.voltage_time):
                    label = f"t={self.voltage_time[i]:.0f}s"
                self.proj_combo.addItem(label)
            self.proj_combo.blockSignals(False)
        else:
            self._proj_voltage_time = None

    def _on_proj_combo_changed(self, idx):
        """Redraw when projection selection changes."""
        if self._projections is not None and len(self._projections) > 0:
            self._draw_plots()

    def _project_and_redraw(self):
        """(Re)compute projections and redraw."""
        if self.spectrogram_data is None or self.voltage_time is None:
            QMessageBox.warning(self, "Warning", "Load spectrogram and voltage data first")
            return
        self._compute_projections()
        self._draw_plots()
        self._save_config()

    def _detect_peaks_one(self, freq, amp, frame_amp=None):
        """Replicate ImportData.detect_peaks_and_widths logic on a single projection.

        Uses threshold profile + prominence + distance matching the main GUI's settings.
        If frame_amp (n_frames × n_freqs) is provided, also computes per-frame net area
        and its uncertainty from frame-to-frame variation.

        Returns (peaks, peak_freqs, peak_heights, peak_widths_freq, peak_areas,
                 peak_areas_err, peak_means, peak_stds, left_idxs, right_idxs,
                 bg_levels, bg_left_idxs, bg_right_idxs).
        """
        # 1) Height threshold
        if self.threshold_profile_freqs is not None and self.threshold_profile_vals is not None:
            height_thresh = np.interp(
                freq,
                self.threshold_profile_freqs,
                self.threshold_profile_vals,
                left=self.threshold_profile_vals[0],
                right=self.threshold_profile_vals[-1]
            )
            height_thresh = np.nan_to_num(height_thresh, nan=0.0, posinf=0.0, neginf=0.0)
            height_thresh = np.maximum(height_thresh, 0.0)
        else:
            # Fallback: 20 % of max
            height_thresh = np.full_like(amp, np.max(amp) * 0.2, dtype=float)

        # 2) Same parameters as detect_peaks_and_widths
        df = max(freq[1] - freq[0], 1e-12)
        try:
            min_dist = max(1.0, float(self.peak_dist_edit.text()) / df)
        except (ValueError, AttributeError):
            min_dist = 3.0
        min_prom  = np.maximum(height_thresh * 0.3, 0.0)
        min_w     = 1

        peaks, props = find_peaks(
            amp,
            height=height_thresh,
            distance=min_dist,
            prominence=min_prom,
            width=min_w,
        )

        peak_freqs = freq[peaks]
        peak_heights = props['peak_heights'] if 'peak_heights' in props else amp[peaks]
        peak_widths_freq = np.zeros_like(peak_freqs)
        peak_areas = np.zeros_like(peak_freqs)
        peak_areas_err = np.zeros_like(peak_freqs)
        peak_means = np.zeros_like(peak_freqs)
        peak_stds = np.zeros_like(peak_freqs)
        left_idxs = np.zeros_like(peaks, dtype=int)
        right_idxs = np.zeros_like(peaks, dtype=int)
        bg_levels = np.zeros_like(peak_freqs)
        bg_left_idxs = np.zeros_like(peaks, dtype=int)
        bg_right_idxs = np.zeros_like(peaks, dtype=int)
        if len(peaks) > 0:
            try:
                # Read shared integration width from text box
                df = max(freq[1] - freq[0], 1e-12)
                try:
                    half_mhz = max(0.0, float(self.bg_width_edit.text())) / 2.0
                except (ValueError, AttributeError):
                    half_mhz = 0.0005
                half_bins = max(1, int(round(half_mhz / df)))
                try:
                    gap_mhz = max(0.0, float(self.bg_gap_edit.text()))
                except (ValueError, AttributeError):
                    gap_mhz = 0.001
                gap = max(1, int(round(gap_mhz / df)))

                for j, p in enumerate(peaks):
                    li = max(0, p - half_bins)
                    ri = min(len(freq) - 1, p + half_bins)
                    left_idxs[j] = li
                    right_idxs[j] = ri
                    peak_widths_freq[j] = freq[ri] - freq[li]
                    peak_areas[j] = np.trapz(amp[li:ri+1], freq[li:ri+1])
                    f_region = freq[li:ri+1]
                    a_region = amp[li:ri+1]
                    a_sum = np.sum(a_region)
                    if a_sum > 0:
                        peak_means[j] = np.average(f_region, weights=a_region)
                        variance = np.average((f_region - peak_means[j])**2, weights=a_region)
                        peak_stds[j] = np.sqrt(variance)
                    else:
                        peak_means[j] = f_region[len(f_region)//2]
                        peak_stds[j] = 0.0

                    # Background region: same width, placed left with a gap
                    bg_li = li - half_bins*2 - gap
                    bg_ri = bg_li + half_bins*2
                    if bg_li < 0 or bg_ri >= len(freq):
                        bg_li = min(len(freq) - 1 - half_bins*2, ri + gap)
                        bg_ri = bg_li + half_bins*2
                    bg_levels[j] = np.trapz(amp[bg_li:bg_ri+1], freq[bg_li:bg_ri+1])
                    bg_left_idxs[j] = int(bg_li)
                    bg_right_idxs[j] = int(bg_ri)

                    # Per-frame area statistics (error from frame-to-frame variation)
                    if frame_amp is not None:
                        nf = frame_amp.shape[0]
                        pk_per_frame = np.array([
                            np.trapz(frame_amp[k, li:ri+1], freq[li:ri+1])
                            for k in range(nf)
                        ])
                        bg_per_frame = np.array([
                            np.trapz(frame_amp[k, bg_li:bg_ri+1], freq[bg_li:bg_ri+1])
                            for k in range(nf)
                        ])
                        net_per_frame = pk_per_frame - bg_per_frame
                        if nf > 1 and np.std(net_per_frame, ddof=1) > 0:
                            peak_areas_err[j] = np.std(net_per_frame, ddof=1) * np.sqrt(nf)

            except Exception:
                pass

        return (peaks, peak_freqs, peak_heights, peak_widths_freq, peak_areas,
                peak_areas_err, peak_means, peak_stds, left_idxs, right_idxs,
                bg_levels, bg_left_idxs, bg_right_idxs)

    def _find_peaks_projections(self):
        """Find peaks in each baseline-removed projection and save to CSV."""
        if self._projections is None or len(self._projections) == 0:
            QMessageBox.warning(self, "Warning", "No projections computed yet — click Project first")
            return
        data_source = self._projections_clean if self._projections_clean is not None else self._projections
        if data_source is None:
            return

        freq = self.freq_data
        all_peaks = []
        self._write_debug(f"DEBUG: finding peaks in {len(data_source)} projections")
        for i, proj in enumerate(data_source):
            n_frames = int(self._proj_nframes[i]) if self._proj_nframes is not None and i < len(self._proj_nframes) else 1
            try:
                frame_amp = (self._projections_frame_data[i]
                             if self._projections_frame_data is not None
                             and i < len(self._projections_frame_data)
                             else None)
                (peaks, pf, ph, pwf, pa, pae, pm, ps,
                 lidx, ridx, bg_l, bg_li, bg_ri) = self._detect_peaks_one(freq, proj, frame_amp)
                all_peaks.append({
                    'indices': peaks, 'freqs': pf, 'heights': ph,
                    'widths_freq': pwf, 'areas': pa, 'areas_err': pae,
                    'means': pm, 'stds': ps,
                    'left_idxs': lidx, 'right_idxs': ridx,
                    'bg_levels': bg_l, 'bg_left_idxs': bg_li, 'bg_right_idxs': bg_ri,
                    'n_frames': n_frames,
                })
            except Exception as e:
                self._write_debug(f"DEBUG: peak finding failed for projection {i}: {e}")
                all_peaks.append({
                    'indices': np.array([], dtype=int),
                    'freqs': np.array([]), 'heights': np.array([]),
                    'widths_freq': np.array([]), 'areas': np.array([]), 'areas_err': np.array([]),
                    'means': np.array([]), 'stds': np.array([]),
                    'left_idxs': np.array([], dtype=int),
                    'right_idxs': np.array([], dtype=int),
                    'bg_levels': np.array([]),
                    'bg_left_idxs': np.array([], dtype=int),
                    'bg_right_idxs': np.array([], dtype=int),
                    'n_frames': n_frames,
                })

        self._projections_peaks = [p['indices'] for p in all_peaks]
        self._projections_peak_masks = [
            np.ones(len(p['indices']), dtype=bool) for p in all_peaks
        ]
        self._projections_peak_data = all_peaks
        n_total = sum(len(p['indices']) for p in all_peaks)
        print(f"✅ Found {n_total} peaks in {len(data_source)} projections")
        self._save_peaks_csv(all_peaks)
        self._redraw_projections()
        self._save_config()

    def _save_peaks_csv(self, all_peaks):
        """Export peaks to CSV: time, voltage, frequency, height, FWHM, area, area_err, mean, std."""
        csv_path = os.path.join(os.getcwd(), "afc_peaks.csv")
        try:
            with open(csv_path, 'w') as f:
                f.write("time_s,voltage_V,frequency_MHz,height,FWHM_MHz,area,area_err,mean,std\n")
                for i, pk in enumerate(all_peaks):
                    tv = self.voltage_time[i] if self.voltage_time is not None and i < len(self.voltage_time) else i
                    vv = (self._proj_voltages[i] if self._proj_voltages is not None
                          and i < len(self._proj_voltages) else 0)
                    for j in range(len(pk['freqs'])):
                        ae = pk.get('areas_err', [0])[j] if j < len(pk.get('areas_err', [])) else 0
                        mn = pk.get('means', [0])[j] if j < len(pk.get('means', [])) else 0
                        sd = pk.get('stds', [0])[j] if j < len(pk.get('stds', [])) else 0
                        f.write(f"{tv:.1f},{vv:.3f},{pk['freqs'][j]:.6f},"
                                f"{pk['heights'][j]:.6e},{pk['widths_freq'][j]:.6f},"
                                f"{pk['areas'][j]:.6e},{ae:.6e},{mn:.6e},{sd:.6e}\n")
            print(f"✅ Peaks saved to {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save peaks CSV: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Right-click peak removal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _connect_projection_right_click(self, ax):
        """Connect right-click handler to a projection axis (called once per draw)."""
        cid = getattr(ax, '_right_click_cid', None)
        if cid is not None:
            try:
                ax.figure.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        ax._right_click_cid = ax.figure.canvas.mpl_connect(
            'button_press_event', self._on_projection_right_click
        )

    def _on_projection_right_click(self, event):
        """Handle right-click on a projection plot to toggle/remove peaks."""
        if event.button != 3:  # right click only
            return
        ax = event.inaxes
        if ax is None:
            return
        proj_idx = getattr(ax, '_peak_idx', None)
        if proj_idx is None:
            return
        if (self._projections_peaks is None
                or proj_idx >= len(self._projections_peaks)):
            return

        peaks = self._projections_peaks[proj_idx]
        mask = (self._projections_peak_masks[proj_idx]
                if self._projections_peak_masks is not None
                and proj_idx < len(self._projections_peak_masks)
                else np.ones(len(peaks), dtype=bool))

        if len(peaks) == 0 or not np.any(mask):
            return

        # Find nearest peak in frequency
        click_freq = event.xdata
        shown = peaks[mask]
        shown_freqs = self.freq_data[shown]
        dists = np.abs(shown_freqs - click_freq)
        nearest_idx_in_shown = np.argmin(dists)

        # Check if click is close enough (within 1 MHz)
        if dists[nearest_idx_in_shown] > 1.0:
            return

        # Find its position in the full peaks array
        actual_peak_idx = np.where(peaks == shown[nearest_idx_in_shown])[0][0]

        # Build context menu
        menu = QMenu(self)
        menu.setStyleSheet("font-size: 10pt;")
        remove_action = menu.addAction(f"Remove peak at {shown_freqs[nearest_idx_in_shown]:.4f} MHz")
        keep_only_action = menu.addAction("Keep this peak only")
        menu.addSeparator()
        restore_action = menu.addAction("Restore all peaks")

        action = menu.exec_(QCursor.pos())

        if action == remove_action:
            self._projections_peak_masks[proj_idx][actual_peak_idx] = False
            print(f"🗑️ Removed peak at {shown_freqs[nearest_idx_in_shown]:.4f} MHz")
        elif action == keep_only_action:
            new_mask = np.zeros(len(peaks), dtype=bool)
            new_mask[actual_peak_idx] = True
            self._projections_peak_masks[proj_idx] = new_mask
            print(f"🔍 Kept only peak at {shown_freqs[nearest_idx_in_shown]:.4f} MHz")
        elif action == restore_action:
            self._projections_peak_masks[proj_idx] = np.ones(len(peaks), dtype=bool)
            print("🔄 Restored all peaks")
        else:
            return

        # Re-save CSV with updated masks
        self._resave_peaks_csv()
        self._redraw_projections()

    def _resave_peaks_csv(self):
        """Re-save afc_peaks.csv respecting current masks."""
        if self._projections_peak_data is None:
            return
        csv_path = os.path.join(os.getcwd(), "afc_peaks.csv")
        try:
            with open(csv_path, 'w') as f:
                f.write("time_s,voltage_V,frequency_MHz,height,FWHM_MHz,area,area_err,mean,std,kept\n")
                for i, pk in enumerate(self._projections_peak_data):
                    mask = (self._projections_peak_masks[i]
                            if self._projections_peak_masks is not None
                            and i < len(self._projections_peak_masks)
                            else np.ones(len(pk['freqs']), dtype=bool))
                    tv = self.voltage_time[i] if self.voltage_time is not None and i < len(self.voltage_time) else i
                    vv = (self._proj_voltages[i] if self._proj_voltages is not None
                          and i < len(self._proj_voltages) else 0)
                    for j in range(len(pk['freqs'])):
                        kept = "1" if mask[j] else "0"
                        ae = pk.get('areas_err', [0])[j] if j < len(pk.get('areas_err', [])) else 0
                        mn = pk.get('means', [0])[j] if j < len(pk.get('means', [])) else 0
                        sd = pk.get('stds', [0])[j] if j < len(pk.get('stds', [])) else 0
                        f.write(f"{tv:.1f},{vv:.3f},"
                                f"{pk['freqs'][j]:.6f},{pk['heights'][j]:.6e},"
                                f"{pk['widths_freq'][j]:.6f},{pk['areas'][j]:.6e},"
                                f"{ae:.6e},{mn:.6e},{sd:.6e},{kept}\n")
            print(f"✅ Peaks re-saved to {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not re-save peaks CSV: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Harmonic fitting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ──────────────────────────────────────────────
    # Resonance fitting: A(f) = A0 / (1 + Q²(f/f_sys - f_sys/f)²)
    # ──────────────────────────────────────────────

    @staticmethod
    def _resonance_func(f, A0, Q_sys, f_sys):
        """Lorentzian-like resonance curve."""
        x = f / f_sys
        return A0 / (1.0 + Q_sys**2 * (x - 1.0 / x)**2)

    def _fit_resonance(self, freqs, areas):
        """Fit area~frequency with the resonance curve.

        Returns (A0, Q_sys, f_sys) or (0, 0, 0) on failure.
        """
        if len(freqs) < 4:
            return 0.0, 0.0, 0.0
        try:
            p0 = [1e4, 1e4, 310.0]
            popt, _ = curve_fit(
                self._resonance_func, freqs, areas,
                p0=p0, maxfev=50000,
                bounds=([0, 0, freqs[0]], [np.inf, 1e6, freqs[-1]])
            )
            return float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            return 0.0, 0.0, 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Global normalized-area plot
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _load_peaks_from_csv(self, csv_path=None):
        """Load peak data from afc_peaks.csv into _projections_peak_data.

        Called as fallback when no in-memory peak data exists.
        Returns True on success.
        """
        if csv_path is None:
            csv_path = os.path.join(os.getcwd(), "afc_peaks.csv")
        if not os.path.exists(csv_path):
            return False
        try:
            # Read CSV manually to handle both old (8-col) and new (9-col) formats
            with open(csv_path, 'r') as cf:
                lines = [l.strip() for l in cf if l.strip()]
            if len(lines) < 2:
                return False
            header = lines[0].split(',')
            has_ae = 'area_err' in header
            data = np.array([line.split(',') for line in lines[1:]], dtype=str)
            if len(data) == 0:
                return False
            times   = data[:, 0].astype(float)
            volts   = data[:, 1].astype(float)
            freqs   = data[:, 2].astype(float)
            heights = data[:, 3].astype(float)
            fwhms   = data[:, 4].astype(float)
            areas   = data[:, 5].astype(float)
            if has_ae and data.shape[1] > 6:
                areas_err = data[:, 6].astype(float)
                mcol = 7; scol = 8
            else:
                areas_err = np.zeros_like(areas)
                mcol = 6; scol = 7
            means   = data[:, mcol].astype(float) if data.shape[1] > mcol else np.zeros_like(areas)
            stds    = data[:, scol].astype(float) if data.shape[1] > scol else np.zeros_like(areas)
            unique_times, inv_idx = np.unique(times, return_inverse=True)
            self._projections_peak_data = []
            self._projections_peak_masks = []
            self._proj_voltages = []
            self._proj_nframes = []
            self.voltage_time = []
            for pi in range(len(unique_times)):
                mask = (inv_idx == pi)
                n_peaks = int(mask.sum())
                if n_peaks == 0:
                    continue
                self._projections_peak_data.append({
                    'indices':      np.arange(n_peaks, dtype=int),
                    'freqs':        freqs[mask].copy(),
                    'heights':      heights[mask].copy(),
                    'widths_freq':  fwhms[mask].copy(),
                    'areas':        areas[mask].copy(),
                    'areas_err':    areas_err[mask].copy(),
                    'means':        means[mask].copy(),
                    'stds':         stds[mask].copy(),
                    'left_idxs':    np.zeros(n_peaks, dtype=int),
                    'right_idxs':   np.zeros(n_peaks, dtype=int),
                    'bg_levels':    np.zeros(n_peaks),
                    'bg_left_idxs': np.zeros(n_peaks, dtype=int),
                    'bg_right_idxs':np.zeros(n_peaks, dtype=int),
                    'n_frames':     1,
                })
                self._projections_peak_masks.append(np.ones(n_peaks, dtype=bool))
                self._proj_voltages.append(float(np.mean(volts[mask])))
                self.voltage_time.append(float(unique_times[pi]))
            n_total = sum(len(p['freqs']) for p in self._projections_peak_data)
            print(f"Loaded {len(self._projections_peak_data)} projections ({n_total} peaks) from {csv_path}")
            return True
        except Exception as e:
            print(f"Could not load peaks from CSV: {e}")
            return False

    def _plot_normalized_areas(self):
        """Area-ratio AFC resonance fit (AFC_npzmonitor.ipynb approach).

        For each projection, uses the selected reference peak (Ref combo).
        Builds frequency pairs (f_i, f_ref) with area ratio R = A_i / A_ref and
        error dR from area_err propagation.  Model:

            R_theory(f1, f2) = |H(f1)|^2 / |H(f2)|^2
            H(f) = 1 / sqrt(1 + Q^2 * (f/f_sys - f_sys/f)^2)

        Weighted least squares (curve_fit) + MC (5000 iter).
        """
        if not self._load_peaks_from_csv():
            QMessageBox.warning(self, "Warning", "No afc_peaks.csv found -- click Find Peaks first")
            return

        # -- 1) Build (f1, f2, R, dR) pairs per projection --
        ref_text = self.norm_ref_combo.currentText()

        def _get_ref_idx(areas, mode):
            if mode == "Ref: 1st peak":
                return 0
            elif mode == "Ref: last peak":
                return len(areas) - 1
            else:  # "Ref: Max Area"
                return int(np.argmax(areas))

        f1_list, f2_list, R_list, dR_list = [], [], [], []
        A1_list, A1_err_list, Aref_list, Aref_err_list = [], [], [], []
        proj_idx_list = []

        for i, pk in enumerate(self._projections_peak_data):
            mask = (self._projections_peak_masks[i]
                    if self._projections_peak_masks is not None
                    and i < len(self._projections_peak_masks)
                    else np.ones(len(pk['freqs']), dtype=bool))
            kept = mask.astype(bool)
            kept_areas = pk['areas'][kept]
            kept_areas_err = pk.get('areas_err', np.zeros_like(kept_areas))[kept]
            kept_freqs = pk.get('means', pk['freqs'])[kept]

            if len(kept_areas) < 2:
                continue

            ridx = _get_ref_idx(kept_areas, ref_text)
            A_ref = kept_areas[ridx]
            A_ref_err = kept_areas_err[ridx]
            f_ref = kept_freqs[ridx]

            for j in range(len(kept_areas)):
                if j == ridx:
                    continue
                R = kept_areas[j] / A_ref
                dA_i = max(kept_areas_err[j], kept_areas[j] * 0.01)
                dA_r = max(A_ref_err, A_ref * 0.01)
                dR = R * np.sqrt((dA_i / kept_areas[j])**2 + (dA_r / A_ref)**2)
                f1_list.append(kept_freqs[j])
                f2_list.append(f_ref)
                R_list.append(R)
                dR_list.append(dR)
                A1_list.append(kept_areas[j]); A1_err_list.append(dA_i)
                Aref_list.append(A_ref); Aref_err_list.append(dA_r)
                proj_idx_list.append(i)

        if len(R_list) < 4:
            QMessageBox.warning(self, "Warning",
                                "Not enough peak pairs for fitting (need >= 4)")
            return

        f1 = np.array(f1_list)
        f2 = np.array(f2_list)
        R_exp = np.array(R_list)
        delta_R = np.array(dR_list)
        A1_exp = np.array(A1_list); A1_err_exp = np.array(A1_err_list)
        Aref_exp = np.array(Aref_list); Aref_err_exp = np.array(Aref_err_list)
        proj_idx_arr = np.array(proj_idx_list)

        # Filter invalid data
        valid = ~np.isnan(R_exp) & ~np.isinf(R_exp) & (delta_R > 0) & (R_exp > 0)
        f1 = f1[valid]; f2 = f2[valid]
        R_exp = R_exp[valid]; delta_R = delta_R[valid]
        A1_exp = A1_exp[valid]; A1_err_exp = A1_err_exp[valid]
        Aref_exp = Aref_exp[valid]; Aref_err_exp = Aref_err_exp[valid]
        proj_idx_arr = proj_idx_arr[valid]

        if len(R_exp) < 4:
            QMessageBox.warning(self, "Warning",
                                "Not enough valid pairs after filtering")
            return

        # -- 2) Model functions --
        def _H_resp(f, Q_sys, f_sys):
            x = f / f_sys
            return 1.0 / np.sqrt(1.0 + Q_sys**2 * (x - 1.0 / x)**2)

        def _ratio_model(f_pair, Q_sys, f_sys):
            f1_, f2_ = f_pair
            return (_H_resp(f1_, Q_sys, f_sys)**2) / (_H_resp(f2_, Q_sys, f_sys)**2)

        # -- 3) Weighted least-squares fit --
        f_pair = np.vstack([f1, f2])
        f_sys_guess = np.median(np.concatenate([f1, f2]))
        p0 = [10000, f_sys_guess]
        bounds = ((10, f_sys_guess * 0.9), (20000, f_sys_guess * 1.1))

        try:
            popt, pcov = curve_fit(_ratio_model, f_pair, R_exp,
                                   p0=p0, sigma=delta_R,
                                   absolute_sigma=True,
                                   bounds=bounds, maxfev=10000)
        except Exception as e:
            QMessageBox.warning(self, "Fit Error",
                                f"curve_fit failed:\n{str(e)}")
            return

        Q_fit, f_sys_fit = popt
        Q_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0.0
        f_sys_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.0
        self._last_res_params = (1.0, Q_fit, f_sys_fit)

        # -- 4) Monte Carlo (5000 iter) --
        N_MC = 5000
        rng = np.random.default_rng()
        Q_samples, f_samples = [], []
        for _ in range(N_MC):
            R_sim = R_exp + rng.normal(0, delta_R)
            R_sim = np.clip(R_sim, 1e-6, None)
            try:
                ps, _ = curve_fit(_ratio_model, f_pair, R_sim,
                                  p0=p0, sigma=delta_R,
                                  absolute_sigma=True,
                                  bounds=bounds, maxfev=2000)
                Q_samples.append(ps[0]); f_samples.append(ps[1])
            except RuntimeError:
                continue

        Q_samples = np.array(Q_samples); f_samples = np.array(f_samples)
        if len(Q_samples) > 0:
            Q_mu, Q_sigma = np.mean(Q_samples), np.std(Q_samples)
            f_mu, f_sigma = np.mean(f_samples), np.std(f_samples)
        else:
            Q_mu, Q_sigma = Q_fit, Q_err
            f_mu, f_sigma = f_sys_fit, f_sys_err

        # -- 5) Build result dialog --
        dialog = QDialog(self)
        dialog.setWindowTitle("AFC Resonance Fit - Area Ratio Method (MC)")
        dialog.resize(850, 700)
        layout = QVBoxLayout(dialog)

        summary = (f"Least Squares:  Q = {Q_fit:.1f} +/- {Q_err:.1f}"
                   f"    f_sys = {f_sys_fit:.6f} +/- {f_sys_err:.2e} MHz\n"
                   f"MC ({len(Q_samples)} iter):  Q = {Q_mu:.1f} +/- {Q_sigma:.1f}"
                   f"    f_sys = {f_mu:.6f} +/- {f_sigma:.2e} MHz")
        lbl = QLabel(summary)
        lbl.setFont(QFont("Consolas", 10))
        lbl.setStyleSheet("padding: 4px; background: #f5f5f5; border: 1px solid #ccc;")
        layout.addWidget(lbl)

        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.35, height_ratios=[2, 1])
        ax_top = fig.add_subplot(gs[0, :])
        ax_res = fig.add_subplot(gs[1, 0])
        ax_mcq = fig.add_subplot(gs[1, 1])
        ax_mcf = fig.add_subplot(gs[1, 2])

        import matplotlib.cm as _cm
        unique_projs = np.unique(proj_idx_arr)
        colours = _cm.tab10(np.linspace(0, 1, len(unique_projs)))

        f2_med = np.median(f2)
        f_grid = np.linspace(f1.min(), f1.max(), 500)
        R_curve = _ratio_model(np.vstack([f_grid, np.full_like(f_grid, f2_med)]),
                                Q_fit, f_sys_fit)

        for ci, pi in enumerate(unique_projs):
            pidx = proj_idx_arr == pi
            vv = (self._proj_voltages[pi] if self._proj_voltages is not None
                  and pi < len(self._proj_voltages) else pi)
            label = (f"{vv:.3f} V" if isinstance(vv, (int, float))
                     else f"Proj {pi}")
            ax_top.errorbar(f1[pidx], R_exp[pidx], yerr=delta_R[pidx],
                            fmt='o', c=colours[ci], ms=4, capsize=2,
                            label=label, alpha=0.7)

        ax_top.plot(f_grid, R_curve, 'r-', linewidth=1.5)
        txt = (f"$Q$ = {Q_fit:.0f} $\\pm$ {Q_err:.0f}"
               f"  [MC: {Q_sigma:.0f}]\n"
               r"$f_{\rm sys}$" + f" = {f_sys_fit:.6f} $\\pm$ {f_sys_err:.2e}"
               f"  [MC: {f_sigma:.2e}] MHz\n"
               f"MC samples: {len(Q_samples)}")
        ax_top.text(0.97, 0.97, txt, transform=ax_top.transAxes,
                    fontsize=9, fontfamily='monospace',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax_top.set_ylabel("Area Ratio $R = A_i / A_{\\rm ref}$")
        ax_top.set_xlabel("Frequency (MHz)")
        ax_top.legend(fontsize=7, loc='upper left')
        ax_top.grid(alpha=0.3)

        # Residuals
        R_pred = _ratio_model(f_pair, Q_fit, f_sys_fit)
        resid = R_exp - R_pred
        for ci, pi in enumerate(unique_projs):
            pidx = proj_idx_arr == pi
            ax_res.errorbar(f1[pidx], resid[pidx], yerr=delta_R[pidx],
                           fmt='o', c=colours[ci], ms=4, capsize=2, alpha=0.7)
        ax_res.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax_res.set_xlabel("Frequency (MHz)")
        ax_res.set_ylabel("Residual")
        ax_res.grid(alpha=0.3)

        # Q MC histogram
        if len(Q_samples) > 1:
            ax_mcq.hist(Q_samples, bins=30, density=True,
                        color='skyblue', edgecolor='k')
            ax_mcq.axvline(Q_mu, color='red', linestyle='dashed', linewidth=1)
            ax_mcq.set_xlabel("$Q$")
            ax_mcq.set_ylabel("Density")
            ax_mcq.set_title(f"MC $Q$: {Q_mu:.0f} $\\pm$ {Q_sigma:.0f}", fontsize=9)
            ax_mcq.grid(alpha=0.3)

        # f_sys MC histogram
        if len(f_samples) > 1:
            ax_mcf.hist(f_samples, bins=30, density=True,
                        color='salmon', edgecolor='k')
            ax_mcf.axvline(f_mu, color='red', linestyle='dashed', linewidth=1)
            ax_mcf.set_xlabel("$f_{\\rm sys}$ (MHz)")
            ax_mcf.set_ylabel("Density")
            ax_mcf.set_title(f"MC $f_{{\\rm sys}}$: {f_mu:.6f} $\\pm$ {f_sigma:.2e}", fontsize=9)
            ax_mcf.grid(alpha=0.3)

        canvas.draw()

        # -- 6) Save CSV --
        csv_path = os.path.join(os.getcwd(), "afc_norm_area.csv")
        try:
            with open(csv_path, 'w') as f:
                f.write("proj_idx,time_s,voltage_V,f1_MHz,f_ref_MHz,"
                        "area1,area1_err,area_ref,area_ref_err,"
                        "area_ratio,delta_ratio,fitted_ratio,residual\n")
                for ci, pi in enumerate(unique_projs):
                    pidx = proj_idx_arr == pi
                    tv = (self.voltage_time[pi] if self.voltage_time is not None
                          and pi < len(self.voltage_time) else pi)
                    vv = (self._proj_voltages[pi] if self._proj_voltages is not None
                          and pi < len(self._proj_voltages) else 0)
                    for j in np.where(pidx)[0]:
                        val_fit = _ratio_model(f_pair[:, j:j+1], Q_fit, f_sys_fit)[0]
                        val_res = R_exp[j] - val_fit
                        f.write(f"{pi},{tv:.1f},{vv:.3f},"
                                f"{f1[j]:.8f},{f2[j]:.8f},"
                                f"{A1_exp[j]:.6e},{A1_err_exp[j]:.6e},"
                                f"{Aref_exp[j]:.6e},{Aref_err_exp[j]:.6e},"
                                f"{R_exp[j]:.6e},{delta_R[j]:.6e},"
                                f"{val_fit:.6e},{val_res:.6e}\n")
            print(f"Normalized area saved to {csv_path}")
        except Exception as exc:
            print(f"CSV save error: {exc}")

        dialog.exec_()

    def _load_harmonics_from_csv(self, csv_path=None):
        """Load peak data with harmonic numbers from afc_harmonics.csv.

        Reconstructs _projections_peak_data with har_n populated.
        Returns True on success.
        """
        if csv_path is None:
            csv_path = os.path.join(os.getcwd(), "afc_harmonics.csv")
        if not os.path.exists(csv_path):
            return False
        try:
            raw = np.genfromtxt(csv_path, delimiter=',',
                                dtype=None, names=True, encoding='utf-8')
            if raw is None or len(raw) == 0:
                return False
            if raw.ndim == 0:
                raw = np.array([raw])

            proj_idx  = np.array([r['proj_idx']       for r in raw], dtype=int)
            times     = np.array([r['time_s']          for r in raw], dtype=float)
            volts     = np.array([r['voltage_V']        for r in raw], dtype=float)
            har_n     = np.array([r['harmonic_n']       for r in raw], dtype=int)
            freqs     = np.array([r['freq_MHz']         for r in raw], dtype=float)
            heights   = np.array([r['height']           for r in raw], dtype=float)
            fwhms     = np.array([r['FWHM_MHz']         for r in raw], dtype=float)
            areas     = np.array([r['area']             for r in raw], dtype=float)
            col_names = raw.dtype.names
            if 'area_err' in col_names:
                areas_err = np.array([r['area_err']     for r in raw], dtype=float)
            else:
                areas_err = np.zeros_like(areas)
            means     = np.array([r['mean_freq_MHz']    for r in raw], dtype=float)
            stds      = np.array([r['std_freq_MHz']     for r in raw], dtype=float)
            kept_col  = np.array([r['kept']             for r in raw], dtype=int)

            unique_idx = np.unique(proj_idx)
            self._projections_peak_data = []
            self._projections_peak_masks = []
            self._proj_voltages = []
            self._proj_nframes = []
            self.voltage_time = []

            for pi in unique_idx:
                mask = proj_idx == pi
                n_peaks = int(mask.sum())
                if n_peaks == 0:
                    continue
                self._projections_peak_data.append({
                    'indices':      np.arange(n_peaks, dtype=int),
                    'har_n':        har_n[mask].copy(),
                    'freqs':        freqs[mask].copy(),
                    'heights':      heights[mask].copy(),
                    'widths_freq':  fwhms[mask].copy(),
                    'areas':        areas[mask].copy(),
                    'areas_err':    areas_err[mask].copy(),
                    'means':        means[mask].copy(),
                    'stds':         stds[mask].copy(),
                    'left_idxs':    np.zeros(n_peaks, dtype=int),
                    'right_idxs':   np.zeros(n_peaks, dtype=int),
                    'bg_levels':    np.zeros(n_peaks),
                    'bg_left_idxs': np.zeros(n_peaks, dtype=int),
                    'bg_right_idxs':np.zeros(n_peaks, dtype=int),
                    'n_frames':     1,
                })
                self._projections_peak_masks.append(kept_col[mask].astype(bool))
                self._proj_voltages.append(float(np.mean(volts[mask])))
                self.voltage_time.append(float(np.mean(times[mask])))

            n_total = sum(len(p['freqs']) for p in self._projections_peak_data)
            print(f"Loaded {len(self._projections_peak_data)} projections ({n_total} peaks) from {csv_path}")
            return True
        except Exception as e:
            print(f"Could not load harmonics from CSV: {e}")
            return False

    def _plot_harmonic_vs_time(self):
        """Plot raw and resonance-corrected area of a specific harmonic vs time."""
        if not self._load_harmonics_from_csv():
            QMessageBox.warning(self, "Warning", "No afc_harmonics.csv found -- click Find Peaks + Fit Har first")
            return

        try:
            target_har = int(self.har_vs_time_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Enter a valid harmonic number")
            return

        # Collect area & time for the target harmonic from each projection
        times = []
        volts = []
        freqs = []
        raw_areas = []  # net area = area - bg_level
        raw_areas_err = []  # error on net area

        for i, pk in enumerate(self._projections_peak_data):
            har_arr = pk.get('har_n', None)
            if har_arr is None or len(har_arr) == 0:
                continue
            # Find the peak with matching harmonic number
            match_idx = np.where(har_arr == target_har)[0]
            if len(match_idx) == 0:
                continue
            match_idx = match_idx[0]  # take first match

            mask = (self._projections_peak_masks[i]
                    if self._projections_peak_masks is not None and i < len(self._projections_peak_masks)
                    else np.ones(len(pk['freqs']), dtype=bool))
            if not mask[match_idx]:
                continue

            tv = (self.voltage_time[i] if self.voltage_time is not None and i < len(self.voltage_time)
                  else float(i))
            vv = (self._proj_voltages[i] if self._proj_voltages is not None and i < len(self._proj_voltages)
                  else 0)

            freq_val = pk.get('means', pk['freqs'])[match_idx]
            area_val = pk['areas'][match_idx] if match_idx < len(pk['areas']) else 0
            area_err_val = (pk.get('areas_err', np.zeros(len(pk['areas'])))[match_idx]
                           if match_idx < len(pk.get('areas_err', [])) else 0)
            bg_val = (pk['bg_levels'][match_idx] if 'bg_levels' in pk and match_idx < len(pk['bg_levels'])
                      else 0)
            net_area = area_val - bg_val
            # Divide by frames to match the A/f shown on projection plots
            nfr = int(self._proj_nframes[i]) if self._proj_nframes is not None and i < len(self._proj_nframes) else 1
            net_area /= max(1, nfr)
            net_area_err = area_err_val / max(1, nfr)

            times.append(float(tv))
            volts.append(float(vv))
            freqs.append(float(freq_val))
            raw_areas.append(float(net_area))
            raw_areas_err.append(float(net_area_err))

        if len(raw_areas) < 3:
            QMessageBox.warning(self, "Warning", f"Not enough projections with harmonic #{target_har} (need ≥3, got {len(raw_areas)})")
            return

        arr_t = np.array(times)
        arr_a = np.array(raw_areas)
        arr_a_err = np.array(raw_areas_err)
        arr_f = np.array(freqs)

        # Build dialog with resonance params + plot
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Harmonic #{target_har} Area vs Time")
        dialog.resize(700, 500)
        layout = QVBoxLayout(dialog)

        # Parameter row: Q, f_sys inputs (H(f) model, no A0)
        param_layout = QHBoxLayout()
        q_edit = QLineEdit()
        fs_edit = QLineEdit()
        q_edit.setMaximumWidth(70)
        fs_edit.setMaximumWidth(70)
        if self._last_res_params is not None:
            q_edit.setText(f"{self._last_res_params[1]:.0f}")
            fs_edit.setText(f"{self._last_res_params[2]:.6f}")
        else:
            q_edit.setText("10000")
            fs_edit.setText("310")
        param_layout.addWidget(QLabel("Q:"))
        param_layout.addWidget(q_edit)
        param_layout.addWidget(QLabel("f_sys:"))
        param_layout.addWidget(fs_edit)
        update_btn = QPushButton("Update")
        update_btn.setStyleSheet("background-color: #2196F3; color: white; border-radius: 3px; padding: 2px 12px;")
        param_layout.addWidget(update_btn)
        param_layout.addStretch()
        layout.addLayout(param_layout)

        # Matplotlib figure
        fig = Figure(figsize=(7, 4.5), dpi=100, constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Net Area")
        ax.set_title(f"Harmonic #{target_har}: Raw & Corrected Area vs Time")
        ax.grid(alpha=0.3)

        def _redraw():
            try:
                p_Q = float(q_edit.text())
                p_fs = float(fs_edit.text())
            except ValueError:
                return

            ax.clear()
            # Raw area (blue) with error bars
            ax.errorbar(arr_t, arr_a, yerr=arr_a_err, fmt='o',
                       c='#2196F3', ms=4, capsize=2, label='Raw (net)', zorder=3)
            ax.plot(arr_t, arr_a, color='#2196F3', linewidth=0.8, alpha=0.5)

            # Corrected area = net area / H(freq)  (AFC correction, no normalization)
            def _H_avt(f): return 1.0 / np.sqrt(1.0 + p_Q**2 * (f/p_fs - p_fs/f)**2)
            r_factor = np.maximum(_H_avt(arr_f), 1e-30)
            corr_a = arr_a / r_factor
            corr_a_err = arr_a_err / r_factor
            ax.errorbar(arr_t, corr_a, yerr=corr_a_err, fmt='s',
                       c='#E91E63', ms=4, capsize=2, label='Corrected (AFC)', zorder=3)
            ax.plot(arr_t, corr_a, color='#E91E63', linewidth=0.8, alpha=0.5)

            # Weighted exponential decay fit with scipy
            try:
                from scipy.optimize import curve_fit as _cf
                t0 = float(arr_t[0])
                t_shifted = arr_t - t0
                w_err = np.maximum(corr_a_err, corr_a * 0.001)
                popt_e, pcov_e = _cf(
                    lambda t, A0, lam: A0 * np.exp(-lam * t),
                    t_shifted, corr_a,
                    sigma=w_err, absolute_sigma=True,
                    p0=[corr_a[0], 0.01],
                    maxfev=10000
                )
                A0_fit = float(popt_e[0])
                lam = float(popt_e[1])
                lam_err = float(np.sqrt(pcov_e[1, 1])) if pcov_e[1, 1] > 0 else 0
                half_life = np.log(2) / lam if lam > 0 else float('inf')
                half_life_err = half_life * (lam_err / lam) if lam > 0 else 0

                t_fine = np.linspace(t_shifted.min(), t_shifted.max(), 300)
                a_fit_exp = A0_fit * np.exp(-lam * t_fine)
                ax.plot(t_fine + t0, a_fit_exp, 'g-', linewidth=1.8, label='Exp fit (weighted)', zorder=4)

                txt = (f"$A_0$ = {A0_fit:.4e}   "
                       f"$\lambda$ = {lam:.4e} $\pm$ {lam_err:.4e} s$^{{-1}}$   "
                       f"$T_{{1/2}}$ = {half_life:.2f} $\pm$ {half_life_err:.2f} s")
                ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                        fontsize=9, fontfamily='monospace',
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.85))
            except Exception as e:
                print(f"Warning: Weighted exponential fit skipped: {e}")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Net Area")
            ax.set_title(f"Harmonic #{target_har}: Raw \& Corrected Area vs Time")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            canvas.draw()
        update_btn.clicked.connect(_redraw)
        _redraw()
        dialog.exec_()

    def _self_consistent_fit(self):
        """Iteratively fit resonance curve, correcting via AF_i/AF_ref each iteration.

        Like repeatedly clicking "Plot Norm Area" — each iteration:
          a) Fit resonance R(f) to current corrected areas.
          b) AF-correct each projection's areas by R(f_ref)/R(f_first_of_proj).
        Converges when Q / f_sys stabilise.
        After convergence, fit decay curve for the target harmonic.
        """
        if not self._load_harmonics_from_csv():
            if not self._load_peaks_from_csv():
                QMessageBox.warning(self, "Warning", "No afc_peaks.csv found — click Find Peaks first")
                return

        try:
            target_har = int(self.har_vs_time_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Enter a valid harmonic number in Har#:")
            return

        # ── 1) Collect kept (freq, area) pairs per projection (same as _plot_normalized_areas) ──
        proj_freqs = []
        proj_areas = []
        proj_voltages = []

        for i, pk in enumerate(self._projections_peak_data):
            mask = (self._projections_peak_masks[i]
                    if self._projections_peak_masks is not None
                    and i < len(self._projections_peak_masks)
                    else np.ones(len(pk['freqs']), dtype=bool))
            kept = mask.astype(bool)
            kept_areas = pk['areas'][kept]
            kept_means = pk.get('means', pk['freqs'])[kept]
            if len(kept_areas) < 2:
                continue
            proj_freqs.append(kept_means)
            proj_areas.append(kept_areas)
            vv = (self._proj_voltages[i] if self._proj_voltages is not None
                  and i < len(self._proj_voltages) else None)
            proj_voltages.append(vv)

        if not proj_areas:
            QMessageBox.warning(self, "Warning", "No valid projections with ≥2 peaks")
            return

        # Normalize within each projection
        def _ref_idx_ri(areas):
            mode = self.norm_ref_combo.currentText()
            if mode == "Ref: 1st peak": return 0
            elif mode == "Ref: last peak": return len(areas) - 1
            else: return int(np.argmax(areas))

        all_freqs_raw = []
        all_areas_norm = []
        kept_counts = []
        for pi in range(len(proj_freqs)):
            norm = proj_areas[pi] / proj_areas[pi][_ref_idx_ri(proj_areas[pi])]
            all_freqs_raw.extend(proj_freqs[pi].tolist())
            all_areas_norm.extend(norm.tolist())
            kept_counts.append(len(proj_areas[pi]))

        arr_f = np.array(all_freqs_raw)
        arr_a0 = np.array(all_areas_norm)
        if len(arr_f) < 4:
            QMessageBox.warning(self, "Warning", "Not enough peaks for fitting (need ≥4)")
            return

        f_ref = float(proj_freqs[0][_ref_idx_ri(proj_areas[0])])  # first projection, reference peak

        # ── Collect target-harmonic data for decay fitting ──
        har_times, har_freqs, har_raw = [], [], []
        for i, pk in enumerate(self._projections_peak_data):
            har_arr = pk.get('har_n', None)
            if har_arr is None or len(har_arr) == 0:
                continue
            match_idx = np.where(har_arr == target_har)[0]
            if len(match_idx) == 0:
                continue
            midx = match_idx[0]
            mask = (self._projections_peak_masks[i] if self._projections_peak_masks is not None
                    and i < len(self._projections_peak_masks)
                    else np.ones(len(pk['freqs']), dtype=bool))
            if not mask[midx]:
                continue
            tv = (self.voltage_time[i] if self.voltage_time is not None and i < len(self.voltage_time)
                  else float(i))
            har_times.append(float(tv))
            har_freqs.append(float(pk.get('means', pk['freqs'])[midx]))
            bg_v = (pk['bg_levels'][midx] if 'bg_levels' in pk and midx < len(pk['bg_levels']) else 0)
            net_v = pk['areas'][midx] - bg_v
            nfr = int(self._proj_nframes[i]) if self._proj_nframes is not None and i < len(self._proj_nframes) else 1
            har_raw.append(float(net_v / max(1, nfr)))

        arr_har_t = np.array(har_times)
        arr_har_f = np.array(har_freqs)
        arr_har_a = np.array(har_raw)
        t0_har = arr_har_t[0] if len(arr_har_t) > 0 else 0

        # ── Initial resonance params ──
        if self._last_res_params is not None:
            A0, Q_sys, f_sys = self._last_res_params
        else:
            f_mid = (arr_f.min() + arr_f.max()) / 2.0
            A0, Q_sys, f_sys = 1.0, 5000, f_mid

        # ── Build dialog ──
        dialog = QDialog(self)
        dialog.setWindowTitle("Self-consistent Resonance & Decay Fit")
        dialog.resize(750, 750)
        layout = QVBoxLayout(dialog)

        param_layout = QHBoxLayout()
        for lbl, key, val in [("A0:", "A0", f"{A0:.6e}"),
                              ("Q:", "Q", f"{Q_sys:.0f}"),
                              ("f_sys:", "fsys", f"{f_sys:.6f}")]:
            param_layout.addWidget(QLabel(lbl))
            le = QLineEdit(val)
            le.setMaximumWidth(80)
            setattr(self, f'_sc_{key}_edit', le)
            param_layout.addWidget(le)
        param_layout.addStretch()
        layout.addLayout(param_layout)

        iter_info = QLabel("")
        layout.addWidget(iter_info)

        fig = Figure(figsize=(7, 8), dpi=100, constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        gs = fig.add_gridspec(2, 1, hspace=0.15)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])

        # ── Helper: apply AF correction to normalized areas ──
        def _apply_af(fit_A0, fit_Q, fit_fs):
            af_ref = max(self._resonance_func(f_ref, fit_A0, fit_Q, fit_fs), 1e-30)
            def __ref_idx(areas, mode):
                if mode == "Ref: 1st peak": return 0
                elif mode == "Ref: last peak": return len(areas) - 1
                else: return int(np.argmax(areas))
            ref_mode = self.norm_ref_combo.currentText()
            out = []
            for pi in range(len(proj_freqs)):
                ridx = __ref_idx(proj_areas[pi], ref_mode)
                norm_area = proj_areas[pi] / proj_areas[pi][ridx]
                f_first = float(proj_freqs[pi][ridx])
                af_i = max(self._resonance_func(f_first, fit_A0, fit_Q, fit_fs), 1e-30)
                out.extend((norm_area * (af_i / af_ref)).tolist())
            return np.array(out)

        # ── Helper: redraw both panels ──
        def _draw_panels(corr_a, fit_A0, fit_Q, fit_fs, title_extra="",
                         A0_err=None, Q_err=None, fs_err=None):
            ax_top.clear()
            ax_bot.clear()
            colors = __import__('matplotlib.cm', fromlist=['tab10']).tab10(
                np.linspace(0, 1, len(proj_freqs)))

            off = 0
            for pi in range(len(proj_freqs)):
                end = off + kept_counts[pi]
                v_lbl = (f"{proj_voltages[pi]:.3f} V" if proj_voltages[pi] is not None
                         else f"Proj {pi}")
                ax_top.scatter(proj_freqs[pi], corr_a[off:end],
                               c=[colors[pi]], s=15, label=v_lbl, alpha=0.7)
                off = end

            f_grid = np.linspace(arr_f.min(), arr_f.max(), 500)
            r_grid = self._resonance_func(f_grid, fit_A0, fit_Q, fit_fs)
            ax_top.plot(f_grid, r_grid, 'r-', linewidth=1.5)

            A0_str = f"{fit_A0:.4e}" if A0_err is None else f"{fit_A0:.4e} +/- {A0_err:.4e}"
            Q_str = f"{fit_Q:.0f}" if Q_err is None else f"{fit_Q:.0f} +/- {Q_err:.0f}"
            fs_str = f"{fit_fs:.6f}" if fs_err is None else f"{fit_fs:.6f} +/- {fs_err:.6f}"
            r_txt = (f"$A_0$ = {A0_str}\n"
                     f"$Q$ = {Q_str}\n"
                     r"$f_{{\rm sys}}$" + f" = {fs_str} MHz\n"
                     f"$f_{{\\rm ref}}$ = {f_ref:.6f} MHz")
            ax_top.text(0.97, 0.97, r_txt, transform=ax_top.transAxes,
                        fontsize=8, fontfamily='monospace',
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax_top.set_yscale('log')
            ax_top.set_xticklabels([])
            ax_top.set_ylabel("Corrected Normalized Area")
            ax_top.legend(fontsize=5, ncol=2)
            ax_top.grid(alpha=0.3)

            # Residuals
            r_pts = self._resonance_func(arr_f, fit_A0, fit_Q, fit_fs)
            resid = corr_a - r_pts
            off = 0
            for pi in range(len(proj_freqs)):
                end = off + kept_counts[pi]
                v_lbl = (f"{proj_voltages[pi]:.3f} V" if proj_voltages[pi] is not None
                         else f"Proj {pi}")
                ax_bot.scatter(proj_freqs[pi], resid[off:end],
                               c=[colors[pi]], s=15, label=v_lbl, alpha=0.7)
                off = end
            ax_bot.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax_bot.set_xlabel("Frequency (MHz)")
            ax_bot.set_ylabel("Residual")
            ax_bot.set_title(title_extra, fontsize=8)
            ax_bot.grid(alpha=0.3)
            canvas.draw()

        # ── Button row ──
        btn_layout = QHBoxLayout()
        iterate_btn = QPushButton("Iterate (1 step)")
        iterate_btn.setStyleSheet("background-color: #673AB7; color: white; border-radius: 3px; padding: 4px 16px;")
        btn_layout.addWidget(iterate_btn)
        fit_decay_btn = QPushButton("Fit Decay")
        fit_decay_btn.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 3px; padding: 4px 16px;")
        btn_layout.addWidget(fit_decay_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        dialog.show()
        QApplication.processEvents()

        import ROOT as _R
        _R.gErrorIgnoreLevel = _R.kWarning + 1

        cur_iter = [0]
        cur_corr = [None]

        def _do_one_iteration():
            nonlocal A0, Q_sys, f_sys
            if cur_iter[0] >= 20:
                iter_info.setText("Reached max 20 iterations")
                return
            try:
                xmn, xmx = float(arr_f.min()), float(arr_f.max())
                rf = _R.TF1("sc_r", "[0] / (1.0 + [1]*[1] * (x/[2] - [2]/x)*(x/[2] - [2]/x))", xmn, xmx)
                if cur_iter[0] == 0:
                    A0 = float(self._sc_A0_edit.text())
                    Q_sys = float(self._sc_Q_edit.text())
                    f_sys = float(self._sc_fsys_edit.text())
                rf.SetParameter(0, A0)
                rf.SetParameter(1, Q_sys)
                rf.SetParameter(2, f_sys)
                rf.SetParLimits(0, 0, 1e10)
                rf.SetParLimits(1, 0, 1e7)
                rf.SetParLimits(2, xmn, xmx)
                data_fit = cur_corr[0] if cur_corr[0] is not None else arr_a0
                _R.TGraph(len(arr_f), arr_f, data_fit).Fit("sc_r", "QN")
                A0_n = rf.GetParameter(0); Q_n = rf.GetParameter(1); fs_n = rf.GetParameter(2)
                if not (Q_n > 0 and fs_n > 0):
                    raise RuntimeError("unphysical")
            except Exception as e:
                iter_info.setText(f"Iter {cur_iter[0]+1}: failed — {e}")
                return
            A0, Q_sys, f_sys = A0_n, Q_n, fs_n
            self._last_res_params = (A0, Q_sys, f_sys)
            self._sc_A0_edit.setText(f"{A0:.6e}")
            self._sc_Q_edit.setText(f"{Q_sys:.0f}")
            self._sc_fsys_edit.setText(f"{f_sys:.6f}")
            A0_e = rf.GetParError(0); Q_e = rf.GetParError(1); fs_e = rf.GetParError(2)
            cur_corr[0] = _apply_af(A0, Q_sys, f_sys)
            cur_iter[0] += 1
            _draw_panels(cur_corr[0], A0, Q_sys, f_sys, f"Iteration {cur_iter[0]}",
                         A0_err=A0_e, Q_err=Q_e, fs_err=fs_e)
            iter_info.setText(f"Iter {cur_iter[0]}: A0={A0:.4e} +/- {A0_e:.4e}  Q={Q_sys:.0f} +/- {Q_e:.0f}  f_sys={f_sys:.6f} +/- {fs_e:.6f}")

        def _do_fit_decay():
            if len(arr_har_a) < 3:
                iter_info.setText("Need >=3 harmonics for decay fit")
                return
            try:
                r_fac = np.maximum(self._resonance_func(arr_har_f, A0, Q_sys, f_sys), 1e-30)
                har_c = arr_har_a / r_fac
                har_c = har_c / har_c[0]
                t_s = arr_har_t - t0_har
                ed = _R.TF1("fd", "exp(-[0]*x)", float(t_s.min()), float(t_s.max()))
                ed.SetParameter(0, 0.01); ed.SetParLimits(0, 1e-10, 1e3)
                _R.TGraph(len(t_s), t_s, har_c).Fit("fd", "QN")
                lam_f = ed.GetParameter(0); lam_e = ed.GetParError(0)
                hl = np.log(2)/lam_f; hl_e = hl*lam_e/lam_f
                ax_bot.clear()
                ax_bot.scatter(arr_har_t, har_c, c='#E91E63', s=20, marker='s', label=f'Har #{target_har}', zorder=3)
                tf = np.linspace(arr_har_t.min(), arr_har_t.max(), 300)
                ax_bot.plot(tf, np.exp(-lam_f*(tf-t0_har)), 'g-', linewidth=1.8, label=f'Fit')
                ax_bot.set_xlabel("Time (s)"); ax_bot.set_ylabel("Norm. Area (R-corrected)")
                d_txt = (f"$\\lambda$={lam_f:.4e}$\\pm${lam_e:.4e} s$^{{-1}}$\n"
                         f"$T_{{1/2}}$={hl:.2f}$\\pm${hl_e:.2f} s")
                ax_bot.text(0.97, 0.97, d_txt, transform=ax_bot.transAxes,
                            fontsize=9, fontfamily='monospace', verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.85))
                ax_bot.legend(fontsize=8); ax_bot.grid(alpha=0.3)
                canvas.draw()
                iter_info.setText(f"lambda={lam_f:.4e}+/-{lam_e:.4e} s-1, T1/2={hl:.2f}+/-{hl_e:.2f} s")
            except Exception as e:
                iter_info.setText(f"Decay fit failed: {e}")

        iterate_btn.clicked.connect(_do_one_iteration)
        fit_decay_btn.clicked.connect(_do_fit_decay)
        dialog.exec_()

    def _fit_harmonics(self):
        """Fit harmonic numbers to detected peaks for each projection.

        Sorted peaks are assigned harmonic numbers starting from h_ref.
        A linear fit f = n * f0 + offset yields f0 and offset.
        Results are saved to afc_harmonics.csv.
        """
        if self._projections_peak_data is None:
            QMessageBox.warning(self, "Warning", "No peaks found yet — click Find Peaks first")
            return

        from scipy import stats as _stats
        csv_path = os.path.join(os.getcwd(), "afc_harmonics.csv")
        self._projections_har_fit = []

        try:
            with open(csv_path, 'w') as fh:
                fh.write("proj_idx,time_s,voltage_V,harmonic_n,"
                         "freq_MHz,height,FWHM_MHz,area,area_err,mean_freq_MHz,std_freq_MHz,"
                         "f0_fit_MHz,offset_fit_MHz,residual_MHz,kept\n")

                for i, pk in enumerate(self._projections_peak_data):
                    mask = (self._projections_peak_masks[i]
                            if self._projections_peak_masks is not None
                            and i < len(self._projections_peak_masks)
                            else np.ones(len(pk['freqs']), dtype=bool))
                    kept_mask = mask.astype(bool)
                    kept_freqs = pk.get('means', pk['freqs'])[kept_mask]
                    tv = (self.voltage_time[i] if self.voltage_time is not None
                          and i < len(self.voltage_time) else i)
                    vv = (self._proj_voltages[i] if self._proj_voltages is not None
                          and i < len(self._proj_voltages) else 0)

                    n_kept = len(kept_freqs)
                    if n_kept < 2:
                        self._projections_har_fit.append(None)
                        for j in range(len(pk['freqs'])):
                            hn = j if kept_mask[j] else 0
                            kw = "1" if kept_mask[j] else "0"
                            area_j = pk['areas'][j] if j < len(pk['areas']) else 0
                            mean_j = pk['means'][j] if j < len(pk['means']) else 0
                            std_j = pk['stds'][j] if j < len(pk['stds']) else 0
                            ae_j = pk.get('areas_err', [0])[j] if j < len(pk.get('areas_err', [])) else 0
                            fh.write(f"{i},{tv:.1f},{vv:.3f},{hn},{pk['freqs'][j]:.6f},"
                                     f"{pk['heights'][j]:.6e},{pk['widths_freq'][j]:.6f},"
                                     f"{area_j:.6e},{ae_j:.6e},{mean_j:.6e},{std_j:.6e},,,,,{kw}\n")
                        continue

                    # Sort kept peaks by frequency
                    sort_idx = np.argsort(kept_freqs)
                    kf_sorted = kept_freqs[sort_idx]

                    # Linear fit: f = n * f0 + offset,  n = 0, 1, 2, ...
                    n_vals = np.arange(n_kept, dtype=float)
                    slope, intercept, r_val, p_val, std_err = _stats.linregress(n_vals, kf_sorted)
                    f0, offset = slope, intercept

                    # Read per-projection offsets (space-separated "idx:off" pairs)
                    proj_off_map = {}
                    per_text = self.har_per_offset_edit.text().strip()
                    if per_text:
                        for token in per_text.split():
                            if ':' in token:
                                try:
                                    p_idx, p_off = token.split(':')
                                    proj_off_map[int(p_idx)] = int(p_off)
                                except ValueError:
                                    pass

                    # Global harmonic offset
                    try:
                        h_off = int(self.har_offset_edit.text())
                    except ValueError:
                        h_off = 0
                    # Per-projection offset (overrides global)
                    h_off = proj_off_map.get(i, h_off)

                    # Re-assign harmonic numbers using n = round(f / f0)
                    har_numbers = np.round(kf_sorted / f0).astype(int) + h_off
                    # Re-fit with corrected harmonic numbers
                    n_vals_corrected = har_numbers.astype(float)
                    slope2, intercept2, r_val2, _, _ = _stats.linregress(n_vals_corrected, kf_sorted)
                    f0, offset = slope2, intercept2
                    residuals = kf_sorted - (n_vals_corrected * f0 + offset)

                    self._projections_har_fit.append({
                        'f0': f0, 'offset': offset,
                        'r_squared': r_val2**2, 'std_err': std_err,
                        'n_peaks': n_kept,
                    })
                    print(f"📊 Projection {i}: f0={f0:.6f} MHz, offset={offset:.6e} MHz, "
                          f"R²={r_val2**2:.4f}, N={n_kept} peaks, h_off={h_off}")

                    # Map sorted results back to original peak order
                    kept_indices = np.where(kept_mask)[0]
                    orig_to_har = {}
                    orig_to_resid = {}
                    for s, oi in enumerate(kept_indices[sort_idx]):
                        orig_to_har[oi] = int(har_numbers[s])
                        orig_to_resid[oi] = residuals[s]

                    # Store harmonic numbers in peak data
                    har_arr = np.zeros(len(pk['freqs']), dtype=int)
                    for j in range(len(pk['freqs'])):
                        if kept_mask[j]:
                            har_arr[j] = orig_to_har.get(j, 0)
                    pk['har_n'] = har_arr

                    for j in range(len(pk['freqs'])):
                        ae_j = pk.get('areas_err', [0])[j] if j < len(pk.get('areas_err', [])) else 0
                        if kept_mask[j]:
                            hn = orig_to_har.get(j, 0)
                            resid = orig_to_resid.get(j, 0.0)
                            area_j = pk['areas'][j] if j < len(pk['areas']) else 0
                            mean_j = pk['means'][j] if j < len(pk['means']) else 0
                            std_j = pk['stds'][j] if j < len(pk['stds']) else 0
                            fh.write(f"{i},{tv:.1f},{vv:.3f},{hn},{pk['freqs'][j]:.6f},"
                                     f"{pk['heights'][j]:.6e},{pk['widths_freq'][j]:.6f},"
                                     f"{area_j:.6e},{ae_j:.6e},{mean_j:.6e},{std_j:.6e},"
                                     f"{f0:.6f},{offset:.6e},{resid:.6e},1\n")
                        else:
                            area_j = pk['areas'][j] if j < len(pk['areas']) else 0
                            mean_j = pk['means'][j] if j < len(pk['means']) else 0
                            std_j = pk['stds'][j] if j < len(pk['stds']) else 0
                            fh.write(f"{i},{tv:.1f},{vv:.3f},0,{pk['freqs'][j]:.6f},"
                                     f"{pk['heights'][j]:.6e},{pk['widths_freq'][j]:.6f},"
                                     f"{area_j:.6e},{ae_j:.6e},{mean_j:.6e},{std_j:.6e},,,,,0\n")

                    # Summary comment line
                    fh.write(f"# Proj {i}: f0={f0:.6f} MHz, offset={offset:.6e} MHz, "
                             f"R²={r_val2**2:.4f}, N={n_kept}, t={tv:.1f}s, V={vv:.3f}V\n")

            print(f"✅ Harmonic fit saved to {csv_path}")
            self._redraw_projections()
            self._save_config()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Harmonic fitting failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _lock_current_positions(self):
        """Save current axis positions so they can be restored on redraw."""
        self._locked_ax_positions = []
        for ax in self._proj_axes:
            bbox = ax.get_position().bounds
            self._locked_ax_positions.append(list(bbox))

    def _restore_locked_positions(self):
        """Restore locked axis positions to prevent layout shifts."""
        if not self._locked_ax_positions:
            return
        for ax, pos in zip(self._proj_axes, self._locked_ax_positions):
            ax.set_position(pos)

    def _redraw_projections(self):
        """Lightweight redraw of projection axes only (skip full figure rebuild).

        Call this when only threshold/peaks change, not the spectrogram.
        """
        if not self._proj_axes:
            self._draw_plots()
            return
        combo_idx = self.proj_combo.currentIndex()
        n_proj_total = len(self._projections)
        if combo_idx <= 0:
            show_indices = list(range(n_proj_total))
        else:
            proj_sel = combo_idx - 1
            show_indices = [proj_sel] if proj_sel < n_proj_total else list(range(n_proj_total))
        for i, ax in enumerate(self._proj_axes):
            ax.clear()
            if i < len(show_indices):
                orig_idx = show_indices[i]
                self._draw_single_projection(ax, orig_idx, is_bottom=(i == 0))
            else:
                ax.set_visible(False)
        # Redraw with constrained_layout disabled so locked positions are preserved
        was_cl = self.canvas.fig.get_constrained_layout()
        self.canvas.fig.set_constrained_layout(False)
        self.canvas.fig.canvas.draw()
        self._restore_locked_positions()
        self.canvas.fig.canvas.draw_idle()
        self.canvas.fig.set_constrained_layout(was_cl)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Plotting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_spectrogram_norm(self):
        """Return LogNorm if Log Z is checked, else None (linear)."""
        if not self.logz_checkbox.isChecked():
            return None
        data = self.spectrogram_data
        if data is not None and data.ndim == 2:
            vmin = data[data > 0].min() if np.any(data > 0) else 1e-300
            vmax = data.max()
            if vmax > vmin > 0:
                return LogNorm(vmin=vmin, vmax=vmax)
        return Normalize()

    def _draw_plots(self):
        """Main drawing method.

        When "Show Projections" is unchecked: full-width spectrogram (single panel).
        When checked: spectrogram on left, individual projection subplots on right
        with log-scale y-axis, stacked vertically.
        """
        self.canvas.clear()
        self._proj_axes = []
        fig = self.canvas.fig

        has_spec = self.spectrogram_data is not None or (
            self.time_data is not None and self.freq_data is not None
            and len(self.time_data) > 1 and len(self.freq_data) > 1
        )

        if not has_spec:
            ax_empty = fig.add_subplot(111)
            ax_empty.text(0.5, 0.5, "Load a data file to plot",
                          ha='center', va='center', fontsize=14)
            ax_empty.set_xlim(0, 1)
            ax_empty.set_ylim(0, 1)
            self.canvas.draw()
            return

        show_proj = (self.show_proj_checkbox.isChecked()
                     and self._projections is not None
                     and len(self._projections) > 0)

        if show_proj:
            # ── Two-column layout: spectrogram left, projections right ──
            try:
                left_pct = max(10, min(90, float(self.split_ratio_edit.text())))
            except (ValueError, AttributeError):
                left_pct = 65
            gs = gridspec.GridSpec(1, 2, figure=fig,
                                   width_ratios=[left_pct, 100 - left_pct],
                                   wspace=0.15)
            ax_spec = fig.add_subplot(gs[0])
            self._draw_spectrogram(ax_spec)
            self._apply_axis_limits(ax_spec)
            self._draw_voltage_markers(ax_spec)

            # Right column: projection subplots
            n_proj = len(self._projections)
            combo_idx = self.proj_combo.currentIndex()
            if combo_idx <= 0:
                show_indices = list(range(n_proj))
            else:
                proj_sel = combo_idx - 1
                show_indices = [proj_sel] if proj_sel < n_proj else list(range(n_proj))
            n_display = len(show_indices)
            gs_right = gridspec.GridSpecFromSubplotSpec(
                n_display, 1, subplot_spec=gs[1], hspace=0
            )
            ax_bottom = None
            for i, orig_idx in enumerate(show_indices):
                disp_idx = n_display - 1 - i
                if ax_bottom is None:
                    ax_bottom = fig.add_subplot(gs_right[disp_idx])
                    ax_p = ax_bottom
                else:
                    ax_p = fig.add_subplot(gs_right[disp_idx], sharex=ax_bottom)
                self._proj_axes.append(ax_p)
                self._draw_single_projection(ax_p, orig_idx, is_bottom=(ax_p is ax_bottom))
            # Remove vertical gap between projection subplots (via constrained_layout pads)
            fig.set_constrained_layout_pads(h_pad=0.04 * self._font_scale, hspace=0)
        else:
            # ── Single full-width spectrogram ──
            ax_spec = fig.add_subplot(111)
            self._draw_spectrogram(ax_spec)
            self._apply_axis_limits(ax_spec)
            self._draw_voltage_markers(ax_spec)

        self.canvas.draw()
        if self._proj_axes:
            self._lock_current_positions()

    def _draw_spectrogram(self, ax):
        """Draw the 2D spectrogram on the given axes."""
        if self.spectrogram_data is not None and self.spectrogram_data.ndim == 2:
            extent = None
            if self.time_data is not None and self.freq_data is not None:
                f_min, f_max = self.freq_data.min(), self.freq_data.max()
                t_max = self.time_data.max()
                extent = [f_min, f_max, 0, t_max]

            im = ax.imshow(
                self.spectrogram_data,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap='jet',
                interpolation='bilinear',
                norm=self._get_spectrogram_norm()
            )
            self.canvas.fig.colorbar(im, ax=ax, label="Amplitude")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Time (s)")

        elif self.time_data is not None and self.freq_data is not None:
            n = min(len(self.time_data), len(self.freq_data))
            t = self.time_data[:n]
            f = self.freq_data[:n]

            if n > 10000:
                hb = ax.hexbin(f, t, gridsize=100, cmap='jet', mincnt=1)
                self.canvas.fig.colorbar(hb, ax=ax, label="Counts")
            else:
                ax.scatter(f, t, s=1, c='blue', alpha=0.5)

            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Time (s)")

    def _draw_voltage_markers(self, ax):
        """Draw voltage event markers (red dashed lines) on the spectrogram,
        and projection-end markers (green dashed lines)."""
        # ── Parse offset & dt for projection end times ──
        try:
            offset_s = float(self.proj_offset_edit.text())
        except (ValueError, AttributeError):
            offset_s = 5.0
        try:
            dt_s = float(self.proj_dt_edit.text())
        except (ValueError, AttributeError):
            dt_s = 20.0

        volt_times = None
        if self.voltage_time is not None and len(self.voltage_time) > 0:
            volt_times = self.voltage_time
            self._write_debug(f"DEBUG draw: voltage_time = {volt_times[:min(20, len(volt_times))]}")
        elif self.voltage_data is not None and len(self.voltage_data) > 0:
            volt_times = np.arange(len(self.voltage_data))

        if volt_times is not None:
            xlim = ax.get_xlim()
            text_x = xlim[1] - (xlim[1] - xlim[0]) * 0.02

            for i, vt in enumerate(volt_times):
                # ── Red line at start time ──
                ax.axhline(y=vt, color='red', linestyle='--',
                           linewidth=1.5, alpha=0.7)
                if self.voltage_data is not None and i < len(self.voltage_data):
                    label = f"{self.voltage_data[i]:.3f}"
                    ax.text(text_x, vt, label,
                            color='red', fontsize=round(7 * self._font_scale), fontweight='bold',
                            va='center', ha='right',
                            bbox=dict(boxstyle='round,pad=0.15',
                                      facecolor='white', edgecolor='none',
                                      alpha=0.6))

                # ── Green line at projection end time ──
                end_t = vt + offset_s + dt_s
                ax.axhline(y=end_t, color='lime', linestyle='--',
                           linewidth=1.0, alpha=0.6)

            from matplotlib.lines import Line2D
            red_line = Line2D([0], [0], color='red', linestyle='--',
                              linewidth=1.5, alpha=0.7)
            green_line = Line2D([0], [0], color='lime', linestyle='--',
                                linewidth=1.0, alpha=0.6)
            ax.legend([red_line, green_line],
                      ['Voltage events', 'Proj. end'],
                      loc='upper right', fontsize=round(8 * self._font_scale), ncol=1)

    def _draw_single_projection(self, ax, idx, is_bottom=False):
        """Draw one frequency projection with log y-axis.

        Shows three curves when baseline-removed data is available:
          - grey: raw spectrum
          - orange: estimated baseline
          - blue: baseline-removed spectrum
        Red triangles mark detected peaks.
        """
        freq = self.freq_data
        raw = self._projections[idx]

        # Always show raw spectrum in grey
        floor_raw = max(raw[raw > 0].min() if np.any(raw > 0) else 1e-10, 1e-10)
        ax.semilogy(freq, np.maximum(raw, floor_raw), color='black',
                    linewidth=0.4, alpha=0.6, label='raw')

        has_baseline = (self._projections_baseline is not None
                        and idx < len(self._projections_baseline))
        has_clean = (self._projections_clean is not None
                     and idx < len(self._projections_clean))

        if has_baseline:
            bl = self._projections_baseline[idx]
            floor_bl = max(bl[bl > 0].min() if np.any(bl > 0) else 1e-10, 1e-10)
            ax.semilogy(freq, np.maximum(bl, floor_bl), color='#FF9800',
                        linewidth=0.8, alpha=0.8, label='baseline')

        # ── Threshold curve (only if Show Threshold checked) ──
        thresh_curve = self._get_threshold_curve()
        if thresh_curve is not None and self.show_thresh_checkbox.isChecked():
            floor_th = max(thresh_curve[thresh_curve > 0].min() if np.any(thresh_curve > 0) else 1e-10, 1e-10)
            ax.semilogy(freq, np.maximum(thresh_curve, floor_th), color='red',
                        linewidth=0.6, linestyle='-', alpha=0.5, label='threshold')

        # Determine which data source to use for peak amplitudes
        pk_source = raw
        pk_floor = floor_raw
        if has_clean:
            clean = self._projections_clean[idx]
            floor_cl = max(clean[clean > 0].min() if np.any(clean > 0) else 1e-10, 1e-10)
            ax.semilogy(freq, np.maximum(clean, floor_cl), color='#2196F3',
                        linewidth=0.8, label='cleaned')
            pk_source = clean
            pk_floor = floor_cl

        # ── Mark peaks (always, independent of baseline) ──
        if self._projections_peaks is not None and idx < len(self._projections_peaks):
            peaks = self._projections_peaks[idx]
            mask = (self._projections_peak_masks is not None
                    and idx < len(self._projections_peak_masks)
                    and self._projections_peak_masks[idx])
            if len(peaks) > 0:
                if mask is not False and np.any(mask):
                    shown = peaks[mask]
                    pk_amp = np.maximum(pk_source[shown], pk_floor)
                    ax.scatter(freq[shown], pk_amp, marker='v', color='red',
                               s=20, zorder=5)

                    # Peaks peak data for integration range & background info
                    pk_data = (self._projections_peak_data[idx]
                               if self._projections_peak_data is not None
                               and idx < len(self._projections_peak_data)
                               else None)

                    for j_in_shown, pf in enumerate(freq[shown]):
                        ax.axvline(x=pf, color='red', linestyle=':',
                                   linewidth=0.7, alpha=0.5)
                        # Map shown peak back to full peak index
                        full_pk_idx = np.where(peaks == shown[j_in_shown])[0][0]

                        # ── Draw integration range (FWHM) as green span ──
                        if pk_data is not None:
                            lidx_arr = pk_data.get('left_idxs', None)
                            ridx_arr = pk_data.get('right_idxs', None)
                            if lidx_arr is not None and full_pk_idx < len(lidx_arr):
                                li = int(lidx_arr[full_pk_idx])
                                ri = int(ridx_arr[full_pk_idx])
                                ax.axvspan(freq[li], freq[ri], color='green', alpha=0.08, zorder=2)

                            # ── Draw background region as orange span ──
                            bg_li_arr = pk_data.get('bg_left_idxs', None)
                            bg_ri_arr = pk_data.get('bg_right_idxs', None)
                            if bg_li_arr is not None and full_pk_idx < len(bg_li_arr):
                                bg_li = int(bg_li_arr[full_pk_idx])
                                bg_ri = int(bg_ri_arr[full_pk_idx])
                                ax.axvspan(freq[bg_li], freq[bg_ri], color='orange', alpha=0.1, zorder=2)

                            # ── Compute & annotate per-frame average area ──
                            pk_areas = pk_data.get('areas', None)
                            bg_levels = pk_data.get('bg_levels', None)
                            n_frames = pk_data.get('n_frames', 1)
                            if pk_areas is not None and full_pk_idx < len(pk_areas):
                                net_area = pk_areas[full_pk_idx]
                                if bg_levels is not None and full_pk_idx < len(bg_levels):
                                    net_area -= bg_levels[full_pk_idx]
                                per_frame = net_area / max(1, n_frames)
                                ax.annotate(f'A/f={per_frame:.2e}',
                                            (pf, pk_amp[j_in_shown]),
                                            textcoords='offset points',
                                            xytext=(-25, -14), fontsize=round(4.5 * self._font_scale),
                                            color='#0000CC', fontweight='bold',
                                            ha='center', va='top',
                                            bbox=dict(boxstyle='round,pad=0.15',
                                                      facecolor='white', edgecolor='none', alpha=0.6))

                        # Label harmonic number above peak
                        if pk_data is not None:
                            harr = pk_data.get('har_n', None)
                            if harr is not None and len(harr) == len(peaks):
                                if full_pk_idx < len(harr):
                                    hn = harr[full_pk_idx]
                                    if hn > 0:
                                        ax.annotate(f'n={hn}',
                                                    (pf, pk_amp[j_in_shown]),
                                                    textcoords='offset points',
                                                    xytext=(0, 8), fontsize=round(5 * self._font_scale),
                                                    color='#E91E63', fontweight='bold',
                                                    ha='center', va='bottom')
                # Store peak data for right-click context menu
                ax._peak_idx = idx
                self._connect_projection_right_click(ax)

        # Labels
        volt_label = ""
        if self._proj_voltages is not None and idx < len(self._proj_voltages):
            volt_label = f"V={self._proj_voltages[idx]:.3f}V"
        time_label = ""
        if self.voltage_time is not None and idx < len(self.voltage_time):
            time_label = f"t={self.voltage_time[idx]:.0f}s"

        # Append harmonic fit info if available
        har_info = ""
        if self._projections_har_fit is not None and idx < len(self._projections_har_fit):
            fit = self._projections_har_fit[idx]
            if fit is not None:
                har_info = f"  f0={fit['f0']:.6f}MHz  Δf={fit['offset']:.2e}Hz"

        fs = self._font_scale
        label_text = f"{time_label}  {volt_label}{har_info}"
        if is_bottom:
            ax.set_xlabel("Freq (MHz)", fontsize=round(7 * fs))
        else:
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelbottom=False)
        # Title inside the plot for ALL subplots
        ax.set_title("")
        ax.text(0.02, 0.96, label_text, transform=ax.transAxes,
                fontsize=round(6 * fs), fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.6, edgecolor='none'))
        ax.tick_params(axis='both', labelsize=round(6 * fs))
        ax.grid(True, alpha=0.3)

        # Legend (compact)
        if has_baseline or has_clean:
            ax.legend(fontsize=max(4, round(5 * fs)), loc='upper right', ncol=1,
                      handlelength=1.5, handletextpad=0.3,
                      borderpad=0.2, labelspacing=0.2)


# ──────────────────────────────────────────────
# Standalone test entry
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = AFCCalculatorDialog()
    dialog.show()
    sys.exit(app.exec_())
