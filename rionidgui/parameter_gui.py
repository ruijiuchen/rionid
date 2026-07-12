from PyQt5.QtWidgets import (QApplication,QWidget, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox,
                             QFileDialog, QMessageBox, QGroupBox, QToolButton,
                             QDialog, QDoubleSpinBox, QDialogButtonBox, QSpinBox,
                             QProgressBar)
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QFont
import toml
import argparse
import logging as log
from loguru import logger
from rionidgui.gui_controller import import_controller
from rionidgui.afc_calculator import AFCCalculatorDialog
from rionid.importdata import ImportData
import sys
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel, QLineEdit, QHBoxLayout, QScrollArea
import time

log.basicConfig(level=log.DEBUG)
common_font = QFont()
common_font.setPointSize(12) #font size


class HistogramConfigDialog(QDialog):
    """柱状图参数配置对话框：设置 x 轴范围、bin 宽度，支持预览。"""
    def __init__(self, raw_freqs_hz, parent=None):
        super().__init__(parent)
        self.raw_freqs_hz = raw_freqs_hz
        self.setWindowTitle("柱状图参数设置")
        self.setMinimumWidth(400)

        # 频率范围 (MHz)
        self.freq_min_hz = float(raw_freqs_hz.min())
        self.freq_max_hz = float(raw_freqs_hz.max())
        freq_range_mhz = (self.freq_max_hz - self.freq_min_hz) / 1e6

        layout = QVBoxLayout()

        # --- x_min ---
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("x 轴最小值 (MHz):"))
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setDecimals(4)
        self.spin_min.setRange(0, 10000)
        self.spin_min.setValue(self.freq_min_hz / 1e6)
        h1.addWidget(self.spin_min)
        layout.addLayout(h1)

        # --- x_max ---
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("x 轴最大值 (MHz):"))
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setDecimals(4)
        self.spin_max.setRange(0, 10000)
        self.spin_max.setValue(self.freq_max_hz / 1e6)
        h2.addWidget(self.spin_max)
        layout.addLayout(h2)

        # --- bin 宽度 ---
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Bin 宽度 (MHz):"))
        self.spin_bw = QDoubleSpinBox()
        self.spin_bw.setDecimals(6)
        self.spin_bw.setRange(1e-6, 1000)
        self.spin_bw.setValue(freq_range_mhz / 200)  # default ~200 bins
        h3.addWidget(self.spin_bw)
        layout.addLayout(h3)

        # 显示 bins 数量（只读）
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("→ 对应 bins 数:"))
        self.label_nbins = QLabel("200")
        self.label_nbins.setFont(common_font)
        h4.addWidget(self.label_nbins)
        h4.addStretch()
        layout.addLayout(h4)

        # 连接信号：改变 x_min/x_max/bin_width 时更新 bins 数显示
        self.spin_min.valueChanged.connect(self._update_nbins)
        self.spin_max.valueChanged.connect(self._update_nbins)
        self.spin_bw.valueChanged.connect(self._update_nbins)

        # --- 按钮 ---
        btn_layout = QHBoxLayout()
        self.btn_preview = QPushButton("预览")
        self.btn_preview.clicked.connect(self._preview)
        btn_layout.addWidget(self.btn_preview)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        btn_layout.addWidget(button_box)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self._update_nbins()

    def _update_nbins(self):
        """根据当前范围/宽度计算并显示 bins 数"""
        fmin = self.spin_min.value()
        fmax = self.spin_max.value()
        bw = self.spin_bw.value()
        if bw > 0 and fmax > fmin:
            nbins = int((fmax - fmin) / bw)
            self.label_nbins.setText(str(max(nbins, 1)))
        else:
            self.label_nbins.setText("—")

    def _preview(self):
        """用当前参数绘制柱状图预览"""
        import matplotlib.pyplot as plt
        fmin = self.spin_min.value() * 1e6
        fmax = self.spin_max.value() * 1e6
        bw = self.spin_bw.value() * 1e6
        if bw <= 0 or fmax <= fmin:
            QMessageBox.warning(self, "参数错误", "请确保 x_max > x_min 且 bin 宽度 > 0")
            return
        nbins = max(int((fmax - fmin) / bw), 1)

        # 过滤频率
        mask = (self.raw_freqs_hz >= fmin) & (self.raw_freqs_hz <= fmax)
        filtered = self.raw_freqs_hz[mask]
        if len(filtered) == 0:
            QMessageBox.warning(self, "无数据", "该范围内没有数据点")
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(filtered / 1e6, bins=nbins, range=(fmin/1e6, fmax/1e6),
                color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Counts")
        ax.set_title(f"柱状图预览 (bins={nbins}, n={len(filtered)})")
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        plt.show()

    def get_params(self):
        """返回 (freq_min_Hz, freq_max_Hz, bins)"""
        fmin = self.spin_min.value() * 1e6
        fmax = self.spin_max.value() * 1e6
        bw = self.spin_bw.value() * 1e6
        nbins = max(int((fmax - fmin) / bw), 1)
        return fmin, fmax, nbins


class RionID_GUI(QWidget):
    visualization_signal = pyqtSignal(object)
    overlay_sim_signal    = pyqtSignal(object)           # new—just overlays one simulation
    clear_sim_signal      = pyqtSignal()           # ← new
    signalError           = pyqtSignal(str)
    def __init__(self, plot_widget, *args, **kwargs):
        super().__init__()
        self.visualization_widget = plot_widget
        self._stop_SMS_pid = False
        self._stop_IMS_pid = False
        self._afc_time_file = ""
        self._afc_voltage_file = ""
        self._afc_xmin = ""
        self._afc_xmax = ""
        self._afc_ymin = ""
        self._afc_ymax = ""
        self._afc_logz = False
        self._afc_show_proj = False
        self._afc_offset = "5"
        self._afc_dt = "20"
        self._afc_split = "65"
        self._afc_threshold_path = ""
        self._afc_peak_dist = "3"
        self._afc_show_thresh = True
        self.initUI()
        self.load_parameters()  # Load parameters after initializing UI
        
    @pyqtSlot()
    def onPlotClicked(self):
        """Called when user clicks inside the plot area."""
        self._stop_SMS_pid = True
        self._stop_IMS_pid = True
    
    def initUI(self):
        self.setup_layout()

    def load_parameters(self, filepath='parameters_cache.toml'):
        try:
            with open(filepath, 'r') as f:
                parameters = toml.load(f)
                self.datafile_edit.setText(parameters.get('datafile', ''))
                self.filep_edit.setText(parameters.get('filep', ''))
                self.remove_baseline_checkbox.setChecked(parameters.get('remove_baseline_checkbox', True))
                self.psd_baseline_removed_l_edit.setText(parameters.get('psd_baseline_removed_l', ''))
                self.psd_baseline_removed_ratio_edit.setText(parameters.get('psd_baseline_removed_ratio', ''))
                self.alphap_edit.setText(parameters.get('alphap', ''))
                self.alphap_min_edit.setText(parameters.get('alphap_min', ''))
                self.alphap_max_edit.setText(parameters.get('alphap_max', ''))
                self.alphap_step_edit.setText(parameters.get('alphap_step', ''))
                self.threshold_edit.setText(str(parameters.get('threshold', '')))
                self.matching_freq_min_edit.setText(str(parameters.get('matching_freq_min', '')))
                self.matching_freq_max_edit.setText(str(parameters.get('matching_freq_max', '')))
                self.fref_min_edit.setText(parameters.get('fref_min', ''))
                self.fref_max_edit.setText(parameters.get('fref_max', ''))
                self.peak_thresh_value = float(parameters.get('peak_threshold_pct', 0.05))
                self.min_distance_edit.setText(str(parameters.get('min_distance', '')))
                self.harmonics_edit.setText(parameters.get('harmonics', ''))
                self.ref_harmonic_edit.setText(parameters.get('ref_harmonic', ''))
                self.refion_edit.setText(parameters.get('refion', ''))
                self.highlight_ions_edit.setText(parameters.get('highlight_ions', ''))
                self.circumference_edit.setText(parameters.get('circumference', ''))
                self.mode_combo.setCurrentText(parameters.get('mode', 'Frequency'))
                self.sim_scalingfactor_edit.setText(parameters.get('sim_scalingfactor', ''))
                self.value_edit.setText(parameters.get('value', ''))
                self.reload_data_checkbox.setChecked(parameters.get('reload_data', True))
                self.nions_edit.setText(parameters.get('nions', ''))
                self.simulation_result_edit.setText(parameters.get('simulation_result', ''))
                self.matched_result_edit.setText(parameters.get('matched_result', ''))
                self.brho_min_edit.setText(parameters.get('brho_min', ''))
                self.brho_max_edit.setText(parameters.get('brho_max', ''))
                self.brho_step_edit.setText(parameters.get('brho_step', ''))
                self.circ_min_edit.setText(parameters.get('circ_min', ''))
                self.circ_max_edit.setText(parameters.get('circ_max', ''))
                self.circ_step_edit.setText(parameters.get('circ_step', ''))
                # 恢复柱状图参数
                self._hist_freq_min = parameters.get('hist_freq_min')
                self._hist_freq_max = parameters.get('hist_freq_max')
                self._hist_bins = parameters.get('hist_bins')
                # 恢复 AFC&gtr Calculator 参数
                self._afc_time_file = parameters.get('afc_time_file', '')
                self._afc_voltage_file = parameters.get('afc_voltage_file', '')
                self._afc_xmin = parameters.get('afc_xmin', '')
                self._afc_xmax = parameters.get('afc_xmax', '')
                self._afc_ymin = parameters.get('afc_ymin', '')
                self._afc_ymax = parameters.get('afc_ymax', '')
                self._afc_logz = parameters.get('afc_logz', False)
                self._afc_show_proj = parameters.get('afc_show_proj', False)
                self._afc_offset = parameters.get('afc_offset', '5')
                self._afc_dt = parameters.get('afc_dt', '20')
                self._afc_split = parameters.get('afc_split', '65')
                self._afc_threshold_path = parameters.get('afc_threshold_path', '')
                self._afc_peak_dist = parameters.get('afc_peak_dist', '3')
                self.saved_data=None
                
        except FileNotFoundError:
            pass  # No parameters file exists yet

    def save_parameters(self, filepath='parameters_cache.toml'):
        parameters = {
            'datafile': self.datafile_edit.text(),
            'filep': self.filep_edit.text(),
            'remove_baseline_checkbox': self.remove_baseline_checkbox.isChecked(),
            'psd_baseline_removed_l': self.psd_baseline_removed_l_edit.text(),
            'psd_baseline_removed_ratio': self.psd_baseline_removed_ratio_edit.text(),
            'alphap': self.alphap_edit.text(),
            'alphap_min': self.alphap_min_edit.text(),
            'alphap_max': self.alphap_max_edit.text(),
            'alphap_step': self.alphap_step_edit.text(),
            'threshold': self.threshold_edit.text(),
            'matching_freq_min': self.matching_freq_min_edit.text(),
            'matching_freq_max': self.matching_freq_max_edit.text(),
            'peak_threshold_pct': self.peak_thresh_value if hasattr(self, 'peak_thresh_value') else 0.05,
            'min_distance': float(self.min_distance_edit.text()) if self.min_distance_edit.text().strip() else 0.0,
            'fref_min': self.fref_min_edit.text(),
            'fref_max': self.fref_max_edit.text(),
            'harmonics': self.harmonics_edit.text(),
            'ref_harmonic': self.ref_harmonic_edit.text(),
            'refion': self.refion_edit.text(),
            'highlight_ions': self.highlight_ions_edit.text(),
            'circumference': self.circumference_edit.text(),
            'mode': self.mode_combo.currentText(),
            'value': self.value_edit.text(),
            'sim_scalingfactor': self.sim_scalingfactor_edit.text(),
            'reload_data': self.reload_data_checkbox.isChecked(),
            'nions': self.nions_edit.text(),
            'simulation_result': self.simulation_result_edit.text(),
            'matched_result': self.matched_result_edit.text(),
            'brho_min': self.brho_min_edit.text(),
            'brho_max': self.brho_max_edit.text(),
            'brho_step': self.brho_step_edit.text(),
            'circ_min': self.circ_min_edit.text(),
            'circ_max': self.circ_max_edit.text(),
            'circ_step': self.circ_step_edit.text(),
            'hist_freq_min': getattr(self, '_hist_freq_min', None),
            'hist_freq_max': getattr(self, '_hist_freq_max', None),
            'hist_bins': getattr(self, '_hist_bins', None),
            'afc_time_file': getattr(self, '_afc_time_file', ''),
            'afc_voltage_file': getattr(self, '_afc_voltage_file', ''),
            'afc_xmin': getattr(self, '_afc_xmin', ''),
            'afc_xmax': getattr(self, '_afc_xmax', ''),
            'afc_ymin': getattr(self, '_afc_ymin', ''),
            'afc_ymax': getattr(self, '_afc_ymax', ''),
            'afc_logz': getattr(self, '_afc_logz', False),
            'afc_show_proj': getattr(self, '_afc_show_proj', False),
            'afc_offset': getattr(self, '_afc_offset', '5'),
            'afc_dt': getattr(self, '_afc_dt', '20'),
            'afc_split': getattr(self, '_afc_split', '65'),
            'afc_threshold_path': getattr(self, '_afc_threshold_path', ''),
            'afc_peak_dist': getattr(self, '_afc_peak_dist', '3'),
        }
        with open(filepath, 'w') as f:
            toml.dump(parameters, f)
            
    def _overlay_simulation(self, data):
        # this won’t clear the existing curves
        self.visualization_widget.plot_simulated_data(data)
        # force Qt to repaint so you actually see each new curve
        QApplication.processEvents()
        
    def setup_layout(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
    
        scroll_content = QWidget()
        self.vbox = QVBoxLayout(scroll_content)
        self.scroll_area.setWidget(scroll_content)       
        # Set scroll_area as main layout of this QWidget
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll_area)

        # ──── AFC&gtr Calculator button at the very top ────
        self.afc_calculator_button = QPushButton('AFC && gtr Calculator')
        self.afc_calculator_button.setFont(common_font)
        self.afc_calculator_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.afc_calculator_button.clicked.connect(self._open_afc_calculator)
        hbox_afc = QHBoxLayout()
        hbox_afc.addWidget(self.afc_calculator_button)
        self.vbox.addLayout(hbox_afc)

        
        self.setup_file_selection()
        self.setup_parameters()

    def setup_file_selection(self):
        self.datafile_label = QLabel('Experimental Data File:')
        self.datafile_edit = QLineEdit()
        self.datafile_label.setFont(common_font)
        self.datafile_edit.setFont(common_font)
        self.datafile_button = QPushButton('Browse')
        self.datafile_button.setFont(common_font)
        self.datafile_button.clicked.connect(self.browse_datafile)

        self.filep_label = QLabel('.lpp File:')
        self.filep_edit = QLineEdit()
        self.filep_label.setFont(common_font)
        self.filep_edit.setFont(common_font)
        self.filep_button = QPushButton('Browse')
        self.filep_button.setFont(common_font)

        self.filep_button.clicked.connect(self.browse_lppfile)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.datafile_label)
        hbox1.addWidget(self.datafile_edit)
        hbox1.addWidget(self.datafile_button)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.filep_label)
        hbox2.addWidget(self.filep_edit)
        hbox2.addWidget(self.filep_button)

        self.vbox.addLayout(hbox1)
        self.vbox.addLayout(hbox2)

    def enterPlotPickMode(self, target: QLineEdit):
        self._plot_pick_target = target
        self._plot_pick_original_style = target.styleSheet()
        # Highlight the field in gray to indicate “pick” mode
        target.setStyleSheet("background-color: lightgray;")
        
        pw = self.visualization_widget.getPlotWidget()
        pw.setCursor(Qt.CrossCursor)
        # now hook the scene-click so your _onPlotPicked(ev) gets the MouseClickEvent
        pw.scene().sigMouseClicked.connect(self._onPlotPicked)

    def _onPlotPicked(self, ev):
        # same as before
        vb = self.visualization_widget.getViewBox()
        # Ignore clicks outside the plot area
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
            
        x_val = vb.mapSceneToView(ev.scenePos()).x()*1e6
        self._plot_pick_target.setText(f"{x_val:.2f}")

        # Restore the QLineEdit’s original stylesheet
        self._plot_pick_target.setStyleSheet(self._plot_pick_original_style)

        pw = self.visualization_widget.getPlotWidget()
        pw.unsetCursor()
        pw.scene().sigMouseClicked.disconnect(self._onPlotPicked)

        # Clear temporary attributes
        self._plot_pick_target = None
        self._plot_pick_original_style = None
        
    def setup_parameters(self):
        # ── 载入数据按钮 ──
        self.load_data_button = QPushButton('📂 Load Data')
        self.load_data_button.setFont(common_font)
        self.load_data_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.load_data_button.clicked.connect(self.load_data)
        hbox_load = QHBoxLayout()
        hbox_load.addWidget(self.load_data_button)
        self.vbox.addLayout(hbox_load)

        # Is the experimental data reloaded?
        self.reload_data_checkbox = QCheckBox('Reload Experimental Data')
        self.reload_data_checkbox.setFont(common_font)
        self.reload_data_checkbox.setChecked(True)
    
        # Is the experimental data reloaded?
        self.remove_baseline_checkbox = QCheckBox('Remove baseline(Q. Wang et al. NUCL SCI TECH, 33: 148 (2022).)')
        self.remove_baseline_checkbox.setFont(common_font)
        self.remove_baseline_checkbox.setChecked(True)
        
        # psd_baseline_removed_l estimate parameters input
        self.psd_baseline_removed_l_label = QLabel('l(e.g. 1000000):')
        self.psd_baseline_removed_l_edit = QLineEdit()
        self.psd_baseline_removed_l_label.setFont(common_font)
        self.psd_baseline_removed_l_edit.setFont(common_font)
    
        # psd_baseline_removed_ratio estimate parameters input
        self.psd_baseline_removed_ratio_label = QLabel('ratio(e.g. 1e-6):')
        self.psd_baseline_removed_ratio_edit = QLineEdit()
        self.psd_baseline_removed_ratio_label.setFont(common_font)
        self.psd_baseline_removed_ratio_edit.setFont(common_font)
    
        # αₚ main input
        self.alphap_label = QLabel('<i>α<sub>p</sub> or γ<sub>t</sub> :</i>')
        self.alphap_edit = QLineEdit()
        self.alphap_label.setFont(common_font)
        self.alphap_edit.setFont(common_font)
    
        # ——— Other parameters ———
        # Harmonic
        self.harmonics_label = QLabel('Harmonics (e.g. 124 125 126):')
        self.harmonics_edit = QLineEdit()
        self.harmonics_label.setFont(common_font)
        self.harmonics_edit.setFont(common_font)
        # Reference ion
        self.refion_label = QLabel('Reference ion (AAEl+QQ):')
        self.refion_edit = QLineEdit()
        self.refion_label.setFont(common_font)
        self.refion_edit.setFont(common_font)
        # Highlighted ion
        self.highlight_ions_label = QLabel('Highlight ions (comma-separated):')
        self.highlight_ions_edit = QLineEdit()
        self.highlight_ions_label.setFont(common_font)
        self.highlight_ions_edit.setFont(common_font)
        # Mode selection
        self.mode_label = QLabel('Mode:')
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Frequency', 'Bρ', 'Kinetic Energy'])
        self.mode_label.setFont(common_font)
        self.mode_combo.setFont(common_font)
        # The circumference of the storage ring
        self.circumference_label = QLabel('Circumference (m):')
        self.circumference_edit = QLineEdit()
        self.circumference_label.setFont(common_font)
        self.circumference_edit.setFont(common_font)
        # Scaling factor
        self.sim_scalingfactor_label = QLabel('Scaling factor:')
        self.sim_scalingfactor_edit = QLineEdit()
        self.sim_scalingfactor_label.setFont(common_font)
        self.sim_scalingfactor_edit.setFont(common_font)
        # value
        self.value_label = QLabel('Value:')
        self.value_edit = QLineEdit()
            
        # psd_baseline_removed_l parameters input
        self.vbox.addWidget(self.remove_baseline_checkbox)
        hbox_psd_baseline_removed_l = QHBoxLayout()
        hbox_psd_baseline_removed_l.addWidget(self.psd_baseline_removed_l_label)
        hbox_psd_baseline_removed_l.addWidget(self.psd_baseline_removed_l_edit)
        self.vbox.addLayout(hbox_psd_baseline_removed_l)

        hbox_psd_baseline_removed_ratio = QHBoxLayout()
        hbox_psd_baseline_removed_ratio.addWidget(self.psd_baseline_removed_ratio_label)
        hbox_psd_baseline_removed_ratio.addWidget(self.psd_baseline_removed_ratio_edit)
        self.vbox.addLayout(hbox_psd_baseline_removed_ratio)

        # Apply baseline once button
        self.apply_baseline_button = QPushButton('Apply Baseline Once')
        self.apply_baseline_button.setFont(common_font)
        self.apply_baseline_button.clicked.connect(self._apply_baseline_once)
        hbox_apply_baseline = QHBoxLayout()
        hbox_apply_baseline.addWidget(self.apply_baseline_button)
        self.vbox.addLayout(hbox_apply_baseline)

        # ──── Threshold Profile controls ────
        self.thresh_profile_label = QLabel('Threshold Profile:')
        self.thresh_profile_label.setFont(common_font)
        self.thresh_profile_edit = QLineEdit()
        self.thresh_profile_edit.setPlaceholderText("Select height_thresh.csv path...")
        self.thresh_profile_edit.setFont(common_font)
        self.thresh_profile_edit.textChanged.connect(
            lambda path: self.visualization_widget._apply_threshold_path(path, refresh=False)
        )

        self.thresh_browse_button = QPushButton('Browse')
        self.thresh_browse_button.setFont(common_font)
        self.thresh_browse_button.clicked.connect(self._on_thresh_browse)

        self.thresh_toggle_button = QPushButton('Start Click Threshold')
        self.thresh_toggle_button.setFont(common_font)
        self.thresh_toggle_button.clicked.connect(self._on_toggle_threshold_click)

        # Connect signal from the visualization widget to update button state
        self.visualization_widget.thresholdClickModeChanged.connect(self._on_threshold_click_mode_changed)

        hbox_thresh_profile = QHBoxLayout()
        hbox_thresh_profile.addWidget(self.thresh_profile_label)
        hbox_thresh_profile.addWidget(self.thresh_profile_edit)
        self.vbox.addLayout(hbox_thresh_profile)

        self.thresh_clear_button = QPushButton('Clear')
        self.thresh_clear_button.setFont(common_font)
        self.thresh_clear_button.clicked.connect(self._on_thresh_clear)

        hbox_thresh_buttons = QHBoxLayout()
        hbox_thresh_buttons.addWidget(self.thresh_browse_button)
        hbox_thresh_buttons.addWidget(self.thresh_toggle_button)
        hbox_thresh_buttons.addWidget(self.thresh_clear_button)
        self.vbox.addLayout(hbox_thresh_buttons)

        # ──── Find Peaks button ────
        self.find_peaks_button = QPushButton('🔍 Find Peaks')
        self.find_peaks_button.setFont(common_font)
        self.find_peaks_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.find_peaks_button.clicked.connect(self._find_peaks)
        hbox_find_peaks = QHBoxLayout()
        hbox_find_peaks.addWidget(self.find_peaks_button)
        self.vbox.addLayout(hbox_find_peaks)

        hbox_mode = QHBoxLayout()
        hbox_mode.addWidget(self.mode_label)
        hbox_mode.addWidget(self.mode_combo)
        hbox_mode.addWidget(self.value_edit)
        self.vbox.addLayout(hbox_mode)

        # circumference
        hbox_circ = QHBoxLayout()
        hbox_circ.addWidget(self.circumference_label)
        hbox_circ.addWidget(self.circumference_edit)
        self.vbox.addLayout(hbox_circ)
        

        hbox_alphap = QHBoxLayout()
        hbox_alphap.addWidget(self.alphap_label)
        hbox_alphap.addWidget(self.alphap_edit)
        self.vbox.addLayout(hbox_alphap)



        # --- Reference harmonic number and auto-calculation display ---
        self.ref_harmonic_label = QLabel('Ref. Harmonic:')
        self.ref_harmonic_label.setFont(common_font)
        self.ref_harmonic_edit = QLineEdit()
        self.ref_harmonic_edit.setFont(common_font)
        # Only allow a single positive integer (no spaces, no multiple numbers)
        from PyQt5.QtGui import QIntValidator
        self.ref_harmonic_edit.setValidator(QIntValidator(1, 100000, self))

        self.f0_label = QLabel('Fundamental f0 (Hz):')
        self.f0_label.setFont(common_font)
        self.f0_display = QLineEdit()
        self.f0_display.setFont(common_font)
        self.f0_display.setReadOnly(True)

        self.harmonic_freqs_label = QLabel('Harmonic Frequencies (Hz):')
        self.harmonic_freqs_label.setFont(common_font)
        self.harmonic_freqs_display = QLineEdit()
        self.harmonic_freqs_display.setFont(common_font)
        self.harmonic_freqs_display.setReadOnly(True)

        # --- Layouts in desired order ---
        # 1) Reference Harmonic
        hbox_ref_harmonic = QHBoxLayout()
        hbox_ref_harmonic.addWidget(self.ref_harmonic_label)
        hbox_ref_harmonic.addWidget(self.ref_harmonic_edit)
        self.vbox.addLayout(hbox_ref_harmonic)

        # 2) Fundamental frequency (calculated from ref_harmonic)
        hbox_f0 = QHBoxLayout()
        hbox_f0.addWidget(self.f0_label)
        hbox_f0.addWidget(self.f0_display)
        self.vbox.addLayout(hbox_f0)

        # 3) Reference Ion
        hbox_refion = QHBoxLayout()
        hbox_refion.addWidget(self.refion_label)
        hbox_refion.addWidget(self.refion_edit)
        self.vbox.addLayout(hbox_refion)

        # 4) Harmonics
        hbox_harmonics = QHBoxLayout()
        hbox_harmonics.addWidget(self.harmonics_label)
        hbox_harmonics.addWidget(self.harmonics_edit)
        self.vbox.addLayout(hbox_harmonics)

        # 5) Harmonic Frequencies (calculated from harmonics)
        hbox_harmonic_freqs = QHBoxLayout()
        hbox_harmonic_freqs.addWidget(self.harmonic_freqs_label)
        hbox_harmonic_freqs.addWidget(self.harmonic_freqs_display)
        self.vbox.addLayout(hbox_harmonic_freqs)

        # --- Connect auto-calculation signals ---
        self.mode_combo.currentTextChanged.connect(self._update_harmonic_calculation)
        self.value_edit.textChanged.connect(self._update_harmonic_calculation)
        self.ref_harmonic_edit.textChanged.connect(self._update_harmonic_calculation)
        self.harmonics_edit.textChanged.connect(self._update_harmonic_calculation)

        self.vbox.addWidget(self.highlight_ions_label)
        self.vbox.addWidget(self.highlight_ions_edit)

        # Next, pack scaling‐factor and Run button together
        hbox_sf = QHBoxLayout()
        hbox_sf.addWidget(self.sim_scalingfactor_label)
        hbox_sf.addWidget(self.sim_scalingfactor_edit)
        # ——— Add the Run button *before* the Optional Features section ———
        self.vbox.addLayout(hbox_sf)

        self.min_distance_label = QLabel('Peak min distance (Hz):')
        self.min_distance_label.setFont(common_font)
        self.min_distance_edit  = QLineEdit()
        self.min_distance_edit.setFont(common_font)
        hbox_peak_min_distance = QHBoxLayout()
        hbox_peak_min_distance.addWidget(self.min_distance_label)
        hbox_peak_min_distance.addWidget(self.min_distance_edit)
        self.vbox.addLayout(hbox_peak_min_distance)
        # 
        hbox_matching_freq_min = QHBoxLayout()
        self.matching_freq_min_label = QLabel('Peak search range min. (Hz):')
        self.matching_freq_min_label.setFont(common_font)
        self.matching_freq_min_edit = QLineEdit()
        self.matching_freq_min_edit.setFont(common_font)
        hbox_matching_freq_min.addWidget(self.matching_freq_min_edit)
        self.pick_matching_freq_min_button = QPushButton('Pick')
        self.pick_matching_freq_min_button.setFont(common_font)
        self.pick_matching_freq_min_button.clicked.connect(
            lambda: self.enterPlotPickMode(target=self.matching_freq_min_edit)
        )        
        hbox_matching_freq_min.addWidget(self.pick_matching_freq_min_button)
        self.vbox.addWidget(self.matching_freq_min_label)
        self.vbox.addLayout(hbox_matching_freq_min)
         
        hbox_matching_freq_max = QHBoxLayout()
        self.matching_freq_max_label = QLabel('Peak search range max. (Hz):')
        self.matching_freq_max_label.setFont(common_font)
        self.matching_freq_max_edit = QLineEdit()
        self.matching_freq_max_edit.setFont(common_font)
        self.vbox.addWidget(self.matching_freq_max_label)
        hbox_matching_freq_max.addWidget(self.matching_freq_max_edit)
        self.pick_matching_freq_max_button = QPushButton('Pick')
        self.pick_matching_freq_max_button.setFont(common_font)
        self.pick_matching_freq_max_button.clicked.connect(
            lambda: self.enterPlotPickMode(target=self.matching_freq_max_edit)
        )        
        hbox_matching_freq_max.addWidget(self.pick_matching_freq_max_button)
        self.vbox.addLayout(hbox_matching_freq_max)

        # Matching threshold
        hbox_threshold = QHBoxLayout()
        self.threshold_label = QLabel('Sim. - Exp. max. distance (Hz):')
        self.threshold_label.setFont(common_font)
        self.threshold_edit  = QLineEdit()
        self.threshold_edit.setFont(common_font)
        hbox_threshold.addWidget(self.threshold_label)
        hbox_threshold.addWidget(self.threshold_edit)
        self.vbox.addLayout(hbox_threshold)

        # Optional feature fold group
        self.nions_label = QLabel('Number of ions to display:')
        self.nions_edit = QLineEdit()
        hbox_nions = QHBoxLayout()
        hbox_nions.addWidget(self.nions_label)
        hbox_nions.addWidget(self.nions_edit)
        self.vbox.addLayout(hbox_nions)

        self.correction_label = QLabel('Second-order correction (a0, a1, a2):')
        self.correction_edit = QLineEdit()
        self.vbox.addWidget(self.correction_label)
        self.vbox.addWidget(self.correction_edit)
        
        self.run_button = QPushButton('Run')
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_button.clicked.connect(self.run_script)
        hbox_run_button = QHBoxLayout()
        hbox_run_button.addWidget(self.run_button)
        self.vbox.addLayout(hbox_run_button)



        
        # ——— SMS mode 设置 ———
        # αₚ scan range
        self.alphap_min_label = QLabel('<i>α<sub>p</sub> or γ<sub>t</sub> min:</i>')
        self.alphap_min_edit  = QLineEdit()
        self.alphap_min_label.setFont(common_font)
        self.alphap_min_edit.setFont(common_font)
    
        self.alphap_max_label = QLabel('<i>α<sub>p</sub> or γ<sub>t</sub> max:</i>')
        self.alphap_max_edit  = QLineEdit()
        self.alphap_max_label.setFont(common_font)
        self.alphap_max_edit.setFont(common_font)
    
        self.alphap_step_label = QLabel('<i>α<sub>p</sub> or γ<sub>t</sub> step:</i>')
        self.alphap_step_edit  = QLineEdit()
        self.alphap_step_label.setFont(common_font)
        self.alphap_step_edit.setFont(common_font)
    

        self.fref_min_label = QLabel('Reference frequency min (Hz):')
        self.fref_min_label.setFont(common_font)
        self.fref_min_edit  = QLineEdit()
        self.fref_min_edit.setFont(common_font)
        # QPushButton to let the user click on the plot and capture the x-coordinate
        # of the desired minimum reference frequency.
        # When clicked, this button should switch into “pick mode” so that
        # the next mouse click on the right-hand plot area is converted into
        # a frequency value and placed into fref_min_edit.
        self.pick_fref_min_button = QPushButton('Pick')
        self.pick_fref_min_button.setFont(common_font)
        self.pick_fref_min_button.clicked.connect(
            lambda: self.enterPlotPickMode(target=self.fref_min_edit)
        )
        
        self.fref_max_label = QLabel('Reference frequency max (Hz):')
        self.fref_max_label.setFont(common_font)
        self.fref_max_edit  = QLineEdit()
        self.fref_max_edit.setFont(common_font)
        self.pick_fref_max_button = QPushButton('Pick')
        self.pick_fref_max_button.setFont(common_font)
        self.pick_fref_max_button.clicked.connect(
            lambda: self.enterPlotPickMode(target=self.fref_max_edit)
        )
        
        # Group the above SMS mode controls together
        SMS_pid_group = QGroupBox("SMS mode(Scan αₚ and ref-f range)")
        SMS_pid_group.setFont(common_font)
        SMS_layout = QVBoxLayout()
        # αₚ scan range
        hbox_alphap_min = QHBoxLayout()
        hbox_alphap_min.addWidget(self.alphap_min_label)
        hbox_alphap_min.addWidget(self.alphap_min_edit)
        SMS_layout.addLayout(hbox_alphap_min)       
        hbox_alphap_max = QHBoxLayout()
        hbox_alphap_max.addWidget(self.alphap_max_label)
        hbox_alphap_max.addWidget(self.alphap_max_edit)
        SMS_layout.addLayout(hbox_alphap_max)   
        hbox_alphap_step = QHBoxLayout()
        hbox_alphap_step.addWidget(self.alphap_step_label)
        hbox_alphap_step.addWidget(self.alphap_step_edit)
        SMS_layout.addLayout(hbox_alphap_step)  
        # reference frequency scan range
        SMS_layout.addWidget(self.fref_min_label)
        # make an HBox for those two
        hbox_ffref_min = QHBoxLayout()
        hbox_ffref_min.addWidget(self.fref_min_edit)
        hbox_ffref_min.addWidget(self.pick_fref_min_button)
        # then add that HBox into your existing vertical layout
        SMS_layout.addLayout(hbox_ffref_min)        
        # make an HBox for those two
        SMS_layout.addWidget(self.fref_max_label)
        hbox_ffref_max = QHBoxLayout()
        hbox_ffref_max.addWidget(self.fref_max_edit)
        hbox_ffref_max.addWidget(self.pick_fref_max_button)
        # then add that HBox into your existing vertical layout
        SMS_layout.addLayout(hbox_ffref_max)        
        
        
        
        # ——— Add 'Run Quick PID' button here ———
        self.SMS_pid_button = QPushButton('Run SMS')
        self.SMS_pid_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.SMS_pid_button.clicked.connect(self.SMS_pid_script)
        SMS_layout.addWidget(self.SMS_pid_button)

        # SMS 进度条
        self.sms_progress = QProgressBar()
        self.sms_progress.setFont(common_font)
        self.sms_progress.setVisible(False)
        SMS_layout.addWidget(self.sms_progress)

        SMS_pid_group.setLayout(SMS_layout)
        self.vbox.addWidget(SMS_pid_group)

        # ——— IMS mode (Bρ & circumference scan) ———
        self.brho_min_label = QLabel('Bρ min (Tm):')
        self.brho_min_edit  = QLineEdit()
        self.brho_min_label.setFont(common_font)
        self.brho_min_edit.setFont(common_font)

        self.brho_max_label = QLabel('Bρ max (Tm):')
        self.brho_max_edit  = QLineEdit()
        self.brho_max_label.setFont(common_font)
        self.brho_max_edit.setFont(common_font)

        self.brho_step_label = QLabel('Bρ step (Tm):')
        self.brho_step_edit  = QLineEdit()
        self.brho_step_label.setFont(common_font)
        self.brho_step_edit.setFont(common_font)

        self.circ_min_label = QLabel('Circumference min (m):')
        self.circ_min_edit  = QLineEdit()
        self.circ_min_label.setFont(common_font)
        self.circ_min_edit.setFont(common_font)

        self.circ_max_label = QLabel('Circumference max (m):')
        self.circ_max_edit  = QLineEdit()
        self.circ_max_label.setFont(common_font)
        self.circ_max_edit.setFont(common_font)

        self.circ_step_label = QLabel('Circumference step (m):')
        self.circ_step_edit  = QLineEdit()
        self.circ_step_label.setFont(common_font)
        self.circ_step_edit.setFont(common_font)

        IMS_pid_group = QGroupBox("IMS mode(Scan Bρ and ring circumference)")
        IMS_pid_group.setFont(common_font)
        IMS_layout = QVBoxLayout()

        hbox_brho_min = QHBoxLayout()
        hbox_brho_min.addWidget(self.brho_min_label)
        hbox_brho_min.addWidget(self.brho_min_edit)
        IMS_layout.addLayout(hbox_brho_min)
        hbox_brho_max = QHBoxLayout()
        hbox_brho_max.addWidget(self.brho_max_label)
        hbox_brho_max.addWidget(self.brho_max_edit)
        IMS_layout.addLayout(hbox_brho_max)
        hbox_brho_step = QHBoxLayout()
        hbox_brho_step.addWidget(self.brho_step_label)
        hbox_brho_step.addWidget(self.brho_step_edit)
        IMS_layout.addLayout(hbox_brho_step)

        hbox_circ_min = QHBoxLayout()
        hbox_circ_min.addWidget(self.circ_min_label)
        hbox_circ_min.addWidget(self.circ_min_edit)
        IMS_layout.addLayout(hbox_circ_min)
        hbox_circ_max = QHBoxLayout()
        hbox_circ_max.addWidget(self.circ_max_label)
        hbox_circ_max.addWidget(self.circ_max_edit)
        IMS_layout.addLayout(hbox_circ_max)
        hbox_circ_step = QHBoxLayout()
        hbox_circ_step.addWidget(self.circ_step_label)
        hbox_circ_step.addWidget(self.circ_step_edit)
        IMS_layout.addLayout(hbox_circ_step)

        self.IMS_pid_button = QPushButton('Run IMS')
        self.IMS_pid_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.IMS_pid_button.clicked.connect(self.IMS_pid_script)
        IMS_layout.addWidget(self.IMS_pid_button)

        # IMS 进度条
        self.ims_progress = QProgressBar()
        self.ims_progress.setFont(common_font)
        self.ims_progress.setVisible(False)
        IMS_layout.addWidget(self.ims_progress)

        IMS_pid_group.setLayout(IMS_layout)
        self.vbox.addWidget(IMS_pid_group)

        self.simulation_result_label = QLabel('Simulation result:')
        self.simulation_result_edit = QLineEdit()
        self.simulation_result_label.setFont(common_font)
        self.simulation_result_edit.setFont(common_font)
        self.simulation_result_button = QPushButton('Browse')
        self.simulation_result_button.setFont(common_font)
        self.simulation_result_button.clicked.connect(self.browse_simulation_result)
        hbox_simulation_result = QHBoxLayout()
        hbox_simulation_result.addWidget(self.simulation_result_label)
        hbox_simulation_result.addWidget(self.simulation_result_edit)
        hbox_simulation_result.addWidget(self.simulation_result_button)
        self.vbox.addLayout(hbox_simulation_result)

        self.matched_result_label = QLabel('matched result:')
        self.matched_result_edit = QLineEdit()
        self.matched_result_label.setFont(common_font)
        self.matched_result_edit.setFont(common_font)
        self.matched_result_button = QPushButton('Browse')
        self.matched_result_button.setFont(common_font)
        self.matched_result_button.clicked.connect(self.browse_matched_result)
        hbox_matched_result = QHBoxLayout()
        hbox_matched_result.addWidget(self.matched_result_label)
        hbox_matched_result.addWidget(self.matched_result_edit)
        hbox_matched_result.addWidget(self.matched_result_button)
        self.vbox.addLayout(hbox_matched_result)

        self.exit_button = QPushButton('Exit')
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.exit_button.clicked.connect(self.close_application)
        hbox_buttons = QHBoxLayout()
        hbox_buttons.addWidget(self.exit_button)
        self.vbox.addLayout(hbox_buttons)
        
    def close_application(self):
        sys.exit()

    def _open_afc_calculator(self):
        """Open the AFC&gtr Calculator dialog with saved paths and settings."""
        dialog = AFCCalculatorDialog(self)
        # Restore saved state
        dialog.time_edit.setText(self._afc_time_file)
        dialog.voltage_edit.setText(self._afc_voltage_file)
        dialog.xmin_edit.setText(self._afc_xmin)
        dialog.xmax_edit.setText(self._afc_xmax)
        dialog.ymin_edit.setText(self._afc_ymin)
        dialog.ymax_edit.setText(self._afc_ymax)
        dialog.logz_checkbox.setChecked(self._afc_logz)
        dialog.show_proj_checkbox.setChecked(self._afc_show_proj)
        dialog.proj_offset_edit.setText(self._afc_offset)
        dialog.proj_dt_edit.setText(self._afc_dt)
        dialog.split_ratio_edit.setText(self._afc_split)
        if self._afc_threshold_path:
            dialog.thresh_path_edit.setText(self._afc_threshold_path)
            dialog._load_threshold_profile(self._afc_threshold_path)
        dialog.peak_dist_edit.setText(self._afc_peak_dist)
        # Save paths immediately when user clicks "Load Data & Plot"
        dialog.paths_changed.connect(self._on_afc_paths_changed)
        dialog.exec_()
        # Read back all state on close
        self._afc_time_file = dialog.time_edit.text().strip()
        self._afc_voltage_file = dialog.voltage_edit.text().strip()
        self._afc_xmin = dialog.xmin_edit.text().strip()
        self._afc_xmax = dialog.xmax_edit.text().strip()
        self._afc_ymin = dialog.ymin_edit.text().strip()
        self._afc_ymax = dialog.ymax_edit.text().strip()
        self._afc_logz = dialog.logz_checkbox.isChecked()
        self._afc_show_proj = dialog.show_proj_checkbox.isChecked()
        self._afc_offset = dialog.proj_offset_edit.text().strip()
        self._afc_dt = dialog.proj_dt_edit.text().strip()
        self._afc_split = dialog.split_ratio_edit.text().strip()
        self._afc_threshold_path = dialog.thresh_path_edit.text().strip()
        self._afc_peak_dist = dialog.peak_dist_edit.text().strip()
        self.save_parameters()

    def _on_afc_paths_changed(self, time_file, voltage_file):
        """Slot for AFCCalculatorDialog.paths_changed — save paths immediately."""
        self._afc_time_file = time_file
        self._afc_voltage_file = voltage_file
        # Fetch all state from the dialog
        sender = self.sender()
        if sender is not None:
            self._afc_xmin = sender.xmin_edit.text().strip()
            self._afc_xmax = sender.xmax_edit.text().strip()
            self._afc_ymin = sender.ymin_edit.text().strip()
            self._afc_ymax = sender.ymax_edit.text().strip()
            self._afc_logz = sender.logz_checkbox.isChecked()
            self._afc_show_proj = sender.show_proj_checkbox.isChecked()
            self._afc_offset = sender.proj_offset_edit.text().strip()
            self._afc_dt = sender.proj_dt_edit.text().strip()
            self._afc_split = sender.split_ratio_edit.text().strip()
            self._afc_threshold_path = sender.thresh_path_edit.text().strip()
            self._afc_peak_dist = sender.peak_dist_edit.text().strip()
        self.save_parameters()

    def browse_datafile(self):
        options = QFileDialog.Options()
        datafile, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*);;NPZ Files (*.npz)", options= options)
        if datafile:
            self.datafile_edit.setText(datafile)

    def browse_lppfile(self):
        options = QFileDialog.Options()
        lppfile, _ = QFileDialog.getOpenFileName(self, "Select .lpp File", "", "All Files (*);;LPP Files (*.lpp)", options= options)
        if lppfile:
            self.filep_edit.setText(lppfile)
            
    def browse_simulation_result(self):
        options = QFileDialog.Options()
        datafile, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*)", options= options)
        if datafile:
            self.simulation_result_edit.setText(datafile)

    def browse_matched_result(self):
        options = QFileDialog.Options()
        datafile, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*)", options= options)
        if datafile:
            self.matched_result_edit.setText(datafile)

    def browse_lppfile(self):
        options = QFileDialog.Options()
        lppfile, _ = QFileDialog.getOpenFileName(self, "Select .lpp File", "", "All Files (*);;LPP Files (*.lpp)", options= options)
        if lppfile:
            self.filep_edit.setText(lppfile)
            
    def _read_ref_harmonic(self):
        """Read ref_harmonic_edit and return int, or None if empty/invalid."""
        try:
            text = self.ref_harmonic_edit.text().strip()
            return int(text) if text else None
        except ValueError:
            return None

    def _apply_baseline_once(self):
        """Apply baseline removal ONCE to the current experimental data.
        The baseline-removed data replaces the cache, so subsequent runs
        skip baseline removal automatically."""
        if self.saved_data is None:
            QMessageBox.information(self, "Info", "Please run the simulation first.")
            return

        data = self.saved_data
        baseline = data.apply_baseline_once()
        if baseline is None:
            QMessageBox.critical(self, "Error", "Baseline removal failed.")
            return

        # Uncheck the checkbox so subsequent runs don't re-apply
        self.remove_baseline_checkbox.setChecked(False)
        # Force reload to use cached baseline-removed data next time
        self.reload_data_checkbox.setChecked(False)

        # Refresh the visualization with the new experimental data
        self.visualization_widget.updateData(data)
        print("Baseline applied once. Checkbox unchecked — subsequent runs will use cached data.")
        QApplication.processEvents()

    def _on_toggle_threshold_click(self):
        """Handle Start Click Threshold button press."""
        enabled = self.visualization_widget.toggle_threshold_click_mode()
        self._update_thresh_toggle_button_style(enabled)

    def _on_threshold_click_mode_changed(self, enabled):
        """Slot for thresholdClickModeChanged signal from CreatePyGUI."""
        self._update_thresh_toggle_button_style(enabled)

    def _update_thresh_toggle_button_style(self, enabled):
        """Update the Start Click Threshold button appearance."""
        if enabled:
            self.thresh_toggle_button.setText('Click Threshold: ON')
            self.thresh_toggle_button.setStyleSheet('background-color: #90EE90;')
        else:
            self.thresh_toggle_button.setText('Start Click Threshold')
            self.thresh_toggle_button.setStyleSheet('')

    def _on_thresh_browse(self):
        """Browse button handler for threshold profile file."""
        current_data = self.visualization_widget.current_data
        default_path = getattr(current_data, 'threshold_profile_path', '') if current_data is not None else ''
        filename, _ = QFileDialog.getSaveFileName(self, 'Select height_thresh.csv', default_path, 'CSV Files (*.csv)')
        if filename:
            if not filename.lower().endswith('.csv'):
                filename = f"{filename}.csv"
            self.thresh_profile_edit.setText(filename)

    def _on_thresh_clear(self):
        """Clear the threshold profile data."""
        data = self.saved_data or self.visualization_widget.current_data
        if data is not None:
            data.threshold_profile_freqs = None
            data.threshold_profile_vals = None
            data.threshold_profile_path = None
        self.visualization_widget.clear_threshold_profile()
        self.thresh_profile_edit.clear()
        print("✅ Threshold profile cleared.")

    def _sync_thresh_profile_path(self, data):
        """Sync the threshold profile path display with the given data object."""
        if data is not None:
            path = getattr(data, 'threshold_profile_path', '') or ''
            # Block signals to avoid recursive call to _apply_threshold_path
            self.thresh_profile_edit.blockSignals(True)
            self.thresh_profile_edit.setText(path)
            self.thresh_profile_edit.blockSignals(False)


    def run_script(self):
        try:
            print("Running script...")
            datafile = self.datafile_edit.text()
            filep = self.filep_edit.text()
            remove_baseline = self.remove_baseline_checkbox.isChecked()
            psd_baseline_removed_l = float(self.psd_baseline_removed_l_edit.text())
            psd_baseline_removed_ratio = float(self.psd_baseline_removed_ratio_edit.text())
            alphap = float(self.alphap_edit.text())
            harmonics = self.harmonics_edit.text()
            refion = self.refion_edit.text()
            highlight_ions = self.highlight_ions_edit.text()
            circumference = float(self.circumference_edit.text())
            mode = self.mode_combo.currentText()
            sim_scalingfactor = float(self.sim_scalingfactor_edit.text())
            value = self.value_edit.text()
            nions = self.nions_edit.text()
            simulation_result= self.simulation_result_edit.text()
            matched_result= self.matched_result_edit.text()
            ref_harmonic = self._read_ref_harmonic()
            try:
                threshold = float(self.threshold_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching threshold")
            try:
                matching_freq_min = float(self.matching_freq_min_edit.text())
                matching_freq_max = float(self.matching_freq_max_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching_freq_min_edit")

            # 使用已载入的数据
            model = self._get_model(emit_signal=False)

            # --- Build baseline (load particles & moqs, run full simulation) ---
            baseline_args = argparse.Namespace(
                datafile=datafile,
                filep=filep,
                remove_baseline=remove_baseline,
                psd_baseline_removed_l=psd_baseline_removed_l,
                psd_baseline_removed_ratio=psd_baseline_removed_ratio,
                alphap=alphap,
                harmonics=harmonics,
                refion=refion,
                highlight_ions=highlight_ions,
                nions=nions,
                circumference=circumference,
                mode=mode,
                sim_scalingfactor=sim_scalingfactor,
                value=value,
                reload_data=True,
                peak_threshold_pct=self.peak_thresh_value if hasattr(self, 'peak_thresh_value') else 0.05,
                min_distance=float(self.min_distance_edit.text()),
                output_results=True,
                saved_data=self.saved_data,
                matching_freq_min=matching_freq_min,
                matching_freq_max=matching_freq_max,
                simulation_result=simulation_result,
                ref_harmonic=ref_harmonic,
                skip_peak_detection=True,
            )
            baseline = import_controller(**vars(baseline_args))
            if baseline is None:
                raise RuntimeError("Failed to build baseline for Run.")

            # 使用已检测的峰
            if hasattr(model, 'peak_freqs') and len(model.peak_freqs) > 0:
                baseline.peak_freqs = model.peak_freqs.copy()
                if hasattr(model, 'peak_heights') and len(model.peak_heights) > 0:
                    baseline.peak_heights = model.peak_heights.copy()
                if hasattr(model, 'peak_widths_freq') and len(model.peak_widths_freq) > 0:
                    baseline.peak_widths_freq = model.peak_widths_freq.copy()
                print(f"♻️ Run: 使用已检测的 {len(baseline.peak_freqs)} 个峰进行匹配")

            # 直接调用 compute_matches（与 IMS 相同的匹配方法）
            harmonics_list = [float(h) for h in harmonics.split()]
            chi2, match_count, highlights = baseline.compute_matches(
                match_threshold=threshold,
                match_frequency_min=matching_freq_min,
                match_frequency_max=matching_freq_max,
                verbose=True,
            )
            baseline.save_matched_result(matched_result)

            # 输出各谐波的匹配数
            if hasattr(baseline, 'matched_sim_items') and baseline.matched_sim_items:
                per_h = {}
                for item in baseline.matched_sim_items:
                    h = int(float(item[2]))
                    per_h[h] = per_h.get(h, 0) + 1
                parts = '  '.join(f'h{k}={v}' for k, v in sorted(per_h.items()))
                print(f"\n📊 匹配结果: χ² = {chi2:.4e}, 总匹配数 = {match_count}")
                print(f"   各谐波匹配数: {parts}")
                print(f"   匹配离子: {highlights}")
                self.setWindowTitle(f"RionID+  χ²={chi2:.2e}  N={match_count}  {highlights[:3] if highlights else ''}")

            self.overlay_sim_signal.emit(baseline)
            self._sync_thresh_profile_path(baseline)
            self.save_parameters()

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')
            log.error("Processing failed", exc_info=True)
            if hasattr(self, 'signalError'):
                self.signalError.emit(str(e))

    def _update_harmonic_calculation(self):
        """Auto-calculate fundamental frequency and per-harmonic frequencies
        when in 'Frequency' mode.  Updates the two read‑only display fields."""
        mode = self.mode_combo.currentText()
        if mode != 'Frequency':
            self.f0_display.clear()
            self.harmonic_freqs_display.clear()
            return

        # 1) reference frequency
        try:
            f_ref = float(self.value_edit.text())
        except (ValueError, AttributeError):
            self.f0_display.clear()
            self.harmonic_freqs_display.clear()
            return

        # 2) reference harmonic number
        try:
            h_ref = int(self.ref_harmonic_edit.text())
        except (ValueError, AttributeError):
            self.f0_display.clear()
            self.harmonic_freqs_display.clear()
            return

        if h_ref <= 0:
            return

        # --- compute ---
        f0 = f_ref / h_ref
        self.f0_display.setText(f"{f0:.2f}")

        # per‑harmonic frequencies
        harm_text = self.harmonics_edit.text().strip()
        if not harm_text:
            self.harmonic_freqs_display.clear()
            return

        try:
            harmonics = [int(h) for h in harm_text.split()]
        except ValueError:
            self.harmonic_freqs_display.setText("Invalid harmonics format")
            self.harmonic_freqs_display.setStyleSheet("color: red;")
            return

        # Reset style in case it was previously red
        self.harmonic_freqs_display.setStyleSheet("color: black;")

        parts = [f"h{h}={f0 * h:.2f}" for h in harmonics]
        self.harmonic_freqs_display.setText(", ".join(parts))

    def mousePressEvent(self, event):
        """
        Any mouse click on this widget will set the stop flag,
        causing the SMS_pid_script / IMS_pid_script loops to exit.
        """
        self._stop_SMS_pid = True
        self._stop_IMS_pid = True
        super().mousePressEvent(event)

    def load_data(self):
        """载入实验数据按钮的处理方法。"""
        datafile = self.datafile_edit.text().strip()
        if not datafile:
            QMessageBox.warning(self, "警告", "请先选择实验数据文件 (Experimental Data File)")
            return

        # ── 峰汇总文件 (.txt)：弹出柱状图配置对话框 ──
        hist_freq_min = None
        hist_freq_max = None
        hist_bins = None
        if datafile.lower().endswith('.txt'):
            # 检查是否已有保存的参数（从 parameters_cache.toml 载入）
            if (getattr(self, '_hist_freq_min', None) is not None
                    and getattr(self, '_hist_freq_max', None) is not None
                    and getattr(self, '_hist_bins', None) is not None):
                hist_freq_min = self._hist_freq_min
                hist_freq_max = self._hist_freq_max
                hist_bins = self._hist_bins
                print(f"♻️ 复用已保存的柱状图参数: {hist_freq_min/1e6:.4f}–{hist_freq_max/1e6:.4f} MHz, {hist_bins} bins")
            else:
                raw_list = []
                with open(datafile, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('总序号') or line.startswith('---'):
                            continue
                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        try:
                            raw_list.append(float(parts[2]) * 1e6)
                        except (ValueError, IndexError):
                            continue
                if raw_list:
                    raw_freqs = np.array(raw_list)
                    dialog = HistogramConfigDialog(raw_freqs, self)
                    if dialog.exec_() == QDialog.Accepted:
                        hist_freq_min, hist_freq_max, hist_bins = dialog.get_params()
                        self._hist_freq_min = hist_freq_min
                        self._hist_freq_max = hist_freq_max
                        self._hist_bins = hist_bins
                    else:
                        print("⚠️ 用户取消了柱状图配置")
                        return

        # 构造 ImportData（始终重新读取文件）
        refion = self.refion_edit.text()
        highlight_ions = self.highlight_ions_edit.text()
        remove_baseline = self.remove_baseline_checkbox.isChecked()
        psd_baseline_removed_l = float(self.psd_baseline_removed_l_edit.text())
        psd_baseline_removed_ratio = float(self.psd_baseline_removed_ratio_edit.text())
        alphap = float(self.alphap_edit.text())
        peak_threshold_pct = self.peak_thresh_value if hasattr(self, 'peak_thresh_value') else 0.05
        min_distance = float(self.min_distance_edit.text())
        circumference = float(self.circumference_edit.text())
        matching_freq_min = float(self.matching_freq_min_edit.text())
        matching_freq_max = float(self.matching_freq_max_edit.text())

        model = ImportData(
            refion=refion, highlight_ions=highlight_ions,
            remove_baseline=remove_baseline or None,
            psd_baseline_removed_l=psd_baseline_removed_l or None,
            psd_baseline_removed_ratio=psd_baseline_removed_ratio or None,
            alphap=alphap, filename=datafile, reload_data=True,
            circumference=circumference,
            peak_threshold_pct=peak_threshold_pct,
            min_distance=min_distance,
            matching_freq_min=matching_freq_min,
            matching_freq_max=matching_freq_max,
            hist_freq_min=hist_freq_min,
            hist_freq_max=hist_freq_max,
            hist_bins=hist_bins,
            skip_peak_detection=True
        )
        if model.experimental_data is None or (hasattr(model.experimental_data, '__len__') and len(model.experimental_data) != 2):
            raise RuntimeError("无法载入数据，请检查数据文件。")
        self.saved_data = model
        self.visualization_signal.emit(model)
        self._sync_thresh_profile_path(model)
        self.save_parameters()
        print(f"✅ 数据载入完成，共 {len(model.peak_freqs)} 个峰")

    def _find_peaks(self):
        """在已载入的数据上重新执行 peak detection 并更新绘图。"""
        model = self._get_model(emit_signal=False)
        try:
            model.detect_peaks_and_widths()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Peak detection failed: {str(e)}")
            return
        print(f"✅ 寻峰完成: {len(model.peak_freqs)} 个峰")
        self.visualization_signal.emit(model)

    def _get_model(self, emit_signal=True):
        """返回 self.saved_data，如果未载入数据则报错。"""
        if (self.saved_data is None
                or not hasattr(self.saved_data, 'experimental_data')
                or self.saved_data.experimental_data is None):
            raise RuntimeError("尚未载入实验数据，请先点击「📂 载入数据」按钮。")
        if emit_signal:
            self.visualization_signal.emit(self.saved_data)
        return self.saved_data

    def SMS_pid_script(self):
        try:
            print("Running SMS_pid_script…")
            datafile = self.datafile_edit.text().strip()
            if not datafile:
                raise ValueError("No experimental data provided.")

            # --- collect constant arguments once ---
            filep = self.filep_edit.text() or None
            remove_baseline = self.remove_baseline_checkbox.isChecked()
            psd_baseline_removed_l = float(self.psd_baseline_removed_l_edit.text())
            psd_baseline_removed_ratio = float(self.psd_baseline_removed_ratio_edit.text())
            alphap = float(self.alphap_edit.text())
            peak_threshold_pct = self.peak_thresh_value if hasattr(self, 'peak_thresh_value') else 0.05
            min_distance = float(self.min_distance_edit.text())
            harmonics = self.harmonics_edit.text()
            refion = self.refion_edit.text()
            highlight_ions = self.highlight_ions_edit.text() or None
            nions = self.nions_edit.text() or None
            circumference = float(self.circumference_edit.text())
            sim_scalingfactor = self.sim_scalingfactor_edit.text().strip()
            sim_scalingfactor = float(sim_scalingfactor) if sim_scalingfactor else None
            simulation_result= self.simulation_result_edit.text()
            matched_result= self.matched_result_edit.text()
            ref_harmonic = self._read_ref_harmonic()

            try:
                threshold = float(self.threshold_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching threshold")
            try:
                matching_freq_min = float(self.matching_freq_min_edit.text())
                matching_freq_max = float(self.matching_freq_max_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching_freq_min_edit")

            fref_min = float(self.fref_min_edit.text() or '-inf')
            fref_max = float(self.fref_max_edit.text() or 'inf')

            # --- 1) 使用已载入的数据 ---
            model = self._get_model(emit_signal=False)
            # experimental peak frequencies (Hz)
            exp_peaks_hz = model.peak_freqs

            print(f"Detected {len(exp_peaks_hz)} experimental peaks.")

            # define your alphap scan range
            alphap_min  = float(self.alphap_min_edit.text())
            alphap_max  = float(self.alphap_max_edit.text())
            alphap_step = float(self.alphap_step_edit.text())
            
            # … your preamble: gather datafile, alphap, exp_peaks_hz, etc. …
            self._stop_SMS_pid = False
            QApplication.processEvents()  # allow pending events (like mousePressEvent) to fire

            results = []
 
            # 过滤实验峰：
            exp_peaks_hz = [f for f in model.peak_freqs]
            exp_peaks_hz_filtering = [f for f in model.peak_freqs if fref_min <= f <= fref_max]
            # Check if the list is empty after filtering
                
            if not exp_peaks_hz:
                # If no peaks are found within the specified range, show an error message
                QMessageBox.critical(self, "Error", "No experimental peaks found within the specified frequency range.\nPlease adjust the frequency range.")
            
                # Change background color of input fields to red
                self.fref_min_edit.setStyleSheet("background-color: red;")
                self.fref_max_edit.setStyleSheet("background-color: red;")
            else:
                # If peaks are found, reset the background color to normal
                self.fref_min_edit.setStyleSheet("")
                self.fref_max_edit.setStyleSheet("")
                   
            # Grab and remember the original styles so we can restore them later
            orig_value_style  = self.value_edit.styleSheet()
            orig_alpha_style  = self.alphap_edit.styleSheet()

            #exp_peaks_hz_filtering.append(model.ref_frequency)

            # --- Convert harmonics string to list once ---
            harmonics_list = [float(h) for h in harmonics.split()]

            # --- Build a baseline ImportData (one time, loads particles + builds yield_data) ---
            # Use the first filtered peak as the initial f_ref; actual scanning uses scan_match.
            if not exp_peaks_hz_filtering:
                QMessageBox.critical(self, "Error",
                    "No experimental peaks found within the specified frequency range.\n"
                    "Please adjust the frequency range.")
                self.fref_min_edit.setStyleSheet("background-color: red;")
                self.fref_max_edit.setStyleSheet("background-color: red;")
                return

            baseline_args = argparse.Namespace(
                datafile=datafile,
                filep=filep,
                remove_baseline=remove_baseline,
                psd_baseline_removed_l=psd_baseline_removed_l,
                psd_baseline_removed_ratio=psd_baseline_removed_ratio,
                alphap=alphap,
                harmonics=harmonics,
                refion=refion,
                highlight_ions=highlight_ions,
                nions=nions,
                circumference=circumference,
                mode='Frequency',
                sim_scalingfactor=sim_scalingfactor,
                value=exp_peaks_hz_filtering[0],
                reload_data=True,
                peak_threshold_pct=peak_threshold_pct,
                min_distance=min_distance,
                output_results=False,
                saved_data=None,
                matching_freq_min=matching_freq_min,
                matching_freq_max=matching_freq_max,
                simulation_result=simulation_result,
                ref_harmonic=ref_harmonic
            )
            baseline = import_controller(**vars(baseline_args))
            if baseline is None:
                raise RuntimeError("Failed to build baseline ImportData for scanning.")
            # 恢复已检测的峰到 baseline 上，供 scan_match 使用
            detected_peak_freqs = model.peak_freqs.copy() if hasattr(model, 'peak_freqs') and len(model.peak_freqs) > 0 else None
            detected_peak_heights = model.peak_heights.copy() if hasattr(model, 'peak_heights') and len(model.peak_heights) > 0 else None
            detected_peak_widths = model.peak_widths_freq.copy() if hasattr(model, 'peak_widths_freq') and len(model.peak_widths_freq) > 0 else None
            if detected_peak_freqs is not None and len(detected_peak_freqs) < len(baseline.peak_freqs):
                baseline.peak_freqs = detected_peak_freqs
                if detected_peak_heights is not None:
                    baseline.peak_heights = detected_peak_heights
                if detected_peak_widths is not None:
                    baseline.peak_widths_freq = detected_peak_widths
                print(f"♻️ SMS: 使用已检测的 {len(baseline.peak_freqs)} 个峰进行扫描")
            self.saved_data = baseline
            QApplication.processEvents()

            # Grab and remember original styles
            orig_value_style = self.value_edit.styleSheet()
            orig_alpha_style = self.alphap_edit.styleSheet()
            results = []

            # 进度条设置
            n_fref = len(exp_peaks_hz_filtering)
            n_alphap = len(np.arange(alphap_min, alphap_max + 1e-12, alphap_step))
            sms_total = n_fref * n_alphap
            sms_step = 0
            self.sms_progress.setVisible(True)
            self.sms_progress.setValue(0)

            for f_ref in exp_peaks_hz_filtering:
                QApplication.processEvents()
                if self._stop_SMS_pid:
                    print("Quick‐PID scan was stopped by user click.")
                    break

                # Highlight current f_ref in the UI
                self.value_edit.setStyleSheet("background-color: #fff8b0;")
                self.value_edit.setText(f"{f_ref:.2f}")
                QApplication.processEvents()

                # Inner loop over a range of test_alphap values
                for test_alphap in np.arange(alphap_min, alphap_max + 1e-12, alphap_step):
                    if self._stop_SMS_pid:
                        print("Quick‐PID scan was stopped by user click.")
                        break

                    # Update UI to show which alphap is being tested
                    self.alphap_edit.setStyleSheet("background-color: #b0fff8;")
                    self.alphap_edit.setText(f"{test_alphap:.6f}")
                    QApplication.processEvents()

                    # --- Lightweight scan_match (no ImportData creation, no particle loop) ---
                    chi2, match_count, highlights = baseline.scan_match(
                        f_ref=f_ref,
                        alphap=test_alphap,
                        harmonics=harmonics_list,
                        match_threshold=threshold,
                        match_frequency_min=matching_freq_min,
                        match_frequency_max=matching_freq_max,
                        mode='Frequency',
                        ref_harmonic=ref_harmonic,
                    )
                    results.append((f_ref, test_alphap, chi2, match_count, highlights))

                    # 更新进度条
                    sms_step += 1
                    if sms_total > 0:
                        self.sms_progress.setValue(int(sms_step / sms_total * 100))
                    QApplication.processEvents()

                # After inner loop: find best for this f_ref
                sorted_results = sorted(results, key=lambda x: (-x[3], x[2]))
                best_fref, best_alphap, best_chi2, best_match_count, best_match_ions = sorted_results[0]
                print(f"  f_ref={best_fref:.2f}Hz → 当前最佳: αp={best_alphap:.6f}, χ²={best_chi2:.3e}, matches={best_match_count}")
                self.alphap_edit.setStyleSheet(orig_alpha_style)

            # after outer loop: plot best result once
            self.sms_progress.setVisible(False)
            self.value_edit.setStyleSheet(orig_value_style)

            sorted_results = sorted(results, key=lambda x: (-x[3], x[2]))
            best_fref, best_alphap, best_chi2, best_match_count, best_match_ions = sorted_results[0]
            print(f"\n→ 全局最佳: f_ref={best_fref:.2f}Hz, alphap={best_alphap:.6f}, χ²={best_chi2:.3e}, matches={best_match_count} {best_match_ions}")

            # Run full simulation for the winning pair (for final visualization & output)
            sim_args = argparse.Namespace(
                datafile=datafile,
                filep=filep,
                remove_baseline=remove_baseline,
                psd_baseline_removed_l=psd_baseline_removed_l,
                psd_baseline_removed_ratio=psd_baseline_removed_ratio,
                alphap=best_alphap,
                harmonics=harmonics,
                refion=refion,
                highlight_ions=highlight_ions,
                nions=nions,
                circumference=circumference,
                mode='Frequency',
                sim_scalingfactor=sim_scalingfactor,
                value=best_fref,
                reload_data=False,
                peak_threshold_pct=peak_threshold_pct,
                min_distance=min_distance,
                output_results=True,
                saved_data=self.saved_data,
                matching_freq_min=matching_freq_min,
                matching_freq_max=matching_freq_max,
                simulation_result=simulation_result,
                ref_harmonic=ref_harmonic
            )
            best_data = import_controller(**vars(sim_args))
            best_chi2, best_match_count, best_match_ions = best_data.compute_matches(threshold, matching_freq_min, matching_freq_max)
            best_data.save_matched_result(matched_result)

            self.mode_combo.setCurrentText('Frequency')
            self.value_edit.setText(f"{best_fref:.2f}")
            self.alphap_edit.setText(f"{best_alphap:.6f}")
            self.overlay_sim_signal.emit(best_data)
            self._sync_thresh_profile_path(best_data)
            QApplication.processEvents()
            self.save_parameters()

        except Exception as e:
            # On any error, also ensure any highlight is reset if needed
            QMessageBox.critical(self, "SMS Error", str(e))
            log.error("SMS_pid_script failed", exc_info=True)

    def IMS_pid_script(self):
        try:
            print("Running IMS_pid_script...")
            datafile = self.datafile_edit.text().strip()
            if not datafile:
                raise ValueError("No experimental data provided.")

            # --- collect constant arguments once ---
            filep = self.filep_edit.text() or None
            remove_baseline = self.remove_baseline_checkbox.isChecked()
            psd_baseline_removed_l = float(self.psd_baseline_removed_l_edit.text())
            psd_baseline_removed_ratio = float(self.psd_baseline_removed_ratio_edit.text())
            alphap = float(self.alphap_edit.text())
            peak_threshold_pct = self.peak_thresh_value if hasattr(self, "peak_thresh_value") else 0.05
            min_distance = float(self.min_distance_edit.text())
            harmonics = self.harmonics_edit.text()
            refion = self.refion_edit.text()
            highlight_ions = self.highlight_ions_edit.text() or None
            nions = self.nions_edit.text() or None
            circumference = float(self.circumference_edit.text())
            sim_scalingfactor = self.sim_scalingfactor_edit.text().strip()
            sim_scalingfactor = float(sim_scalingfactor) if sim_scalingfactor else None
            simulation_result = self.simulation_result_edit.text()
            matched_result = self.matched_result_edit.text()

            try:
                threshold = float(self.threshold_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching threshold")
            try:
                matching_freq_min = float(self.matching_freq_min_edit.text())
                matching_freq_max = float(self.matching_freq_max_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching_freq_min_edit")

            # --- Scan ranges ---
            brho_min     = float(self.brho_min_edit.text() or "1")
            brho_max     = float(self.brho_max_edit.text() or "10")
            brho_step    = float(self.brho_step_edit.text() or "0.1")
            circ_min     = float(self.circ_min_edit.text() or "100")
            circ_max     = float(self.circ_max_edit.text() or "120")
            circ_step    = float(self.circ_step_edit.text() or "1")

            # If baseline was already applied once, skip re-baseline
            if self.saved_data is not None and getattr(self.saved_data, '_baseline_applied', False):
                remove_baseline = False

            # --- 1) 使用已载入的数据 ---
            model = self._get_model(emit_signal=False)
            # 记住已检测的峰（Find Peaks 后的结果），后面 baseline 会覆盖 peak_freqs
            detected_peak_freqs = model.peak_freqs.copy() if hasattr(model, 'peak_freqs') else None
            detected_peak_heights = model.peak_heights.copy() if hasattr(model, 'peak_heights') and len(model.peak_heights) > 0 else None
            detected_peak_widths = model.peak_widths_freq.copy() if hasattr(model, 'peak_widths_freq') and len(model.peak_widths_freq) > 0 else None
            print(f"Detected {len(model.peak_freqs)} experimental peaks.")

            # --- Build baseline (load particles & moqs) ---
            baseline_args = argparse.Namespace(
                datafile=datafile,
                filep=filep,
                remove_baseline=remove_baseline,
                psd_baseline_removed_l=psd_baseline_removed_l,
                psd_baseline_removed_ratio=psd_baseline_removed_ratio,
                alphap=alphap,
                harmonics=harmonics,
                refion=refion,
                highlight_ions=highlight_ions,
                nions=nions,
                circumference=circumference,
                mode="Bρ",
                sim_scalingfactor=sim_scalingfactor,
                value="5.0",
                reload_data=True,
                peak_threshold_pct=peak_threshold_pct,
                min_distance=min_distance,
                output_results=False,
                saved_data=self.saved_data,
                matching_freq_min=matching_freq_min,
                matching_freq_max=matching_freq_max,
                simulation_result=simulation_result
            )
            baseline = import_controller(**vars(baseline_args))
            if baseline is None:
                raise RuntimeError("Failed to build baseline ImportData for scanning.")
            # 恢复已检测的峰到 baseline 上，供 scan_match_brho 使用
            if detected_peak_freqs is not None:
                baseline.peak_freqs = detected_peak_freqs
                if detected_peak_heights is not None:
                    baseline.peak_heights = detected_peak_heights
                if detected_peak_widths is not None:
                    baseline.peak_widths_freq = detected_peak_widths
                print(f"♻️ IMS: 使用已检测的 {len(baseline.peak_freqs)} 个峰进行扫描")
            self.saved_data = baseline
            QApplication.processEvents()

            harmonics_list = [float(h) for h in harmonics.split()]
            results = []
            self._stop_IMS_pid = False
            orig_value_style = self.value_edit.styleSheet()
            # Switch mode to Bρ so value_edit shows the Bρ value during scan
            self.mode_combo.setCurrentText("Bρ")
            QApplication.processEvents()

            # 计算总步数配置进度条
            brho_vals = np.arange(brho_min, brho_max + 1e-12, brho_step)
            circ_vals = np.arange(circ_min, circ_max + 1e-12, circ_step)
            total_steps = len(brho_vals) * len(circ_vals)
            step = 0
            self.ims_progress.setVisible(True)
            self.ims_progress.setValue(0)

            for test_brho in brho_vals:
                QApplication.processEvents()
                if self._stop_IMS_pid:
                    print("IMS scan was stopped by user click.")
                    break

                # Update UI to show current brho
                self.value_edit.setStyleSheet("background-color: #b0fff8;")
                self.value_edit.setText(f"{test_brho:.6f}")
                QApplication.processEvents()

                for test_circ in circ_vals:
                    if self._stop_IMS_pid:
                        print("IMS scan was stopped by user click.")
                        break

                    chi2, match_count, highlights = baseline.scan_match_brho(
                        brho=test_brho,
                        circumference=test_circ,
                        harmonics=harmonics_list,
                        match_threshold=threshold,
                        match_frequency_min=matching_freq_min,
                        match_frequency_max=matching_freq_max,
                        verbose=False,
                    )
                    results.append((test_brho, test_circ, chi2, match_count, highlights))

                    # 更新进度条
                    step += 1
                    if total_steps > 0:
                        self.ims_progress.setValue(int(step / total_steps * 100))
                    QApplication.processEvents()

            self.ims_progress.setVisible(False)

            # --- Find best result ---
            if not results:
                QMessageBox.critical(self, "IMS Error", "No results computed.")
                return

            sorted_results = sorted(results, key=lambda x: (-x[3], x[2]))
            best_brho, best_circ, best_chi2, best_match_count, best_match_ions = sorted_results[0]

            # --- Run full simulation for the winning pair ---
            sim_args = argparse.Namespace(
                datafile=datafile,
                filep=filep,
                remove_baseline=remove_baseline,
                psd_baseline_removed_l=psd_baseline_removed_l,
                psd_baseline_removed_ratio=psd_baseline_removed_ratio,
                alphap=alphap,
                harmonics=harmonics,
                refion=refion,
                highlight_ions=highlight_ions,
                nions=nions,
                circumference=best_circ,
                mode="Bρ",
                sim_scalingfactor=sim_scalingfactor,
                value=str(best_brho),
                reload_data=False,
                peak_threshold_pct=peak_threshold_pct,
                min_distance=min_distance,
                output_results=True,
                saved_data=self.saved_data,
                matching_freq_min=matching_freq_min,
                matching_freq_max=matching_freq_max,
                simulation_result=simulation_result
            )
            best_data = import_controller(**vars(sim_args))
            best_chi2, best_match_count, best_match_ions = best_data.compute_matches(
                threshold, matching_freq_min, matching_freq_max
            )
            best_data.save_matched_result(matched_result)
            self.save_parameters()
            print(f"\n-> Best: Brho={best_brho:.4f}Tm, circ={best_circ:.4f}m, "
                  f"chi2={best_chi2:.3e}, matches={best_match_count} {best_match_ions}")

            self.value_edit.setStyleSheet(orig_value_style)
            self.mode_combo.setCurrentText("Bρ")
            self.value_edit.setText(f"{best_brho:.4f}")
            self.circumference_edit.setText(f"{best_circ:.4f}")
            self.overlay_sim_signal.emit(best_data)
            self._sync_thresh_profile_path(best_data)
            QApplication.processEvents()

        except Exception as e:
            QMessageBox.critical(self, "IMS Error", str(e))
            log.error("IMS_pid_script failed", exc_info=True)


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super(CollapsibleGroupBox, self).__init__(parent)
        self.setTitle("")
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_widget.setLayout(self.content_layout)
        self.content_widget.setVisible(False)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

    def on_pressed(self):
        if self.toggle_button.isChecked():
            self.toggle_button.setArrowType(Qt.DownArrow)
            self.content_widget.setVisible(True)
        else:
            self.toggle_button.setArrowType(Qt.RightArrow)
            self.content_widget.setVisible(False)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)
