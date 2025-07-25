from PyQt5.QtWidgets import QApplication,QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QFileDialog, QMessageBox, QGroupBox, QToolButton
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QFont
import toml
import argparse
import logging as log
from loguru import logger
from rionidgui.gui_controller import import_controller
from rionid.importdata import ImportData
import sys
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel, QLineEdit, QHBoxLayout, QScrollArea
import time

log.basicConfig(level=log.DEBUG)
common_font = QFont()
common_font.setPointSize(12) #font size

class RionID_GUI(QWidget):
    visualization_signal = pyqtSignal(object)
    overlay_sim_signal    = pyqtSignal(object)           # new—just overlays one simulation
    clear_sim_signal      = pyqtSignal()           # ← new
    def __init__(self, plot_widget, *args, **kwargs):
        super().__init__()
        self.visualization_widget = plot_widget
        self._stop_quick_pid = False
        self.initUI()
        self.load_parameters()  # Load parameters after initializing UI
        
    @pyqtSlot()
    def onPlotClicked(self):
        """Called when user clicks inside the plot area."""
        self._stop_quick_pid = True        
    
    def initUI(self):
        self.setup_layout()

    def load_parameters(self, filepath='parameters_cache.toml'):
        try:
            with open(filepath, 'r') as f:
                parameters = toml.load(f)
                self.datafile_edit.setText(parameters.get('datafile', ''))
                self.filep_edit.setText(parameters.get('filep', ''))
                self.remove_baseline_checkbox.setChecked(parameters.get('remove_baseline', True))
                self.psd_baseline_removed_l_edit.setText(parameters.get('psd_baseline_removed_l', ''))
                self.alphap_edit.setText(parameters.get('alphap', ''))
                self.alphap_min_edit.setText(parameters.get('alphap_min', ''))
                self.alphap_max_edit.setText(parameters.get('alphap_max', ''))
                self.alphap_step_edit.setText(parameters.get('alphap_step', ''))
                self.threshold_edit.setText(str(parameters.get('threshold', '')))
                self.matching_freq_min_edit.setText(str(parameters.get('matching_freq_min', '')))
                self.matching_freq_max_edit.setText(str(parameters.get('matching_freq_max', '')))
                self.fref_min_edit.setText(parameters.get('fref_min', ''))
                self.fref_max_edit.setText(parameters.get('fref_max', ''))
                self.peak_thresh_edit.setText(str(parameters.get('peak_threshold_pct', '')))
                self.min_distance_edit.setText(str(parameters.get('min_distance', '')))
                self.harmonics_edit.setText(parameters.get('harmonics', ''))
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
                self.saved_data=None
                
        except FileNotFoundError:
            pass  # No parameters file exists yet

    def save_parameters(self, filepath='parameters_cache.toml'):
        parameters = {
            'datafile': self.datafile_edit.text(),
            'filep': self.filep_edit.text(),
            'remove_baseline_checkbox': self.remove_baseline_checkbox.isChecked(),
            'psd_baseline_removed_l': self.psd_baseline_removed_l_edit.text(),
            'alphap': self.alphap_edit.text(),
            'alphap_min': self.alphap_min_edit.text(),
            'alphap_max': self.alphap_max_edit.text(),
            'alphap_step': self.alphap_step_edit.text(),
            'threshold': self.threshold_edit.text(),
            'matching_freq_min': self.matching_freq_min_edit.text(),
            'matching_freq_max': self.matching_freq_max_edit.text(),
            'peak_threshold_pct': float(self.peak_thresh_edit.text()),
            'min_distance': float(self.min_distance_edit.text()),
            'fref_min': self.fref_min_edit.text(),
            'fref_max': self.fref_max_edit.text(),
            'harmonics': self.harmonics_edit.text(),
            'refion': self.refion_edit.text(),
            'highlight_ions': self.highlight_ions_edit.text(),
            'circumference': self.circumference_edit.text(),
            'mode': self.mode_combo.currentText(),
            'value': self.value_edit.text(),
            'sim_scalingfactor': self.sim_scalingfactor_edit.text(),
            'reload_data': self.reload_data_checkbox.isChecked(),
            'nions': self.nions_edit.text(),
            'simulation_result': self.simulation_result_edit.text(),
            'matched_result': self.matched_result_edit.text()
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
    
        # αₚ main input
        self.alphap_label = QLabel('<i>α<sub>p</sub> or γ<sub>t</sub> :</i>')
        self.alphap_edit = QLineEdit()
        self.alphap_label.setFont(common_font)
        self.alphap_edit.setFont(common_font)
    
        # The circumference of the storage ring
        self.circumference_label = QLabel('Circumference (m):')
        self.circumference_edit = QLineEdit()
        self.circumference_label.setFont(common_font)
        self.circumference_edit.setFont(common_font)
    
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
        # Scaling factor
        self.sim_scalingfactor_label = QLabel('Scaling factor:')
        self.sim_scalingfactor_edit = QLineEdit()
        self.sim_scalingfactor_label.setFont(common_font)
        self.sim_scalingfactor_edit.setFont(common_font)
        # value
        self.value_label = QLabel('Value:')
        self.value_edit = QLineEdit()
            
        # psd_baseline_removed_l parameters input
        self.vbox.addWidget(self.reload_data_checkbox)
        self.vbox.addWidget(self.remove_baseline_checkbox)
        hbox_psd_baseline_removed_l = QHBoxLayout()
        hbox_psd_baseline_removed_l.addWidget(self.psd_baseline_removed_l_label)
        hbox_psd_baseline_removed_l.addWidget(self.psd_baseline_removed_l_edit)
        self.vbox.addLayout(hbox_psd_baseline_removed_l)

        hbox_alphap = QHBoxLayout()
        hbox_alphap.addWidget(self.alphap_label)
        hbox_alphap.addWidget(self.alphap_edit)
        self.vbox.addLayout(hbox_alphap)

        # circumference
        hbox_circ = QHBoxLayout()
        hbox_circ.addWidget(self.circumference_label)
        hbox_circ.addWidget(self.circumference_edit)
        self.vbox.addLayout(hbox_circ)
    
        # other parameters
        for label, widget in (
            (self.harmonics_label, self.harmonics_edit),
            (self.refion_label, self.refion_edit)
        ):
            hb = QHBoxLayout()
            hb.addWidget(label)
            hb.addWidget(widget)
            self.vbox.addLayout(hb)
            
        self.vbox.addWidget(self.highlight_ions_label)
        self.vbox.addWidget(self.highlight_ions_edit)
        
        hbox_mode = QHBoxLayout()
        hbox_mode.addWidget(self.mode_label)
        hbox_mode.addWidget(self.mode_combo)
        hbox_mode.addWidget(self.value_edit)
        self.vbox.addLayout(hbox_mode)
    
        # Next, pack scaling‐factor and Run button together
        hbox_sf = QHBoxLayout()
        hbox_sf.addWidget(self.sim_scalingfactor_label)
        hbox_sf.addWidget(self.sim_scalingfactor_edit)
        # ——— Add the Run button *before* the Optional Features section ———
        self.vbox.addLayout(hbox_sf)
        # Peak threshold (% of max)
        self.peak_thresh_label = QLabel('Peak threshold (of max):')
        self.peak_thresh_label.setFont(common_font)
        self.peak_thresh_edit  = QLineEdit()
        self.peak_thresh_edit.setFont(common_font)
        # Peak threshold (% of max)
        hbox_peak = QHBoxLayout()
        hbox_peak.addWidget(self.peak_thresh_label)
        hbox_peak.addWidget(self.peak_thresh_edit)
        self.vbox.addLayout(hbox_peak)
        
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



        
        # ——— Quick PID 设置 ———
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
        
        # Group the above Quick PID controls together
        quick_pid_group = QGroupBox("Quick PID Settings")
        quick_pid_group.setFont(common_font)
        qp_layout = QVBoxLayout()
        # αₚ scan range
        hbox_alphap_min = QHBoxLayout()
        hbox_alphap_min.addWidget(self.alphap_min_label)
        hbox_alphap_min.addWidget(self.alphap_min_edit)
        qp_layout.addLayout(hbox_alphap_min)       
        hbox_alphap_max = QHBoxLayout()
        hbox_alphap_max.addWidget(self.alphap_max_label)
        hbox_alphap_max.addWidget(self.alphap_max_edit)
        qp_layout.addLayout(hbox_alphap_max)   
        hbox_alphap_step = QHBoxLayout()
        hbox_alphap_step.addWidget(self.alphap_step_label)
        hbox_alphap_step.addWidget(self.alphap_step_edit)
        qp_layout.addLayout(hbox_alphap_step)  
        # reference frequency scan range
        qp_layout.addWidget(self.fref_min_label)
        # make an HBox for those two
        hbox_ffref_min = QHBoxLayout()
        hbox_ffref_min.addWidget(self.fref_min_edit)
        hbox_ffref_min.addWidget(self.pick_fref_min_button)
        # then add that HBox into your existing vertical layout
        qp_layout.addLayout(hbox_ffref_min)        
        # make an HBox for those two
        qp_layout.addWidget(self.fref_max_label)
        hbox_ffref_max = QHBoxLayout()
        hbox_ffref_max.addWidget(self.fref_max_edit)
        hbox_ffref_max.addWidget(self.pick_fref_max_button)
        # then add that HBox into your existing vertical layout
        qp_layout.addLayout(hbox_ffref_max)        
        
        
        
        # ——— Add 'Run Quick PID' button here ———
        self.quick_pid_button = QPushButton('Run Quick PID')
        self.quick_pid_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.quick_pid_button.clicked.connect(self.quick_pid_script)
        qp_layout.addWidget(self.quick_pid_button)
        quick_pid_group.setLayout(qp_layout)
        self.vbox.addWidget(quick_pid_group)

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
            
    def run_script(self):
        try:
            print("Running script...")
            datafile = self.datafile_edit.text()
            #if not datafile:
            #    raise ValueError("No experimental data provided. Please enter any filename and click Run, the program will automatically calculate the simulated data.")

            filep = self.filep_edit.text()
            remove_baseline = self.remove_baseline_checkbox.isChecked()
            psd_baseline_removed_l = float(self.psd_baseline_removed_l_edit.text())
            alphap = float(self.alphap_edit.text())
            peak_threshold_pct = float(self.peak_thresh_edit.text())
            min_distance = float(self.min_distance_edit.text())
            harmonics = self.harmonics_edit.text()
            refion = self.refion_edit.text()
            highlight_ions = self.highlight_ions_edit.text()
            circumference = float(self.circumference_edit.text())
            mode = self.mode_combo.currentText()
            sim_scalingfactor = float(self.sim_scalingfactor_edit.text())
            value = self.value_edit.text()
            reload_data = self.reload_data_checkbox.isChecked()
            nions = self.nions_edit.text()
            simulation_result= self.simulation_result_edit.text()
            matched_result= self.matched_result_edit.text()
            
            try:
                threshold = float(self.threshold_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching threshold")
            try:
                matching_freq_min = float(self.matching_freq_min_edit.text())
                matching_freq_max = float(self.matching_freq_max_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching_freq_min_edit")
                raise ValueError("Please enter a valid number for matching_freq_max_edit")

            args = argparse.Namespace(datafile=datafile,
                                        filep=filep or None,
                                        remove_baseline = remove_baseline or None,
                                        psd_baseline_removed_l=psd_baseline_removed_l or None,
                                        alphap=alphap or None,
                                        harmonics=harmonics or None,
                                        refion=refion or None,
                                        highlight_ions=highlight_ions or None,
                                        nions=nions or None,
                                        circumference=circumference or None,
                                        mode=mode or None,
                                        sim_scalingfactor=sim_scalingfactor or None,
                                        value=value or None,
                                        reload_data=reload_data or None,
                                        peak_threshold_pct=peak_threshold_pct,
                                        min_distance = min_distance,
                                        output_results=True,
                                        saved_data=self.saved_data,
                                        matching_freq_min=matching_freq_min,
                                        matching_freq_max=matching_freq_max,
                                        simulation_result=simulation_result
                                     )
            self.save_parameters()  # Save parameters before running the script
            # Simulate controller execution and emit data
            data = import_controller(**vars(args))
            self.saved_data = data
            if data.experimental_data:
                best_chi2, best_match_count, best_match_ions = data.compute_matches(threshold,matching_freq_min,matching_freq_max)
                data.save_matched_result(matched_result)
            self.visualization_signal.emit(data)        
    
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')
            log.error("Processing failed", exc_info=True)
            self.signalError.emit(str(e))
            
    def mousePressEvent(self, event):
        """
        Any mouse click on this widget will set the stop flag,
        causing the quick_pid_script loops to exit.
        """
        self._stop_quick_pid = True
        super().mousePressEvent(event)
        
    def quick_pid_script(self):
        try:
            print("Running quick_pid_script…")
            datafile = self.datafile_edit.text().strip()
            if not datafile:
                raise ValueError("No experimental data provided.")

            # --- collect constant arguments once ---
            filep = self.filep_edit.text() or None
            remove_baseline = self.remove_baseline_checkbox.isChecked()
            psd_baseline_removed_l = float(self.psd_baseline_removed_l_edit.text())
            alphap = float(self.alphap_edit.text())
            peak_threshold_pct = float(self.peak_thresh_edit.text())
            min_distance = float(self.min_distance_edit.text())
            harmonics = self.harmonics_edit.text()
            refion = self.refion_edit.text()
            highlight_ions = self.highlight_ions_edit.text() or None
            nions = self.nions_edit.text() or None
            circumference = float(self.circumference_edit.text())
            sim_scalingfactor = self.sim_scalingfactor_edit.text().strip()
            sim_scalingfactor = float(sim_scalingfactor) if sim_scalingfactor else None
            reload_data = self.reload_data_checkbox.isChecked()
            simulation_result= self.simulation_result_edit.text()
            matched_result= self.matched_result_edit.text()

            try:
                threshold = float(self.threshold_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching threshold")
            try:
                matching_freq_min = float(self.matching_freq_min_edit.text())
                matching_freq_max = float(self.matching_freq_max_edit.text())
            except ValueError:
                raise ValueError("Please enter a valid number for matching_freq_min_edit")
                raise ValueError("Please enter a valid number for matching_freq_max_edit")
                
            fref_min = float(self.fref_min_edit.text() or '-inf')
            fref_max = float(self.fref_max_edit.text() or 'inf')

            # --- 1) Load experimental data and detect peaks ---
            model = ImportData(
                refion=refion,
                highlight_ions=highlight_ions,
                remove_baseline = remove_baseline or None,
                psd_baseline_removed_l=psd_baseline_removed_l or None,
                alphap=alphap,
                filename=datafile,
                reload_data=reload_data,
                circumference=circumference,
                peak_threshold_pct=peak_threshold_pct,
                min_distance=min_distance,
                matching_freq_min=matching_freq_min,
                matching_freq_max=matching_freq_max
            )
            if not hasattr(model, 'peak_freqs') or len(model.peak_freqs) == 0:
                raise RuntimeError("Could not detect any experimental peaks.")
            self.visualization_signal.emit(model)
            # experimental peak frequencies (Hz)
            exp_peaks_hz = model.peak_freqs
            
            print(f"Detected {len(exp_peaks_hz)} experimental peaks.")

            # define your alphap scan range
            alphap_min  = float(self.alphap_min_edit.text())
            alphap_max  = float(self.alphap_max_edit.text())
            alphap_step = float(self.alphap_step_edit.text())
            
            # … your preamble: gather datafile, alphap, exp_peaks_hz, etc. …
            self._stop_quick_pid = False
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
            reload_data = True
            # Initialize first iteration flag
            first_iteration = True
            for f_ref in exp_peaks_hz_filtering:
                QApplication.processEvents()
                if self._stop_quick_pid:
                    print("Quick‐PID scan was stopped by user click.")
                    break

                # Highlight current f_ref in the UI
                self.value_edit.setStyleSheet("background-color: #fff8b0;")  
                self.value_edit.setText(f"{f_ref:.2f}")
                QApplication.processEvents()

                # Inner loop over a range of test_alphap values
                for test_alphap in np.arange(alphap_min, alphap_max + 1e-12, alphap_step):
                    start_time = time.time()  # Record start time for each test_alphap iteration
                    if self._stop_quick_pid:
                        print("Quick‐PID scan was stopped by user click.")
                        break
                    
                    # Update UI to show which alphap is being tested
                    self.alphap_edit.setStyleSheet("background-color: #b0fff8;")
                    self.alphap_edit.setText(f"{test_alphap:.6f}")
                    QApplication.processEvents()

                    # Run simulation for this combination
                    sim_args = argparse.Namespace(
                        datafile=datafile,
                        filep=filep,
                        remove_baseline = remove_baseline,
                        psd_baseline_removed_l = psd_baseline_removed_l,
                        alphap=test_alphap,
                        harmonics=harmonics,
                        refion=refion,
                        highlight_ions=highlight_ions,
                        nions=nions,
                        circumference=circumference,
                        mode='Frequency',
                        sim_scalingfactor=sim_scalingfactor,
                        value=f_ref,
                        reload_data=reload_data,
                        peak_threshold_pct=peak_threshold_pct,
                        min_distance=min_distance,
                        output_results=False,
                        saved_data=self.saved_data,
                        matching_freq_min=matching_freq_min,
                        matching_freq_max=matching_freq_max,
                        simulation_result=simulation_result
                    )
                    data_i = import_controller(**vars(sim_args))
                    if data_i is None:
                        continue
                    
                    chi2, match_count, highlights = data_i.compute_matches(threshold,matching_freq_min,matching_freq_max)
                    results.append((f_ref, test_alphap, chi2, match_count,highlights))
                    if first_iteration:
                        self.saved_data = data_i
                        self.overlay_sim_signal.emit(self.saved_data)
                        first_iteration = False  # Set flag to False after the first iteration
                    else:
                        reload_data = False
                    
                    del data_i  # Clear memory by deleting data_i after each iteration

                sorted_results = sorted(results, key=lambda x: (-x[3], x[2]))
                best_fref, best_alphap, best_chi2, best_match_count, best_match_ions = sorted_results[0]
                
                # Run simulation for this combination
                sim_args = argparse.Namespace(
                    datafile=datafile,
                    filep=filep,
                    remove_baseline = remove_baseline,
                    psd_baseline_removed_l = psd_baseline_removed_l,
                    alphap=best_alphap,
                    harmonics=harmonics,
                    refion=refion,
                    highlight_ions=highlight_ions,
                    nions=nions,
                    circumference=circumference,
                    mode='Frequency',
                    sim_scalingfactor=sim_scalingfactor,
                    value=best_fref,
                    reload_data=reload_data,
                    peak_threshold_pct=peak_threshold_pct,
                    min_distance=min_distance,
                    output_results=True,
                    saved_data=self.saved_data,
                    matching_freq_min=matching_freq_min,
                    matching_freq_max=matching_freq_max,
                    simulation_result=simulation_result
                )
                best_data = import_controller(**vars(sim_args))
                best_chi2, best_match_count, best_match_ions = best_data.compute_matches(threshold,matching_freq_min,matching_freq_max)
                best_data.save_matched_result(matched_result)
                self.save_parameters()  # Save parameters before running the script
                print(f"\n→ Best: f_ref={best_fref:.2f}Hz, alphap={best_alphap:.4f}, χ²={best_chi2:.3e}, matches={best_match_count} {best_match_ions}")
                
                self.mode_combo.setCurrentText('Frequency')
                self.value_edit.setText(f"{best_fref:.2f}")
                self.alphap_edit.setText(f"{best_alphap:.6f}")
                self.overlay_sim_signal.emit(best_data)
                QApplication.processEvents()
                # after inner loop, restore alphap style
                self.alphap_edit.setStyleSheet(orig_alpha_style)
                
                # after outer loop, restore value style
            self.value_edit.setStyleSheet(orig_value_style)
            self.save_parameters()  # Save parameters before running the script

        except Exception as e:
            # On any error, also ensure any highlight is reset if needed
            QMessageBox.critical(self, "Quick PID Error", str(e))
            log.error("quick_pid_script failed", exc_info=True)

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
