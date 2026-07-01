import sys
import os
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QDesktopWidget, QSpinBox, QLineEdit
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QLoggingCategory, Qt
from PyQt5.QtCore import QEvent, pyqtSignal
from pyqtgraph.exporters import ImageExporter
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtPrintSupport import QPrinter  # Import for PDF export
import re  # For parsing ion labels

class CustomLegendItem(pg.LegendItem):
    def __init__(self, font_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font = QFont("Arial", font_size)  # 使用字体大小参数

    def addItem(self, item, name):
        label = pg.LabelItem(text=name, justify='left')
        label.setFont(self.font)
        super().addItem(item, name)

    def updateFont(self, font_size):
        self.font.setPointSize(font_size)
        # 重新绘制 legend 需要清空并重新添加项，但由于 legend 项动态，这里假设在重新绘制时重建

class CreatePyGUI(QMainWindow):
    '''
    PyView (MVC)
    '''
    # signal to let the controller know the user clicked on the plot
    plotClicked = pyqtSignal()
    thresholdClickModeChanged = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_x_range = None  
        self.saved_y_range = None 
        self.simulated_items = []
        self._stop_quick_pid = False
        self.font_size = 20  # 新增: 默认字体大小参数
        self.setup_ui()
        # install filter on the plot area only
        self.plot_widget.installEventFilter(self)
        self._threshold_refresh_pending = False
        # New lists to store specific colored items
        self.red_lines = []     # For red lines
        self.green_points = []  # For green points (if any; adjust as needed)
        self.yellow_lines = []  # For yellow lines
        self.annotation_ref_highlight_items = []  # Black (ref ion) + Green (highlight ions) for toggle_annotations
        self.red_triangles = None  # For red triangles (peaks)
        self.threshold_profile_line = None
        self.threshold_profile_points = None
        self.use_exp_height = True  # New: Default to use experimental height
        self.current_data = None    # New: Store current data for redrawing
        self.experimental_data = None
        self.remove_baseline=None
        self._threshold_click_enabled = False
        self.psd_baseline_removed = None
        self.psd_baseline = None
        
    def getPlotWidget(self):
        return self.plot_widget    
        
    def getViewBox(self):
        return self.plot_widget.plotItem.getViewBox() 
        
    def eventFilter(self, obj, event):
        if obj is self.plot_widget and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and getattr(self, '_threshold_click_enabled', False):
                self._handle_plot_click(event)
                self._threshold_click_enabled = False
                self.thresholdClickModeChanged.emit(False)
                return True
            if event.button() == Qt.RightButton:
                if self._handle_plot_right_click(event):
                    return True
            self.plotClicked.emit()
            return True
        return super().eventFilter(obj, event)

    def _handle_plot_click(self, event):
        pos = self.plot_widget.plotItem.vb.mapSceneToView(event.pos())
        self._last_click_freq = float(pos.x())
        if self.plot_widget.plotItem.ctrl.logYCheck.isChecked():
            self._last_click_amp = float(10 ** pos.y())
        else:
            self._last_click_amp = float(pos.y())

        if not hasattr(self, 'current_data') or self.current_data is None:
            return

        data = self.current_data
        if data.experimental_data is None:
            return

        if getattr(data, 'remove_baseline', False) and getattr(data, 'psd_baseline_removed', None) is not None:
            freq_axis, amp = data.psd_baseline_removed
        else:
            freq_axis, amp = data.experimental_data

        if len(freq_axis) == 0:
            return

        freq_mhz = self._last_click_freq
        freq_hz = freq_mhz * 1e6
        if freq_hz <= 0:
            return

        try:
            selected_path = self._get_threshold_path_from_ui()
            if selected_path:
                data.threshold_profile_path = selected_path
                data._load_threshold_profile(selected_path)
            data.update_threshold_profile_from_clicks(freq_hz, float(self._last_click_amp))
            data.detect_peaks_and_widths()
            self.updateData(data)
        except Exception as exc:
            print(f"Failed to refresh threshold profile from click: {exc}")

    def _get_threshold_path_from_ui(self):
        if self.current_data is not None:
            return getattr(self.current_data, 'threshold_profile_path', None)
        return None

    def _apply_threshold_path(self, path, refresh=True):
        if not path:
            return
        if self.current_data is not None:
            self.current_data.threshold_profile_path = path
            self.current_data._load_threshold_profile(path)
            if refresh:
                self.updateData(self.current_data)

    def select_threshold_file(self):
        default_path = self._get_threshold_path_from_ui() or ''
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Select height_thresh.csv', default_path, 'CSV Files (*.csv)')
        if filename:
            if not filename.lower().endswith('.csv'):
                filename = f'{filename}.csv'
            self._apply_threshold_path(filename, refresh=True)

    def toggle_threshold_click_mode(self):
        self._threshold_click_enabled = not self._threshold_click_enabled
        self.thresholdClickModeChanged.emit(self._threshold_click_enabled)
        return self._threshold_click_enabled

    def toggle_threshold_visibility(self):
        """Toggle visibility of the threshold profile curve and points."""
        for item in (self.threshold_profile_line, self.threshold_profile_points):
            if item is not None:
                item.setVisible(not item.isVisible())

        
    def setup_ui(self):
        self.setWindowTitle('Schottky Signals Identifier')
        width = QDesktopWidget().screenGeometry(-1).width()
        height = QDesktopWidget().screenGeometry(-1).height()
        self.setGeometry(100, 100, width, height)  # Set window size
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)
        #logging annoying messages
        QLoggingCategory.setFilterRules('*.warning=false\n*.critical=false')
        # Create the plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # White background
        # Set logY to true
        self.plot_widget.plotItem.ctrl.logYCheck.setChecked(True)
        # Enable all borders
        view_box = self.plot_widget.getViewBox()
        view_box.setBorder((50, 50, 50))  # 设置边框宽度 (left, top, right, bottom)，单位像素

        main_layout.addWidget(self.plot_widget)
        # Add legend with font size
        self.legend = CustomLegendItem(self.font_size, offset=(-10, 10))  # 使用自定义 legend
        self.legend.setParentItem(self.plot_widget.graphicsItem())
        self.legend.setBrush(pg.mkBrush('white'))  # Legend background white
        self.legend.setPen(pg.mkPen('black'))     # Legend border black
        self.legend.setLabelTextColor('black')    # Legend text black
        
        self.plot_widget.setLabel(
            "left",
            '<span style="color: black; font-size: {}px">Amplitude (arb. units)</span>'.format(self.font_size)
        )
        self.plot_widget.setLabel(
            "bottom",
            '<span style="color: black; font-size: {}px">Frequency (MHz)</span>'.format(self.font_size)
        )
        
        # Set axis lines and ticks to black
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen('black', width=1))
        self.plot_widget.getAxis('left').setPen(pg.mkPen('black', width=1))
        self.plot_widget.getAxis('bottom').setTextPen('black')
        self.plot_widget.getAxis('left').setTextPen('black')
        
        # Customizing tick label font size
        self.font_ticks = QFont()
        self.font_ticks.setPixelSize(self.font_size)
        self.plot_widget.getAxis('bottom').setTickFont(self.font_ticks)
        self.plot_widget.getAxis("bottom").setStyle(tickTextOffset = 15)
        self.plot_widget.getAxis('left').setTickFont(self.font_ticks)
        self.plot_widget.getAxis("left").setStyle(tickTextOffset = 15)
        
        # Cursor position label
        font = QFont("Times", 12)
        self.cursor_pos_label = QLabel(self)
        self.cursor_pos_label.setFont(font)
        self.cursor_pos_label.setStyleSheet("color: black;")
        main_layout.addWidget(self.cursor_pos_label)
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        
        # Add control buttons
        self.add_buttons(main_layout)
    
    def plot_all_data(self, data):
        print("plotting all data...")
        self.plot_widget.clear()
        self.plot_experimental_data(data)
        self.plot_simulated_data(data)

    def plot_experimental_data(self, data):
        print("plotting experimental data...")
        if data.experimental_data is None:  # Check if experimental data is available
            print("No experimental data available, skipping experimental data plotting.")
            return  # Skip plotting experimental data
            
        # Plot experimental data            
        self.experimental_data = data.experimental_data
        
        self.remove_baseline = data.remove_baseline
        self.psd_baseline_removed = data.psd_baseline_removed
        self.psd_baseline = data.psd_baseline
        
        print("self.remove_baseline ",self.remove_baseline)
        print("self.psd_baseline_removed ",self.psd_baseline_removed)
        print("self.psd_baseline ",self.psd_baseline)
        
        if hasattr(self, 'experimental_data_line'):
            self.plot_widget.removeItem(self.experimental_data_line)

        # Set the initial X-range to encompass all experimental data、
        self.x_exp, self.z_exp = self.experimental_data[0]*1e-6, self.experimental_data[1]
        
        if self.saved_x_range is None:
            self.saved_x_range = (min(self.x_exp), max(self.x_exp))
            self.plot_widget.setXRange(*self.saved_x_range, padding=0.05)

            # Save the Y range as well
            min_z = np.min(self.z_exp)
            max_z = max(self.z_exp)
            if min_z <= 0:
                # Handle the logarithmic scale by setting the minimum to a small value if necessary
                min_z = 1e-10  # or some other small positive value
        
        self.experimental_data_line = self.plot_widget.plot(self.x_exp, self.z_exp, pen=pg.mkPen('blue', width=3))
        self.legend.addItem(self.experimental_data_line, 'Experimental Data')

        # Plot baseline and baseline-removed curves if available
        if self.remove_baseline:
            if self.psd_baseline is not None:
                baseline_x = self.psd_baseline[0] * 1e-6
                baseline_y = self.psd_baseline[1]
                self.baseline_line = self.plot_widget.plot(baseline_x, baseline_y, pen=pg.mkPen('orange', style=Qt.DashLine, width=2))
                self.legend.addItem(self.baseline_line, 'Baseline')
            if self.psd_baseline_removed is not None:
                baseline_removed_x = self.psd_baseline_removed[0] * 1e-6
                baseline_removed_y = self.psd_baseline_removed[1]
                self.baseline_removed_line = self.plot_widget.plot(baseline_removed_x, baseline_removed_y, pen=pg.mkPen('magenta', style=Qt.DotLine, width=1))
                self.legend.addItem(self.baseline_removed_line, 'Baseline Removed')

        if getattr(data, 'threshold_profile_freqs', None) is not None and getattr(data, 'threshold_profile_vals', None) is not None:
            threshold_x = data.threshold_profile_freqs * 1e-6
            threshold_y = data.threshold_profile_vals
            self.threshold_profile_line = self.plot_widget.plot(
                threshold_x,
                threshold_y,
                pen=pg.mkPen('green', width=2, style=Qt.DashLine)
            )
            self.legend.addItem(self.threshold_profile_line, 'Threshold')
            self._plot_threshold_profile_points(data)

        # --- Mark each detected peak with a red triangle ---
        # 1) Convert peak frequencies to MHz and get peak heights
        peak_x = data.peak_freqs * 1e-6
        peak_y = data.peak_heights
    
        # 2) Plot red triangles at (peak_x, peak_y)
        self.red_triangles = self.plot_widget.plot(
            peak_x,
            peak_y,
            pen=None,               # no line
            symbol='t',             # triangle marker
            symbolBrush='r',        # red fill
            symbolSize=12           # size in pixels
        )
        self.legend.addItem(self.red_triangles, 'Peaks')
        
    def to_superscript(self, s):
        supers = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        return ''.join(supers.get(c, c) for c in s)
        
    def plot_simulated_data(self, data):
        print("plotting simulated data...")
        print("data.matching_freq_min = ",data.matching_freq_min)
        print("data.matching_freq_max = ",data.matching_freq_max)
        self.simulated_data = data.simulated_data_dict
        refion = data.ref_ion
        highlight_ions = data.highlight_ions # Get the list of ions to highlight in green
        for i, (harmonic, sdata) in enumerate(self.simulated_data.items()):
            color = pg.intColor(i, hues=len(self.simulated_data))
            line = None
            for entry in sdata:
                freq = float(entry[0])*1e-6
                label = entry[2]
                yield_value = float(entry[1])  # Simulated yield (amplitude)
                freq_range = 0.005  # Adjust this range if needed
                if freq>data.matching_freq_min/1e6 and freq< data.matching_freq_max/1e6:
                    if self.use_exp_height and data.experimental_data is not None:  # New: Check switch
                        z_value = self.get_z_exp_at_freq(freq, freq_range)  # Use experimental height
                        if z_value is None:  # Fallback if no match
                            z_value = yield_value
                    else:
                        z_value = yield_value  # Use simulated height
                    
                    label_color = None
                    if highlight_ions is not None and label in highlight_ions:
                        label_color = 'green'
                    elif label == refion:
                        label_color = 'black'
                    else:
                        label_color = color
                        
                    # Parse and convert label to superscript format
                    match = re.match(r'(\d+)([A-Za-z]+)(\d+)\+', label)
                    if match:
                        mass, elem, charge = match.groups()
                        new_label = self.to_superscript(mass) + elem + self.to_superscript(charge) + '⁺'
                    else:
                        new_label = label
                    # Add harmonic number to the label
                    new_label = f'{new_label} h{int(float(harmonic))}'

                    # Vertical line
                    line = self.plot_widget.plot([freq, freq], [1e-30, z_value], pen=pg.mkPen(color=label_color, width=1, style=Qt.DashLine))
                    # Text label at top with adjustable font size
                    text = pg.TextItem(text=new_label, color=label_color, anchor=(0,0.5))
                    text.setFont(QFont("Arial", self.font_size))  # 新增: 使用 self.font_size 设置离子符号字体
                    self.plot_widget.addItem(text)
                    text.setAngle(90)
                    text_width_pixels = text.boundingRect().width()
                    view_box = self.plot_widget.plotItem.vb
                    text_width_data_units = view_box.mapSceneToView(pg.QtCore.QPointF(text_width_pixels, 0)).x() - view_box.mapSceneToView(pg.QtCore.QPointF(0, 0)).x()
                    logy_checked = self.plot_widget.plotItem.ctrl.logYCheck.isChecked()
                    
                    if logy_checked:
                        y_position = np.log10(z_value) if z_value != 0 else 0
                    else:
                        y_position = z_value
                    x_position = freq
                    text.setPos(x_position, y_position)
                    
                    self.simulated_items.append((line, text))
                    
                    # Store in specific lists based on color
                    if label_color == 'red':
                        self.red_lines.append((line, text))
                    elif label_color in ('green', 'black'):
                        self.green_points.append((line, text))  # Green (highlight) + Black (reference) as one group
                    elif label_color == 'yellow':
                        self.yellow_lines.append((line, text))
                    # Explicitly track reference + highlight ions for toggle_annotations
                    if label == refion or (highlight_ions is not None and label in highlight_ions):
                        self.annotation_ref_highlight_items.append((line, text))

            if line is not None:
                self.legend.addItem(line, f'Harmonic = {float(harmonic)} ; Bρ = {data.brho:.6f} [Tm].')
                self.legend.addItem(
                    line,
                    f"reference frequency = {data.ref_frequency:.2f} Hz ; "
                    f"αₚ = {data.alphap:.4f} ; "
                    f"γₜ = {data.gammat:.4f} ; "
                    f"χ² = {data.chi2:.1f} ; "
                    f"match_count = {int(data.match_count)}"
                )

    def get_z_exp_at_freq(self, freq, freq_range):
        if len(self.x_exp) == 0 or len(self.z_exp) == 0:
            return None

        lower_bound = freq - freq_range
        upper_bound = freq + freq_range

        indices = (self.x_exp >= lower_bound) & (self.x_exp <= upper_bound)
    
        if np.any(indices):
            return np.max(self.z_exp[indices])
        else:
            closest_index = np.argmin(np.abs(self.x_exp - freq))
            return self.z_exp[closest_index]

    
    def updateData(self, data):
        print("Updating data in visualization GUI...")
        self.current_data = data  # New: Save current data
        # Update data file path display
        if hasattr(self, "datafile_path_display") and data is not None:
            path = getattr(data, "filename", "") or ""
            self.datafile_path_display.setText(path)
        self.clear_experimental_data()
        self.clear_simulated_data()
        self.plot_all_data(data)

    def _plot_threshold_profile_points(self, data):
        if self.threshold_profile_points is not None:
            try:
                self.plot_widget.removeItem(self.threshold_profile_points)
            except Exception:
                pass
            self.threshold_profile_points = None

        if getattr(data, 'threshold_profile_freqs', None) is None or getattr(data, 'threshold_profile_vals', None) is None:
            return

        threshold_x = data.threshold_profile_freqs * 1e-6
        threshold_y = data.threshold_profile_vals
        self.threshold_profile_points = self.plot_widget.plot(
            threshold_x,
            threshold_y,
            pen=None,
            symbol='star',
            symbolBrush='green',
            symbolPen=pg.mkPen('darkgreen', width=1),
            symbolSize=14
        )

    def _handle_plot_right_click(self, event):
        if self.current_data is None or self.current_data.threshold_profile_freqs is None:
            return False

        pos = self.plot_widget.plotItem.vb.mapSceneToView(event.pos())
        click_freq_hz = float(pos.x()) * 1e6
        click_y = float(pos.y())

        freqs = self.current_data.threshold_profile_freqs
        vals = self.current_data.threshold_profile_vals
        if len(freqs) == 0:
            return False

        # Find the nearest threshold point in screen space
        nearest_index = None
        nearest_distance = float('inf')
        for idx, (freq_hz, val) in enumerate(zip(freqs, vals)):
            point_scene = self.plot_widget.plotItem.vb.mapViewToScene(pg.Point(freq_hz * 1e-6, val))
            distance = ((point_scene.x() - event.pos().x())**2 + (point_scene.y() - event.pos().y())**2) ** 0.5
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = idx

        # Only show menu when click is close enough to a threshold point
        if nearest_index is None or nearest_distance > 12:
            return False

        menu = QtWidgets.QMenu(self.plot_widget)
        delete_action = menu.addAction('Delete threshold point')
        action = menu.exec_(event.screenPos().toPoint())
        if action == delete_action:
            self._delete_threshold_point(nearest_index)
            return True
        return False

    def _delete_threshold_point(self, index):
        if self.current_data is None:
            return
        freqs = np.array(self.current_data.threshold_profile_freqs, dtype=float)
        vals = np.array(self.current_data.threshold_profile_vals, dtype=float)
        if index < 0 or index >= len(freqs):
            return
        freqs = np.delete(freqs, index)
        vals = np.delete(vals, index)
        self.current_data.threshold_profile_freqs = freqs if len(freqs) > 0 else None
        self.current_data.threshold_profile_vals = vals if len(vals) > 0 else None
        if len(freqs) > 0:
            self.current_data._save_threshold_profile(freqs, vals)
        else:
            if self.current_data.threshold_profile_path and os.path.exists(self.current_data.threshold_profile_path):
                os.remove(self.current_data.threshold_profile_path)
        self.current_data.detect_peaks_and_widths()
        self.updateData(self.current_data)

    def clear_simulated_data(self):
        print("Clearing simulated data plots...")
        while self.simulated_items:
            line, text = self.simulated_items.pop()
            self.plot_widget.removeItem(line)
            self.plot_widget.removeItem(text)
        self.legend.clear()
        self.simulated_data = None
        # Clear specific lists
        self.red_lines = []
        self.green_points = []
        self.yellow_lines = []
        self.annotation_ref_highlight_items = []

    def clear_experimental_data(self):
        if hasattr(self, 'experimental_data_line') and self.experimental_data_line is not None:
            print("Clearing experimental data plot...")
            self.plot_widget.removeItem(self.experimental_data_line)
            self.legend.removeItem(self.experimental_data_line)
        if hasattr(self, 'baseline_line') and self.baseline_line is not None:
            self.plot_widget.removeItem(self.baseline_line)
            self.legend.removeItem(self.baseline_line)
        if hasattr(self, 'baseline_removed_line') and self.baseline_removed_line is not None:
            self.plot_widget.removeItem(self.baseline_removed_line)
            self.legend.removeItem(self.baseline_removed_line)
        if self.red_triangles:
            self.plot_widget.removeItem(self.red_triangles)
            self.legend.removeItem(self.red_triangles)
        if hasattr(self, 'threshold_profile_line') and self.threshold_profile_line is not None:
            self.plot_widget.removeItem(self.threshold_profile_line)
            self.legend.removeItem(self.threshold_profile_line)
        if hasattr(self, 'threshold_profile_points') and self.threshold_profile_points is not None:
            self.plot_widget.removeItem(self.threshold_profile_points)
        self.experimental_data_line = None
        self.experimental_data = None
        self.red_triangles = None
        self.baseline_line = None
        self.baseline_removed_line = None
        self.threshold_profile_line = None
        self.threshold_profile_points = None

    def toggle_simulated_data(self):
        for line, text in self.simulated_items:
            line.setVisible(not line.isVisible())
            text.setVisible(not text.isVisible())

    def toggle_annotations(self):
        # Toggle visibility of red lines
        for line, text in self.red_lines:
            line.setVisible(not line.isVisible())
            text.setVisible(not text.isVisible())
        # Toggle visibility of reference + highlight ions (black + green)
        for line, text in self.annotation_ref_highlight_items:
            line.setVisible(not line.isVisible())
            text.setVisible(not text.isVisible())
        # Toggle visibility of yellow lines
        for line, text in self.yellow_lines:
            line.setVisible(not line.isVisible())
            text.setVisible(not text.isVisible())
        # Toggle visibility of red triangles
        if self.red_triangles:
            self.red_triangles.setVisible(not self.red_triangles.isVisible())

    def toggle_height_source(self):
        self.use_exp_height = not self.use_exp_height  # New: Toggle the switch
        if self.current_data is not None:
            self.updateData(self.current_data)  # Redraw with new height source

    def mouse_moved(self, evt):
        pos = evt[0]  # using signal proxy turns original arguments into a tuple
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            if self.plot_widget.plotItem.ctrl.logYCheck.isChecked():
                self.cursor_pos_label.setText(f"Cursor Position: x={mousePoint.x():.8f}, y={10**(mousePoint.y()):.6e}")
            else:
                self.cursor_pos_label.setText(f"Cursor Position: x={mousePoint.x():.8f}, y={mousePoint.y():.6e}")

    def save_plot_with_dialog(self):
        from PyQt5.QtPrintSupport import QPrinter
        from pyqtgraph.exporters import SVGExporter  # 新增导入 SVGExporter
        options = QtWidgets.QFileDialog.Options()
        filename, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save Plot As...",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",  # 添加 SVG 选项
            options=options
        )
        
        if filename:  
            if filename.lower().endswith(".png") or "png" in selected_filter:
                exporter = ImageExporter(self.plot_widget.plotItem)
                exporter.parameters()['width'] = 1600  # 高宽度以提高质量
                exporter.export(filename)
                print(f"Plot saved as PNG: {filename}")
        
            elif filename.lower().endswith(".pdf") or "pdf" in selected_filter:
                printer = QPrinter()
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(filename)
                printer.setResolution(600)  # 提高到 600 DPI 以改善粗糙
                printer.setFullPage(True)
                painter = QtGui.QPainter(printer)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)  # 抗锯齿
                painter.setRenderHint(QtGui.QPainter.HighQualityAntialiasing, True)
                self.plot_widget.render(painter)
                painter.end()
                print(f"Plot saved as PDF: {filename}")
            
            elif filename.lower().endswith(".svg") or "svg" in selected_filter:
                exporter = SVGExporter(self.plot_widget.plotItem)
                exporter.export(filename)
                print(f"Plot saved as SVG: {filename}. Convert to EPS using Inkscape: inkscape {filename} -o output.eps")
                
    def save_selected_data(self):
        selected_range = self.plot_widget.getViewBox().viewRange()[0]
        mask = (self.x_exp >= selected_range[0]) & (self.x_exp <= selected_range[1])
        selected_data = self.z_exp[mask]
        selected_x = self.x_exp[mask]
        filename = 'selected_data.npz'
        np.savez(filename, x=selected_x, z=selected_data)
        print(f"Data saved to {filename}")

    def reset_view(self):
        # Reset the plot to the original X and Y ranges
        self.plot_widget.setXRange(*self.saved_x_range, padding=0.05)  # Use saved_x_range
        self.plot_widget.setYRange(np.min(self.z_exp), np.max(self.z_exp), padding=0.05)

    def update_fonts(self, new_size):
        self.font_size = new_size
        # 更新轴标签
        self.plot_widget.setLabel(
            "left",
            '<span style="color: black; font-size: {}px">Amplitude (arb. units)</span>'.format(self.font_size)
        )
        self.plot_widget.setLabel(
            "bottom",
            '<span style="color: black; font-size: {}px">Frequency (MHz)</span>'.format(self.font_size)
        )
        # 更新轴刻度字体
        self.font_ticks.setPixelSize(self.font_size)
        self.plot_widget.getAxis('bottom').setTickFont(self.font_ticks)
        self.plot_widget.getAxis('left').setTickFont(self.font_ticks)
        # 更新 legend 字体（重建 legend）
        self.legend.updateFont(self.font_size)
        # 重新绘制图表以更新离子符号字体
        if self.current_data is not None:
            self.updateData(self.current_data)

    def add_buttons(self, main_layout):
        font = QFont("Times", 15)
        font.setBold(True)

        first_row_layout = QHBoxLayout()

        toggle_sim_button = QPushButton("Toggle Simulated Data")
        toggle_sim_button.clicked.connect(self.toggle_simulated_data)
        toggle_sim_button.setFont(font)
        first_row_layout.addWidget(toggle_sim_button)

        # New button for toggling annotations
        toggle_anno_button = QPushButton("Toggle Annotations")
        toggle_anno_button.clicked.connect(self.toggle_annotations)
        toggle_anno_button.setFont(font)
        first_row_layout.addWidget(toggle_anno_button)

        # New button for toggling height source
        toggle_height_button = QPushButton("Toggle Height Source")
        toggle_height_button.clicked.connect(self.toggle_height_source)
        toggle_height_button.setFont(font)
        first_row_layout.addWidget(toggle_height_button)

        save_button = QPushButton("Save Plot")
        save_button.clicked.connect(self.save_plot_with_dialog)
        save_button.setFont(font)
        first_row_layout.addWidget(save_button)

        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(lambda: self.plot_widget.getViewBox().scaleBy((0.5, 0.5)))
        zoom_in_button.setFont(font)
        first_row_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(lambda: self.plot_widget.getViewBox().scaleBy((2, 2)))
        zoom_out_button.setFont(font)
        first_row_layout.addWidget(zoom_out_button)

        # Button to reset the plot view
        reset_view_button = QPushButton("Reset View")
        reset_view_button.setFont(font)
        reset_view_button.clicked.connect(self.reset_view)
        first_row_layout.addWidget(reset_view_button)

        # 新增: Font size spinbox
        font_label = QLabel("Font Size:")
        font_label.setFont(font)
        first_row_layout.addWidget(font_label)
        self.font_spinbox = QSpinBox()
        self.font_spinbox.setRange(10, 30)  # 范围 10-30
        self.font_spinbox.setValue(self.font_size)  # 初始值
        self.font_spinbox.valueChanged.connect(self.update_fonts)  # 绑定更新方法
        first_row_layout.addWidget(self.font_spinbox)

        # Toggle Threshold visibility
        toggle_thresh_button = QPushButton("Toggle Threshold")
        toggle_thresh_button.clicked.connect(self.toggle_threshold_visibility)
        toggle_thresh_button.setFont(font)
        first_row_layout.addWidget(toggle_thresh_button)

        main_layout.addLayout(first_row_layout)


        # Data file path display (read-only)
        self.datafile_path_label = QLabel("Data File Path:")
        self.datafile_path_label.setFont(font)
        self.datafile_path_display = QLineEdit()
        self.datafile_path_display.setFont(font)
        self.datafile_path_display.setReadOnly(True)
        hbox_datafile_path = QHBoxLayout()
        hbox_datafile_path.addWidget(self.datafile_path_label)
        hbox_datafile_path.addWidget(self.datafile_path_display)
        main_layout.addLayout(hbox_datafile_path)


# Example Usage:
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Example data (adjust as per your actual data structure)
    class DummyData:
        experimental_data = np.array([[2.35000019e+08, 9.04612897e-02],
                                      [2.35000057e+08, 9.07298288e-02],
                                      [2.35000095e+08, 9.01448335e-02],
                                      [2.54999905e+08, 9.01264557e-02],
                                      [2.54999943e+08, 9.01772547e-02],
                                      [2.54999981e+08, 9.03425368e-02]])
        peak_freqs = np.array([2.35e+08, 2.55e+08])
        peak_heights = np.array([0.09, 0.09])
        simulated_data_dict = {
            '1.0': np.array([['241127381.22165576', '0.00054777', '80Kr+35'],
                             ['242703150.0762615', '0.0048654', '79Br+35']])
        }
        ref_ion = '80Kr+35'
        highlight_ions = ['79Br+35']
        brho = 5.849431
        ref_frequency = 15686564.45
        alphap = 0.05460
        gammat = 1.3533
        chi2 = 20826.8
        match_count = 13

    data = DummyData()
    sa = CreatePyGUI()
    sa.updateData(data)
    sa.show()
    sys.exit(app.exec_())
