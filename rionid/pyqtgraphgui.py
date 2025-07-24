import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QDesktopWidget, QSpinBox
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
        # New lists to store specific colored items
        self.red_lines = []     # For red lines
        self.green_points = []  # For green points (if any; adjust as needed)
        self.yellow_lines = []  # For yellow lines
        self.red_triangles = None  # For red triangles (peaks)
        self.use_exp_height = True  # New: Default to use experimental height
        self.current_data = None    # New: Store current data for redrawing
        
    def getPlotWidget(self):
        return self.plot_widget    
        
    def getViewBox(self):
        return self.plot_widget.plotItem.getViewBox() 
        
    def eventFilter(self, obj, event):
        if obj is self.plot_widget and event.type() == QEvent.MouseButtonPress:
            # emit a PyQt signal instead of eating it
            self.plotClicked.emit()
            return True
        return super().eventFilter(obj, event)
        
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
        self.exp_data = data.experimental_data
        # Plot experimental data
        if hasattr(self, 'exp_data_line'):
            self.plot_widget.removeItem(self.exp_data_line)

        # Set the initial X-range to encompass all experimental data
        self.x_exp, self.z_exp = self.exp_data[0]*1e-6, self.exp_data[1]
        
        if self.saved_x_range is None:
            self.saved_x_range = (min(self.x_exp), max(self.x_exp))
            self.plot_widget.setXRange(*self.saved_x_range, padding=0.05)

            # Save the Y range as well
            min_z = np.min(self.z_exp)
            max_z = max(self.z_exp)
            if min_z <= 0:
                # Handle the logarithmic scale by setting the minimum to a small value if necessary
                min_z = 1e-10  # or some other small positive value
        
        self.exp_data_line = self.plot_widget.plot(self.x_exp, self.z_exp, pen=pg.mkPen('blue', width=3))
        self.legend.addItem(self.exp_data_line, 'Experimental Data')
        
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
        self.simulated_data = data.simulated_data_dict
        refion = data.ref_ion
        highlight_ions = data.highlight_ions # Get the list of ions to highlight in green
        for i, (harmonic, sdata) in enumerate(self.simulated_data.items()):
            color = pg.intColor(i, hues=len(self.simulated_data))
            for entry in sdata:
                freq = float(entry[0])*1e-6
                label = entry[2]
                yield_value = float(entry[1])  # Simulated yield (amplitude)
                freq_range = 0.005  # Adjust this range if needed
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
                    label_color = 'yellow'
                else:
                    label_color = color
                
                # Parse and convert label to superscript format
                match = re.match(r'(\d+)([A-Za-z]+)(\d+)\+', label)
                if match:
                    mass, elem, charge = match.groups()
                    new_label = self.to_superscript(mass) + elem + self.to_superscript(charge) + '⁺'
                else:
                    new_label = label
                
                # Vertical line
                line = self.plot_widget.plot([freq, freq], [1e-10, z_value], pen=pg.mkPen(color=label_color, width=1, style=Qt.DashLine))
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
                elif label_color == 'green':
                    self.green_points.append((line, text))  # Treating green lines/text as "points"
                elif label_color == 'yellow':
                    self.yellow_lines.append((line, text))

            self.legend.addItem(line, f'Harmonic = {float(harmonic)} ; Bρ = {data.brho:.6f} [Tm].')
            self.legend.addItem(
                line,
                f"reference frequency = {data.ref_frequency:.2f} Hz ; "
                f"αₚ = {data.alphap:.4f} ; "
                f"γₜ = {data.gammat:.4f} ; "
                f"χ² = {data.chi2:.1f} ; "
                f"match_count = {int(data.match_count)}"
            )            
            # compute threshold
            rel_height = getattr(data, 'peak_threshold_pct', 0.05)
            rel_height = max(0.0, min(rel_height, 1.0))
            if hasattr(self, 'z_exp') and self.z_exp is not None:
                threshold_val = rel_height * np.max(self.z_exp)
            else:
                threshold_val = 0.01            
            if self.plot_widget.plotItem.ctrl.logYCheck.isChecked():
                pos = np.log10(threshold_val)
            else:
                pos = threshold_val
                    
            self.threshold_line = pg.InfiniteLine(
                pos=pos, angle=0,
                pen=pg.mkPen('blue', style=Qt.DashLine, width=1)
            )
            self.plot_widget.addItem(self.threshold_line)
        
            if self.saved_x_range is not None:
                x0 = self.saved_x_range[0]
            else:
                x0 = 0        
            self.threshold_label_item = pg.TextItem(
                        html=f'<span style="color:blue;">Threshold = {rel_height*1:.1e}</span>',
                        anchor=(0, 1)
                    )
            self.threshold_label_item.setPos(x0, pos)
            self.plot_widget.addItem(self.threshold_label_item)

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
        self.clear_experimental_data()
        self.clear_simulated_data()
        self.plot_all_data(data)

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

    def clear_experimental_data(self):
        if hasattr(self, 'exp_data_line'):
            print("Clearing experimental data plot...")
            self.plot_widget.removeItem(self.exp_data_line)
            self.legend.removeItem(self.exp_data_line)
        if self.red_triangles:
            self.plot_widget.removeItem(self.red_triangles)
            self.legend.removeItem(self.red_triangles)
        self.exp_data_line = None
        self.exp_data = None
        self.red_triangles = None

    def toggle_simulated_data(self):
        for line, text in self.simulated_items:
            line.setVisible(not line.isVisible())
            text.setVisible(not text.isVisible())

    def toggle_annotations(self):
        # Toggle visibility of red lines
        for line, text in self.red_lines:
            line.setVisible(not line.isVisible())
            text.setVisible(not text.isVisible())
        # Toggle visibility of green points/lines
        for item in self.green_points:
            if isinstance(item, tuple):  # If stored as (line, text)
                line, text = item
                line.setVisible(not line.isVisible())
                text.setVisible(not text.isVisible())
            else:
                item.setVisible(not item.isVisible())
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
            self.cursor_pos_label.setText(f"Cursor Position: x={mousePoint.x():.8f}, y={mousePoint.y():.2f}")

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
        button_layout = QHBoxLayout()

        font = QFont("Times", 15)
        font.setBold(True)

        toggle_sim_button = QPushButton("Toggle Simulated Data")
        toggle_sim_button.clicked.connect(self.toggle_simulated_data)
        toggle_sim_button.setFont(font)
        button_layout.addWidget(toggle_sim_button)

        # New button for toggling annotations
        toggle_anno_button = QPushButton("Toggle Annotations")
        toggle_anno_button.clicked.connect(self.toggle_annotations)
        toggle_anno_button.setFont(font)
        button_layout.addWidget(toggle_anno_button)

        # New button for toggling height source
        toggle_height_button = QPushButton("Toggle Height Source")
        toggle_height_button.clicked.connect(self.toggle_height_source)
        toggle_height_button.setFont(font)
        button_layout.addWidget(toggle_height_button)

        save_button = QPushButton("Save Plot")
        save_button.clicked.connect(self.save_plot_with_dialog)
        save_button.setFont(font)
        button_layout.addWidget(save_button)

        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(lambda: self.plot_widget.getViewBox().scaleBy((0.5, 0.5)))
        zoom_in_button.setFont(font)
        button_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(lambda: self.plot_widget.getViewBox().scaleBy((2, 2)))
        zoom_out_button.setFont(font)
        button_layout.addWidget(zoom_out_button)

        # Button to reset the plot view
        reset_view_button = QPushButton("Reset View")
        reset_view_button.setFont(font)
        reset_view_button.clicked.connect(self.reset_view)
        button_layout.addWidget(reset_view_button)

        # 新增: Font size spinbox
        font_label = QLabel("Font Size:")
        font_label.setFont(font)
        button_layout.addWidget(font_label)
        self.font_spinbox = QSpinBox()
        self.font_spinbox.setRange(10, 30)  # 范围 10-30
        self.font_spinbox.setValue(self.font_size)  # 初始值
        self.font_spinbox.valueChanged.connect(self.update_fonts)  # 绑定更新方法
        button_layout.addWidget(self.font_spinbox)

        main_layout.addLayout(button_layout)


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