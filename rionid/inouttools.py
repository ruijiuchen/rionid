from iqtools.tools import read_rsa_specan_xml, read_rsa_data_csv, read_rsa_result_csv
import numpy as np
import os
import ROOT
from PyQt5.QtWidgets import QDialog, QFormLayout, QComboBox, QHBoxLayout, QPushButton,QLineEdit,QMessageBox,QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class KeySelectionDialog(QDialog):
    def __init__(self, parent=None, keys=None, filename=None):
        super().__init__(parent)
        self.setWindowTitle("Select Keys for NPZ File")
        self.filename = filename
        self.colorbar = None  # To track the colorbar
        
        # Set Matplotlib backend to Qt5Agg
        plt.switch_backend('Qt5Agg')
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Form layout for inputs
        form_layout = QFormLayout()
        
        self.freq_combo = QComboBox()
        self.amp_combo = QComboBox()
        self.time_combo = QComboBox()
        self.time_px_min_edit = QLineEdit()
        self.time_px_max_edit = QLineEdit()
        
        if keys:
            self.freq_combo.addItems(keys)
            self.amp_combo.addItems(keys)
            self.time_combo.addItems(keys)
            # Optionally set default selections
            if 'arr_0' in keys:
                self.freq_combo.setCurrentText('arr_0')
            if 'arr_1' in keys:
                self.amp_combo.setCurrentText('arr_1')
            if 'arr_2' in keys:
                self.time_combo.setCurrentText('arr_2')
        
        form_layout.addRow("Frequency Key:", self.freq_combo)
        form_layout.addRow("Amplitude Key:", self.amp_combo)
        form_layout.addRow("Time Key:", self.time_combo)
        form_layout.addRow("Time(Projection, min):", self.time_px_min_edit)
        form_layout.addRow("Time(Projection, max):", self.time_px_max_edit)
        
        # Buttons
        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        
        self.preview_btn.clicked.connect(self.preview_plot)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Combine layouts
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(buttons)
        self.setLayout(main_layout)
    
    def preview_plot(self):
        if not self.filename:
            QMessageBox.warning(self, "Error", "No filename provided for preview.")
            return
        
        try:
            data = np.load(self.filename)
            freq_key = self.freq_combo.currentText()
            amp_key = self.amp_combo.currentText()
            time_key = self.time_combo.currentText()
            if not all([freq_key, amp_key, time_key]):
                raise ValueError("All keys must be selected.")
            
            freq = data[freq_key]
            amp = data[amp_key]
            time = data[time_key]
            
            time_pro_min_str = self.time_px_min_edit.text()
            time_pro_max_str = self.time_px_max_edit.text()
            
            time_pro_min = float(time_pro_min_str) if time_pro_min_str.strip() else None
            time_pro_max = float(time_pro_max_str) if time_pro_max_str.strip() else None
            
            # Assume amp is (time, freq)
            if len(amp.shape) != 2 or amp.shape[0] != len(time) or amp.shape[1] != len(freq):
                raise ValueError("Amplitude shape mismatch.")
            
            # Clear the entire figure and recreate the subplot
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self.colorbar = None  # Reset colorbar reference
            
            # Use imshow for faster rendering
            img = self.ax.imshow(amp, aspect='auto', origin='lower', extent=[freq.min(), freq.max(), time.min(), time.max()], cmap='viridis')
            
            # Add colorbar
            self.colorbar = self.figure.colorbar(img, ax=self.ax)
            
            # Add horizontal lines for projections
            if time_pro_min is not None:
                self.ax.axhline(y=time_pro_min, color='red', linestyle='--', label='Min Projection')
            if time_pro_max is not None:
                self.ax.axhline(y=time_pro_max, color='red', linestyle='--', label='Max Projection')
            
            self.ax.set_xlabel('Frequency (MHz)')
            self.ax.set_ylabel('Time (s)')
            self.ax.set_title('Low-Mem Combined Spectrogram (avg 1)')
            self.ax.legend()
            
            # Redraw the canvas
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"An error occurred during preview: {str(e)}")
    
    def get_params(self):
        return {
            'frequency': self.freq_combo.currentText(),
            'amplitude': self.amp_combo.currentText(),
            'time': self.time_combo.currentText(),
            'time_px_min': self.time_px_min_edit.text(),
            'time_px_max': self.time_px_max_edit.text()
        }
        
def read_tdsm_bin(path):
    base_path, _ = os.path.splitext(path)
    bin_fre_path = os.path.join(base_path + '.bin_fre')
    bin_time_path = os.path.join(base_path + '.bin_time')
    bin_amp_path = os.path.join(base_path + '.bin_amp')

    try:
        fre = np.fromfile(bin_fre_path, dtype=np.float64)
        time = np.fromfile(bin_time_path, dtype=np.float32)
        #amp = np.fromfile(bin_amp_path, dtype=np.float32)
        amp = np.memmap(bin_amp_path, dtype=np.float32, mode='r', shape=(len(time), len(fre)))
    except IOError as e:
        raise Exception(f"Error reading files: {e}")
    
    if len(time) == 0 or len(fre) == 0:
        raise ValueError("Time or frequency data files are empty")
    
    try:
        amp = amp.reshape((len(time), len(fre)))
    except ValueError as e:
        raise ValueError(f"Amplitude data cannot be reshaped: {e}")

    midpoint = len(fre) // 2
    frequency = np.concatenate((fre[midpoint:], fre[:midpoint]))
    amplitude = np.concatenate((amp[:, midpoint:], amp[:, :midpoint]), axis=1)

    return frequency, time, amplitude

def handle_read_tdsm_bin(path):
    frequency, _, amplitude = read_tdsm_bin(path)
    amplitude_avg = np.average(amplitude, axis = 0)
    return frequency, amplitude_avg

def handle_read_rsa_specan_xml(filename):
    #frequency, amplitude, _ = read_rsa_specan_xml(filename)
    freq, power, _ = read_rsa_specan_xml(self.filename)
    power = power - power.min() #in order to avoid negative values: power - (-|value_min|) > #power
    #normalized_power = power / power.max()
    return freq, power
    #return frequency, amplitude

def handle_read_rsa_data_csv(filename):
    data = read_rsa_data_csv(filename)
    #obtain frequency, power from data
    return data

def handle_read_rsa_result_csv(filename):
    frequency, amplitude = read_rsa_result_csv(filename)
    return frequency, amplitude

def handle_tiqnpz_data(filename, parent=None):
    data = np.load(filename)
    keys = list(data.keys())
    dialog = KeySelectionDialog(parent, keys=keys, filename=filename)
    if dialog.exec_() == QDialog.Accepted:
        params = dialog.get_params()
        frequency_key = params['frequency']
        amplitude_key = params['amplitude']
        time_key = params['time']
        time_pro_min_str = params['time_px_min']
        time_pro_max_str = params['time_px_max']
        
        # Convert min and max strings to floats or None
        time_pro_min = float(time_pro_min_str) if time_pro_min_str.strip() else None
        time_pro_max = float(time_pro_max_str) if time_pro_max_str.strip() else None
        
        frequency = data[frequency_key].flatten()
        amplitude = data[amplitude_key]
        time = data[time_key]
        
        # Find slice indices based on time values
        min_idx = None
        max_idx = None
        if time_pro_min is not None:
            min_idx = np.searchsorted(time, time_pro_min, side='left')
        if time_pro_max is not None:
            max_idx = np.searchsorted(time, time_pro_max, side='right')
        
        # Handle the slicing, allowing None for open-ended slices
        sliced_amplitude = amplitude[min_idx:max_idx, :]
        amplitude_average = np.average(sliced_amplitude, axis=0)
        return frequency, amplitude_average
    else:
        return None, None

def handle_spectrumnpz_data(filename, parent=None):
    data = np.load(filename)
    keys = list(data.keys())
    dialog = KeySelectionDialog(parent, keys=keys)
    if dialog.exec_() == QDialog.Accepted:
        params = dialog.get_params()
        frequency_key = params['frequency']
        amplitude_key = params['amplitude']
        frequency = data[frequency_key].flatten()
        amplitude = data[amplitude_key]
        return frequency, amplitude
    else:
        return None, None
        
def handle_root_data(filename, y1, y2, h2name="h2_baseline_removed"):
    # 打开 ROOT 文件
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        raise IOError(f"Cannot open file: {filename}")

    h2 = f.Get(h2name)
    if not h2:
        raise ValueError(f"Histogram '{h2name}' not found in file.")

    # 获取 y 的 bin index 范围
    y_bin_min = h2.GetYaxis().FindBin(y1)
    y_bin_max = h2.GetYaxis().FindBin(y2)

    # 向 x 投影，限制 y 的范围
    h_proj = h2.ProjectionX("_px", y_bin_min, y_bin_max)

    # 提取 bin 中心（频率）和内容（amplitude）
    nbins = h_proj.GetNbinsX()
    frequency = np.array([h_proj.GetBinCenter(i+1) for i in range(nbins)])
    amplitude = np.array([h_proj.GetBinContent(i+1) for i in range(nbins)])

    f.Close()
    return frequency*1e6, amplitude
    
def handle_prerionidnpz_data(filename):
    data = np.load(filename)
    frequency = data['x']
    amplitude = data['y']
    return frequency, amplitude

def read_psdata(filename, dbm = False):
    if dbm: 
        frequency, amplitude = np.genfromtxt(filename, skip_header = 1, delimiter='|', usecols = (0,2))
    else: 
        frequency, amplitude = np.genfromtxt(filename, skip_header = 1, delimiter='|', usecols = (0,1))

    return frequency, amplitude

def write_arrays_to_ods(file_name, sheet_name, names, *arrays):
    # Create the ods spreadsheet and add a sheet
    spreadsheet = ezodf.newdoc(doctype='ods', filename=file_name)
    max_len = max(len(arr) for arr in arrays)
    sheet = ezodf.Sheet(sheet_name,size=(max_len+1,len(arrays)))
    spreadsheet.sheets += sheet
    
    for i, arr in enumerate(arrays):
        sheet[(0, i)].set_value(str(names[i]))
        for j in range(len(arr)):
            sheet[j+1, i].set_value(arr[j])

    # Save the spreadsheet
    spreadsheet.save()
