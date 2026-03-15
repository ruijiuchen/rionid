from iqtools.tools import read_rsa_specan_xml, read_rsa_data_csv, read_rsa_result_csv
import numpy as np
import os
import ROOT
from PyQt5.QtWidgets import QDialog, QFormLayout, QComboBox, QHBoxLayout, QPushButton,QLineEdit,QMessageBox,QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#import ROOT
#import numpy as np
#from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QPushButton, QLineEdit, QComboBox, QMessageBox
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#import matplotlib.pyplot as plt

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
            if 'arr_2' in keys:
                self.amp_combo.setCurrentText('arr_2')
            if 'arr_1' in keys:
                self.time_combo.setCurrentText('arr_1')
        
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
            
            print('总共有', len(data.files), '个 key')
            print('每个 key 的名称和维度：')
            for k in sorted(data.files):
                print(f'  {k:20}  {data[k].shape}')
                
            freq_key = self.freq_combo.currentText()
            amp_key = self.amp_combo.currentText()
            time_key = self.time_combo.currentText()
            if not all([freq_key, amp_key, time_key]):
                raise ValueError("All keys must be selected.")
    
            freq = data[freq_key]/1e6
            amp = data[amp_key]
            time = data[time_key][:-1]
    
            # 调试输出（上线前可删除）
            print(f"amp shape: {amp.shape}, ndim: {amp.ndim}")
            print(f"freq length: {len(freq)}, time length: {len(time)}")
    
            time_pro_min_str = self.time_px_min_edit.text()
            time_pro_max_str = self.time_px_max_edit.text()
    
            time_pro_min = float(time_pro_min_str) if time_pro_min_str.strip() else None
            time_pro_max = float(time_pro_max_str) if time_pro_max_str.strip() else None
    
            # ─── 判断绘图类型 ───────────────────────────────────────────
            if amp.ndim == 1:
                # 1D 谱图：频率 vs 幅度 → 用折线图
                plot_up_type = "line"
                plot_down_type = "line"
                # 检查 freq 和 amp 长度是否匹配
                if len(amp) != len(freq):
                    raise ValueError(f"Length mismatch: amp({len(amp)}) vs freq({len(freq)})")
    
                full_y = amp
                proj_y = amp
                proj_x = freq  # 下图也用相同 x
    
            elif amp.ndim == 2:
                # 2D 时频谱图
                plot_up_type = "imshow"
                plot_down_type = "line"
                if amp.shape[0] != len(time) or amp.shape[1] != len(freq):
                    raise ValueError(
                        f"2D shape mismatch: expected ({len(time)}, {len(freq)}), got {amp.shape}"
                    )
                full_img = amp
    
                # 准备投影区域（x轴投影：沿时间求和）
                if time_pro_min is not None and time_pro_max is not None:
                    mask = (time >= time_pro_min) & (time <= time_pro_max)
                    if not np.any(mask):
                        raise ValueError("No points in selected time projection range")
                    
                    # x轴投影：沿时间维度（axis=0）求和
                    proj_img = np.sum(full_img[mask, :], axis=0)          # 结果 shape: (len(freq),)
                    proj_y = proj_img
                    proj_x = freq  # 下图也用相同 x
                    proj_time = time[mask]
                else:
                    # 无范围时，也对全图做投影
                    proj_img = np.sum(full_img, axis=0)
                    proj_y = proj_img
                    proj_x = freq  # 下图也用相同 x
                    proj_time = time
            else:
                raise ValueError(f"Unsupported ndim: {amp.ndim} (expected 1 or 2)")
    
            # ─── 绘图 ─────────────────────────────────────────────────────
            self.figure.clf()
            axs = self.figure.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]})
            upper_ax = axs[0]
            lower_ax = axs[1]
    
            # 上图：完整视图
            if plot_up_type == "line":
                upper_ax.plot(freq, full_y, color='blue', lw=1.0)
                upper_ax.set_xlabel('Frequency (Hz)')
                upper_ax.set_ylabel('Amplitude')
                upper_ax.set_title('Full Spectrum')
                upper_ax.grid(True, alpha=0.3)
    
            elif plot_up_type == "imshow":
                img_u = upper_ax.imshow(
                    full_img,
                    aspect='auto',
                    origin='lower',
                    extent=[freq.min(), freq.max(), time.min(), time.max()],
                    cmap='viridis'
                )
                self.figure.colorbar(img_u, ax=upper_ax, label='Intensity')
                upper_ax.set_ylabel('Time (s)')
                upper_ax.set_title('Full Spectrogram')
    
                # 画投影范围线
                if time_pro_min is not None:
                    upper_ax.axhline(time_pro_min, color='red', ls='--', lw=1.5)
                if time_pro_max is not None:
                    upper_ax.axhline(time_pro_max, color='red', ls='--', lw=1.5)
    
            # 下图：投影或相同内容
            if plot_down_type == "line":
                lower_ax.plot(proj_x, proj_y, color='darkgreen', lw=1.2)
                lower_ax.set_xlabel('Frequency (MHz)')
                lower_ax.set_ylabel('Amplitude')
                lower_ax.set_title('Spectrum (same as above)')
                lower_ax.grid(True, alpha=0.3)
    
            elif plot_down_type == "imshow":
                img_l = lower_ax.imshow(
                    proj_img,
                    aspect='auto',
                    origin='lower',
                    extent=[freq.min(), freq.max(), proj_time.min(), proj_time.max()],
                    cmap='viridis'
                )
                self.figure.colorbar(img_l, ax=lower_ax, label='Intensity')
                lower_ax.set_xlabel('Frequency (MHz)')
                lower_ax.set_ylabel('Time (s)')
                lower_ax.set_title(
                    'Projection Region' if (time_pro_min and time_pro_max) else 'Full Spectrogram'
                )
    
            self.figure.tight_layout()
            self.canvas.draw()
    
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Preview Error", f"Error during preview:\n{str(e)}")
    
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
    dialog = KeySelectionDialog(parent, keys=keys, filename = filename)
    if dialog.exec_() == QDialog.Accepted:
        params = dialog.get_params()
        frequency_key = params['frequency']
        amplitude_key = params['amplitude']
        frequency = data[frequency_key].flatten()
        amplitude = data[amplitude_key]
        return frequency, amplitude
    else:
        return None, None
        

class RootInputDialog(QDialog):
    def __init__(self, parent=None, filename=None):
        super().__init__(parent)
        self.setWindowTitle("Select Parameters for ROOT File Processing")
        self.filename = filename
        self.colorbar = None  # To track the colorbar

        # Set Matplotlib backend to Qt5Agg
        plt.switch_backend('Qt5Agg')

        # Main layout
        main_layout = QVBoxLayout()

        # Form layout for inputs
        form_layout = QFormLayout()

        self.h2name_combo = QComboBox()
        self.y1_edit = QLineEdit()
        self.y2_edit = QLineEdit()

        # Try to load histogram names from the ROOT file
        keys = self.get_histogram_keys()
        if keys:
            self.h2name_combo.addItems(keys)
            if 'h2_baseline_removed' in keys:
                self.h2name_combo.setCurrentText('h2_baseline_removed')

        form_layout.addRow("Histogram Name:", self.h2name_combo)
        form_layout.addRow("Y1 (min):", self.y1_edit)
        form_layout.addRow("Y2 (max):", self.y2_edit)

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

        # Matplotlib canvas for preview
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Combine layouts
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(buttons)
        self.setLayout(main_layout)

    def get_histogram_keys(self):
        """Retrieve available histogram names from the ROOT file."""
        if not self.filename:
            return []
        try:
            f = ROOT.TFile.Open(self.filename)
            if not f or f.IsZombie():
                return []
            keys = [key.GetName() for key in f.GetListOfKeys() if key.GetClassName().startswith('TH')]
            f.Close()
            return keys
        except:
            return []

    def preview_plot(self):
        """Generate a preview plot of the projected histogram."""
        if not self.filename:
            QMessageBox.warning(self, "Error", "No filename provided for preview.")
            return

        try:
            h2name = self.h2name_combo.currentText()
            y1_str = self.y1_edit.text()
            y2_str = self.y2_edit.text()

            if not h2name:
                raise ValueError("Histogram name must be selected.")
            if not y1_str.strip() or not y2_str.strip():
                raise ValueError("Y1 and Y2 must be provided.")

            y1 = float(y1_str)
            y2 = float(y2_str)

            # Open ROOT file and get histogram
            f = ROOT.TFile.Open(self.filename)
            if not f or f.IsZombie():
                raise IOError(f"Cannot open file: {self.filename}")

            h2 = f.Get(h2name)
            if not h2:
                raise ValueError(f"Histogram '{h2name}' not found in file.")

            # Get y bin index range
            y_bin_min = h2.GetYaxis().FindBin(y1)
            y_bin_max = h2.GetYaxis().FindBin(y2)

            # Project to x, restricting y range
            h_proj = h2.ProjectionX("_px", y_bin_min, y_bin_max)

            # Extract bin centers (frequency) and contents (amplitude)
            nbins = h_proj.GetNbinsX()
            frequency = np.array([h_proj.GetBinCenter(i+1) for i in range(nbins)]) * 1e6  # Convert to MHz
            amplitude = np.array([h_proj.GetBinContent(i+1) for i in range(nbins)])

            # Clear the entire figure and recreate the subplot
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self.colorbar = None  # Reset colorbar reference

            # Plot the histogram projection
            self.ax.plot(frequency, amplitude, label='Projected Histogram')
            self.ax.set_xlabel('Frequency (MHz)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title(f'Histogram Projection ({h2name}, y1={y1}, y2={y2})')
            self.ax.legend()

            # Redraw the canvas
            self.canvas.draw()

            f.Close()
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"An error occurred during preview: {str(e)}")

    def get_params(self):
        """Return the selected parameters."""
        return {
            'h2name': self.h2name_combo.currentText(),
            'y1': self.y1_edit.text(),
            'y2': self.y2_edit.text()
        }

def handle_root_data(filename,y1, y2, h2name):
    # Create and show the dialog
    dialog = RootInputDialog(filename=filename)
    if dialog.exec_() != QDialog.Accepted:
        raise ValueError("Dialog cancelled by user.")

    params = dialog.get_params()
    try:
        y1 = float(params['y1']) if params['y1'].strip() else None
        y2 = float(params['y2']) if params['y2'].strip() else None
        h2name = params['h2name'] or "h2_baseline_removed"
    except ValueError:
        raise ValueError("Invalid input for y1 or y2. Please enter valid numbers.")

    if y1 is None or y2 is None:
        raise ValueError("Y1 and Y2 must be provided.")

    # Open ROOT file
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        raise IOError(f"Cannot open file: {filename}")

    h2 = f.Get(h2name)
    if not h2:
        raise ValueError(f"Histogram '{h2name}' not found in file.")

    # Get y bin index range
    y_bin_min = h2.GetYaxis().FindBin(y1)
    y_bin_max = h2.GetYaxis().FindBin(y2)

    # Project to x, restricting y range
    h_proj = h2.ProjectionX("_px", y_bin_min, y_bin_max)

    # Extract bin centers (frequency) and contents (amplitude)
    nbins = h_proj.GetNbinsX()
    frequency = np.array([h_proj.GetBinCenter(i+1) for i in range(nbins)])
    amplitude = np.array([h_proj.GetBinContent(i+1) for i in range(nbins)])

    f.Close()
    return frequency * 1e6, amplitude
    
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
