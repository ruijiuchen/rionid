import numpy as np
from numpy import polyval, array, stack, append, sqrt, genfromtxt
import os
import sys
import re

from barion.ring import Ring
from barion.amedata import AMEData
from barion.particle import Particle

from lisereader.reader import LISEreader

from rionid.inouttools import * 
from scipy.signal import find_peaks, peak_widths
import traceback
from scipy.ndimage import gaussian_filter1d  # or use savgol_filter
from scipy.signal import savgol_filter
import re
# add 2025/7/16 15:21
from .nonparams_est import NONPARAMS_EST

class ImportData(object):
    '''
    Model (MVC)
    '''
    def __init__(self, refion, highlight_ions, remove_baseline, psd_baseline_removed_l, psd_baseline_removed_ratio, alphap, filename = None, reload_data = None, circumference = None, peak_threshold_pct=None,min_distance=None,matching_freq_min=None,matching_freq_max=None, ref_harmonic=None, hist_freq_min=None, hist_freq_max=None, hist_bins=None, skip_peak_detection=False):
        self.simulated_data_dict = {}  # Make sure this is initialized
        # Argparser arguments
        self.harmonics=[]
        self.particles_to_simulate = []  # Default to an empty list
        self.protons = dict()
        self.moq = dict()
        self.total_mass = dict()  # Initialize the total mass dictionary
        self.ref_ion = refion
        self.highlight_ions = highlight_ions
        self.alphap = alphap
        self.gammat = (1/(alphap))**0.5
        # Extra objects
        self.ring = Ring('ESR', circumference)
        self.ref_aa, self.ref_element, self.ref_charge = self._parse_ion_name(refion)
        self.experimental_data = None
        self.brho = 0
        self.peak_threshold_pct = float(peak_threshold_pct)
        self.peak_freqs = []
        self.peak_widths_freq = []
        # 柱状图参数（供 load_peaks_summary 使用）
        self.hist_freq_min = hist_freq_min
        self.hist_freq_max = hist_freq_max
        self.hist_bins = hist_bins
        self.skip_peak_detection = skip_peak_detection
        self.peak_heights = []
        self.gammats = []
        self.yield_data=[]
        self.filename = filename
        # Data cache file path
        self.cache_file = self._get_cache_file_path(filename)
        self.threshold_profile_path = self._get_threshold_profile_path(filename)
        self.threshold_profile_freqs = None
        self.threshold_profile_vals = None
        self._load_threshold_profile(self.threshold_profile_path)
        self.ref_harmonic = ref_harmonic  # harmonic number of the reference frequency
        self.chi2= 0
        self.match_count=0
        self.min_distance=min_distance
        self.matching_freq_min=matching_freq_min
        self.matching_freq_max=matching_freq_max
        self.remove_baseline = remove_baseline
        self.psd_baseline_removed = None
        self.psd_baseline = None
        self.psd_baseline_removed_l=psd_baseline_removed_l
        self.psd_baseline_removed_ratio=psd_baseline_removed_ratio
        self.ref_frequency=0
        self._baseline_applied = False
        # Get the experimental data
        print("chenrj ... ImportData 1")
        if filename is not None:
            if filename.lower().endswith('.txt'):
                # 峰汇总文件：没有有意义的缓存，始终直接读取
                print("chenrj ... ImportData 1a (.txt)")
                self._get_experimental_data(filename)
            elif reload_data:
                print("chenrj ... ImportData 1a")
                self._get_experimental_data(filename)
                self._save_experimental_data()
            else:
                print("chenrj ... ImportData 1b")
                self._load_experimental_data()
        else:
            print("No experimental data file provided. Using default or simulated data.")
            self.experimental_data = None  # Set empty or simulated data here
        print("chenrj ... ImportData 2")
    def _parse_ion_name(self, refion):
        """
        Parse an ion name to extract mass number, element symbol, and charge.

        Args:
            refion (str): Ion name in the format '<mass><element><charge>+', e.g., '205Tl81+'

        Returns:
            tuple: (mass_number: int, element_symbol: str, charge: int)

        Raises:
            ValueError: if the ion name format is invalid
        """
        match = re.match(r'^(\d+)([A-Za-z]+)(\d+)\+$', refion)
        if not match:
            raise ValueError(f"Invalid ion name format: {refion}")
        return int(match.group(1)), match.group(2), int(match.group(3))
        
    def compute_matches(self,
                        match_threshold,
                        match_frequency_min=None,
                        match_frequency_max=None,
                        sim_items=None,
                        verbose=True):
        """
        Match the experimental peaks in self.peak_freqs against the simulated spectrum,
        but only for exp_freq in [match_frequency_min, match_frequency_max] if those are set.
    
        Args:
            match_threshold (float): maximum allowed |sim_freq – exp_freq|
            match_frequency_min (float, optional): lowest exp_freq to consider
            match_frequency_max (float, optional): highest exp_freq to consider
    
        Returns:
            tuple: (chi2, match_count, self.highlight_ions)
        """
        # Build list of (frequency, ion_name, harmonic, yield) from simulated_data_dict or external sim_items
        if sim_items is None:
            sim_items = []
            for harmonic_name, sdata in self.simulated_data_dict.items():
                harmonic = int(float(harmonic_name))
                for row in sdata:
                    sim_items.append((float(row[0]), row[2], harmonic, float(row[1])))  # 添加 harmonic_name 和产额
        sim_freqs = np.array([freq for freq, _, _, _ in sim_items])
        # ── 调试：输出频率范围和阈值 ──
        if verbose and len(self.peak_freqs) > 0 and len(sim_freqs) > 0:
            print(f"\n🔍 匹配调试:")
            print(f"   实验峰频率范围: {self.peak_freqs.min()/1e6:.4f} – {self.peak_freqs.max()/1e6:.4f} MHz")
            print(f"   模拟谱频率范围: {sim_freqs.min()/1e6:.4f} – {sim_freqs.max()/1e6:.4f} MHz")
            print(f"   匹配阈值: {match_threshold} Hz = {match_threshold/1e6:.6f} MHz")
            print(f"   实验峰数: {len(self.peak_freqs)}, 模拟峰数: {len(sim_freqs)}")
            print(f"   前5个实验峰 (MHz): {[f/1e6 for f in self.peak_freqs[:5]]}")
            print(f"   前5个模拟峰 (MHz): {[f/1e6 for f in sim_freqs[:5]]}")
        # Initialize accumulators
        chi2 = 0.0
        match_count = 0
        total_weight = 0.0
        matched_ions        = []
        matched_sim_items   = []
        matched_sim_freqs   = []
        matched_exp_freqs   = []
        matched_peak_widths = []
        matched_peak_heights= []
    
        # Loop over each experimental peak
        for exp_freq, width, height in zip(self.peak_freqs,
                                           self.peak_widths_freq,
                                           self.peak_heights):
            # Skip peaks outside the desired frequency window
            if match_frequency_min is not None and exp_freq < match_frequency_min:
                continue
            if match_frequency_max is not None and exp_freq > match_frequency_max:
                continue
    
            # Find the closest simulated frequency
            idx  = np.argmin(np.abs(sim_freqs - exp_freq))
            diff = abs(sim_freqs[idx] - exp_freq)
            if diff <= match_threshold:
                weight = max(1e-12, sim_items[idx][3])
                chi2 += weight * diff**2
                total_weight += weight
                match_count += 1
                matched_ions.append(sim_items[idx][1])
                matched_sim_items.append(sim_items[idx])
                matched_sim_freqs.append(sim_freqs[idx])
                matched_exp_freqs.append(exp_freq)
                matched_peak_widths.append(width)
                matched_peak_heights.append(height)
    
        # Finalize chi² using yield-weighted average
        chi2 = chi2 / total_weight if total_weight > 0 else float('inf')
    
        # Deduplicate and filter out the reference ion
        unique_ions   = sorted(set(matched_ions))
        filtered_ions = [ion for ion in unique_ions if ion != self.ref_ion]
    
        # Store results on self
        self.chi2                = chi2
        self.match_count         = match_count
        self.highlight_ions      = filtered_ions
        self.matched_ions        = matched_ions
        self.matched_sim_items   = matched_sim_items
        self.matched_sim_freqs   = matched_sim_freqs
        self.matched_exp_freqs   = matched_exp_freqs
        self.matched_peak_widths = matched_peak_widths
        self.matched_peak_heights= matched_peak_heights
    
        return chi2, match_count, filtered_ions


    def scan_match(self, f_ref, alphap, harmonics, match_threshold,
                   match_frequency_min=None, match_frequency_max=None,
                   mode='Frequency', ref_harmonic=None):
        """
        Lightweight scan for a single (f_ref, alphap) combination.
        Skips the full _simulated_data overhead — reuses self.moq, self.yield_data,
        self.nuclei_names which must already be populated (via a prior
        _simulated_data call or _build_particle_cache).

        Returns:
            tuple: (chi2, match_count, filtered_ions)
        """
        # Handle gammat conversion
        alphap_val = float(alphap)
        if alphap_val > 1:
            alphap_val = 1 / alphap_val ** 2

        # Update parameters and recalculate srrf (the only alphap-dependent part)
        self.ref_frequency = f_ref
        self.alphap = alphap_val
        self.gammat = (1 / alphap_val) ** 0.5
        self._calculate_srrf(fref=f_ref, correct=False)

        # Build sim_items directly from srrf × harmonic (no yield_data rebuild)
        sim_items = []
        for harmonic in harmonics:
            h = float(harmonic)
            if mode == 'Frequency':
                if ref_harmonic is not None and ref_harmonic != 0:
                    f0 = f_ref / ref_harmonic
                    harmonic_freqs = self.srrf * f0 * h
                else:
                    harmonic_freqs = self.srrf * f_ref
            else:
                # Bρ mode — not optimised for scan, fall back to _simulated_data path
                return self.compute_matches(match_threshold,
                                            match_frequency_min,
                                            match_frequency_max)

            for i, ion_name in enumerate(self.nuclei_names):
                sim_items.append((harmonic_freqs[i], ion_name,
                                  h, self.yield_data[i]))

        # Delegate to the same matching logic
        return self.compute_matches(
            match_threshold,
            match_frequency_min,
            match_frequency_max,
            sim_items=sim_items)

    def scan_match_brho(self, brho, circumference, harmonics, match_threshold,
                        match_frequency_min=None, match_frequency_max=None,
                        verbose=True):
        """
        Lightweight scan for a single (brho, circumference) combination in Br mode.
        Reuses self.moq, self.yield_data, self.nuclei_names from a prior
        _simulated_data call.  Updates self.ring.circumference and self.brho,
        then computes per-ion frequencies using the Br formula.

        Returns:
            tuple: (chi2, match_count, filtered_ions)
        """
        self.brho = brho
        self.ring.circumference = circumference

        moq_keys = list(self.moq.keys())
        harmonic_freq_list = []
        for harmonic in harmonics:
            h = float(harmonic)
            for ion_name in moq_keys:
                try:
                    _A, elem, q = self._parse_ion_name(ion_name)
                    mass_u = AMEData.to_mev(self.moq[ion_name] * q)
                    f_rev = ImportData.calc_ref_rev_frequency(
                        ref_mass=mass_u,
                        ring_circumference=self.ring.circumference,
                        brho=brho,
                        ref_charge=q
                    )
                    this_harmonic_freq = h * f_rev
                    harmonic_freq_list.append(this_harmonic_freq)
                except Exception:
                    continue

        # Build sim_items
        n_ions = len(moq_keys)
        n_harmonics = len(harmonics)
        tiled_yield = np.tile(self.yield_data, n_harmonics)
        tiled_names = np.tile(self.nuclei_names, n_harmonics)

        sim_items = []
        for idx in range(len(harmonic_freq_list)):
            sim_items.append((
                harmonic_freq_list[idx],
                tiled_names[idx],
                float(harmonics[idx // n_ions]),
                tiled_yield[idx]
            ))

        # Delegate to the same matching logic
        return self.compute_matches(
            match_threshold,
            match_frequency_min,
            match_frequency_max,
            sim_items=sim_items,
            verbose=verbose)

    def save_matched_result(self, output_file='best_match_details.csv'):
        """
        1) Compute γₜ for each matched ion by pairing it with its nearest‐frequency neighbor.
        2) Write all matched data plus computed γₜ into a CSV.
        Returns:
            list: self.gammats
        """
        # 1) Initialize gamma list
        self.gammats = []
    
        # 2) Prepare arrays for neighbor search
        ions      = self.matched_ions
        exp_freqs = np.array(self.matched_exp_freqs)
        moqs      = np.array([self.moq[ion] for ion in ions])
    
        # 3) For each ion, find its closest-frequency neighbor and compute gamma_t
        for i, (ion_i, f_i, moq_i) in enumerate(zip(ions, exp_freqs, moqs)):
            # Compute abs differences, ignore self
            diffs = np.abs(exp_freqs - f_i)
            diffs[i] = np.inf
            j        = np.argmin(diffs)
            f_ref    = exp_freqs[j]
            moq_ref  = moqs[j]
    
            # Formula:
            #   –(f_i – f_ref)/f_ref = (1/γ_t²) * ((moq_i – moq_ref)/moq_ref)
            num   = abs(moq_i   - moq_ref) / moq_ref
            denom = abs(f_i     - f_ref)   / f_ref
            gamma_t = np.sqrt(num/denom) if num>0 and denom>0 else np.nan
    
            self.gammats.append(gamma_t)
    
        # 4) Write CSV including gamma_t column
        with open(output_file, 'w', newline='') as f:
            f.write('ion_name,sim_freq[Hz],exp_freq[Hz],'
                    'peak_width[Hz],peak_height,'
                    'm/q,gamma_t,HN,RevT_sim,RevT_exp,RevT_exp_sim,sigma_RevT,Nuclei,Z,Q,Flag,T(ps),Count,SigmaT(ps),TError(ps)\n')
            for ion, sim_f, exp_f, w, h, gt, sim_item  in zip(
                self.matched_ions,
                self.matched_sim_freqs,
                self.matched_exp_freqs,
                self.matched_peak_widths,
                self.matched_peak_heights,
                self.gammats,
                self.matched_sim_items
            ):
                moq = self.moq[ion]
                HN = sim_item[2] 
                RevT_sim = 1e12/(sim_f/HN)
                RevT_exp = 1e12/(exp_f/HN)
                RevT_exp_sim = RevT_exp - RevT_sim
                sigma_RevT = w/2.35/exp_f * RevT_exp
                match = re.match(r'(\d+)([A-Za-z]+)(\d+)\+', ion)
                proton = self.protons[ion]
                if match:
                    mass, elem, charge = match.groups()
                Flag = "Y"
                f.write(f"{ion},{sim_f:.2f},{exp_f:.2f},"
                        f"{w:.2f},{h:.6f},"
                        f"{moq:.12f},{gt:.6f},{HN:.0f},{RevT_sim:.2f},{RevT_exp:.2f},{RevT_exp_sim:.2f},{sigma_RevT:.2f},{mass}{elem},{proton:.0f},{float(charge):.0f},{Flag},{RevT_exp:.2f},{h:.6f},{sigma_RevT:.2f},{sigma_RevT/sqrt(h):.2f}\n")
    
        print(f"Detailed match data saved to '{output_file}'")
        return self.gammats

    def _get_parameters_cache_dir(self):
        candidate_names = ["parameters_cache.toml", "parameters_cache"]
        search_roots = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]

        for root in search_roots:
            current = root
            while True:
                for name in candidate_names:
                    candidate = os.path.join(current, name)
                    if os.path.exists(candidate):
                        return current
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent

        return os.getcwd()

    def _get_threshold_profile_path(self, filename=None):
        if filename is None:
            return None
        base, _ = os.path.splitext(os.path.basename(filename))
        cache_dir = self._get_parameters_cache_dir()
        return os.path.join(cache_dir, f"{base}_height_thresh.csv")

    def _load_threshold_profile(self, threshold_path=None):
        profile_path = threshold_path or self.threshold_profile_path
        self.threshold_profile_path = profile_path
        if not profile_path or not os.path.exists(profile_path):
            self.threshold_profile_freqs = None
            self.threshold_profile_vals = None
            return False

        try:
            data = np.genfromtxt(profile_path, delimiter=',', comments='#', dtype=float)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.size == 0 or data.shape[1] < 2:
                raise ValueError("Threshold profile must have at least two columns: freq, threshold")

            freqs = np.asarray(data[:, 0], dtype=float)
            vals = np.asarray(data[:, 1], dtype=float)
            valid = np.isfinite(freqs) & np.isfinite(vals)
            freqs = freqs[valid]
            vals = vals[valid]
            if freqs.size == 0:
                raise ValueError("Threshold profile contains no valid data")

            order = np.argsort(freqs)
            self.threshold_profile_freqs = freqs[order]
            self.threshold_profile_vals = np.maximum(vals[order], 0.0)
            print(f"Loaded threshold profile from {profile_path} with {len(self.threshold_profile_freqs)} points")
            return True
        except Exception as exc:
            print(f"Warning: failed to load threshold profile from {profile_path}: {exc}")
            self.threshold_profile_freqs = None
            self.threshold_profile_vals = None
            return False

    def _save_threshold_profile(self, freqs=None, vals=None):
        if not self.threshold_profile_path:
            return None

        if freqs is None:
            freqs = self.threshold_profile_freqs
        if vals is None:
            vals = self.threshold_profile_vals
        if freqs is None or vals is None or len(freqs) == 0 or len(vals) == 0:
            return None

        freqs = np.asarray(freqs, dtype=float)
        vals = np.asarray(vals, dtype=float)
        order = np.argsort(freqs)
        freqs = freqs[order]
        vals = np.maximum(vals[order], 0.0)

        directory = os.path.dirname(self.threshold_profile_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        np.savetxt(
            self.threshold_profile_path,
            np.column_stack([freqs, vals]),
            delimiter=',',
            header='freq_hz,threshold',
            comments=''
        )
        self.threshold_profile_freqs = freqs
        self.threshold_profile_vals = vals
        return self.threshold_profile_path

    def update_threshold_profile_from_clicks(self, freq_hz, threshold_value):
        if self.threshold_profile_path is None:
            self.threshold_profile_path = self._get_threshold_profile_path(self.filename)

        if self.threshold_profile_freqs is None or self.threshold_profile_vals is None:
            self.threshold_profile_freqs = np.array([float(freq_hz)], dtype=float)
            self.threshold_profile_vals = np.array([float(threshold_value)], dtype=float)
        else:
            matches = np.isclose(self.threshold_profile_freqs, float(freq_hz), rtol=1e-6, atol=1.0)
            if np.any(matches):
                idx = np.where(matches)[0][0]
                self.threshold_profile_freqs[idx] = float(freq_hz)
                self.threshold_profile_vals[idx] = float(threshold_value)
            else:
                self.threshold_profile_freqs = np.append(self.threshold_profile_freqs, float(freq_hz))
                self.threshold_profile_vals = np.append(self.threshold_profile_vals, float(threshold_value))

        self._save_threshold_profile(self.threshold_profile_freqs, self.threshold_profile_vals)
        print(f"Updated threshold profile at {freq_hz:.6g} Hz with value {threshold_value:.6g}")
        return self.threshold_profile_path

    def _get_cache_file_path(self, filename):
        base, _ = os.path.splitext(filename)
        return f"{base}_cache.npz"
    
    def _get_experimental_data(self, filename):
        print("chenrj ... _get_experimental_data 1")
        base, file_extension = os.path.splitext(filename)
        if file_extension.lower() == '.csv':
            self.experimental_data = read_psdata(filename, dbm = False)
        if file_extension.lower() == '.bin_fre' or file_extension.lower() == '.bin_time' or file_extension.lower() == '.bin_amp':
            self.experimental_data = handle_read_tdsm_bin(filename)
        if file_extension.lower() == '.tdms':
            self.experimental_data = handle_read_tdsm(filename)
            #substitute this
        if file_extension.lower() == '.xml':
            self.experimental_data = handle_read_rsa_specan_xml(filename)
        if file_extension.lower() == '.Specan':
            self.experimental_data = handle_read_rsa_specan_xml(filename)
        if file_extension.lower() == '.root':
            self.experimental_data = handle_root_data(filename, 5,14,h2name="h2_baseline_removed")
        if file_extension.lower() == '.npz':
            if 'spectrum' in base:
                self.experimental_data = handle_spectrumnpz_data(filename)
            else:
                self.experimental_data = handle_tiqnpz_data(filename)

        # ── 峰汇总文件 (.txt)：不包含完整频谱，直接解析预检测的峰列表 ──
        if file_extension.lower() == '.txt':
            print("chenrj ... _get_experimental_data 2")
            print("chenrj ... = filename = ",filename)
            self.load_peaks_summary(filename)
            # experimental_data 保留为 None（无完整频谱），detect_peaks_and_widths 会跳过
            return  # 无需走后续的基线去除 / 寻峰流程

        if self.remove_baseline:
            try:
                if self.experimental_data is not None and len(self.experimental_data) == 2:
                    freq, psd = self.experimental_data
                    
                    print("✅ Baseline removal is working... l= ",self.psd_baseline_removed_l," r= ",self.psd_baseline_removed_ratio)
                    baseline = NONPARAMS_EST(psd).pls('BrPLS', l=self.psd_baseline_removed_l, ratio=self.psd_baseline_removed_ratio)
                    psd_baseline_removed = psd - baseline
                    
                    self.psd_baseline = (freq, baseline)
                    self.psd_baseline_removed = (freq, psd_baseline_removed)
                    print("✅ Baseline removal completed. Result stored in self.psd_baseline_removed.")
                    
                    if self.psd_baseline_removed is not None:
                        freq, psd_new = self.psd_baseline_removed
                        
                        if np.any(np.isnan(psd_new)):
                            raise RuntimeError(
                                "基线去除后 psd_baseline_removed 包含 NaN 值！\n"
                                "这通常意味着：\n"
                                "1. 输入 PSD 数据本身包含大量无效值\n"
                                "2. BrPLS 算法未能产生有效基线（可能参数 l/ratio 不合适）\n"
                                f"当前形状: {psd_new.shape}，NaN 数量: {np.isnan(psd_new).sum()}/{psd_new.size}"
                            )
                        
                        print(f" → 频率点数: {len(freq)}")
                        print(f" → psd_baseline_removed 形状: {psd_new.shape if hasattr(psd_new, 'shape') else '不是数组'}")
                        print(f" → 前5个值: {psd_new[:5]}")
                        print(f" → 后5个值: {psd_new[-5:]}")
                        print(f" → 最小值: {psd_new.min():.6e}, 最大值: {psd_new.max():.6e}, 平均值: {psd_new.mean():.6e}")
                else:
                    print("⚠️ Invalid format of self.experimental_data. Skipping baseline removal.")
                    
            except Exception as e:
                print("❌ Baseline removal failed:", e)
                traceback.print_exc()
                
                # ──────────────── 关键修改：强制终止程序 ────────────────
                print("\n程序因严重错误而终止（基线去除失败）。")
                import sys
                sys.exit(1)   # 退出码 1 表示异常退出

    
        if not self.skip_peak_detection:
            self.detect_peaks_and_widths()

    def detect_peaks_and_widths(self):
        if self.experimental_data is None:
            return

        if self.remove_baseline and self.psd_baseline_removed is not None:
            freq, amp = self.psd_baseline_removed
        else:
            freq, amp = self.experimental_data

        rel_height = max(0.0, min(self.peak_threshold_pct, 1.0))
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
            print(f"Using frequency-dependent threshold profile from {self.threshold_profile_path}")
        else:
            height_thresh = np.full_like(amp, np.max(amp) * rel_height, dtype=float)
            print("Using fallback constant threshold based on peak_threshold_pct")
    
        min_dist    = float(self.min_distance)
        min_prom    = np.maximum(height_thresh * 0.3, 0.0)
        min_w       = 1
        
        peaks, props = find_peaks(
            amp,
            height=height_thresh,
            distance=min_dist,
            prominence=min_prom,
            width=min_w
        )
    
        # 4) measure “true” half-height widths on the smoothed data
        widths, width_heights, left_ips, right_ips = peak_widths(
            amp, peaks, rel_height=0.5
        )
        
        # — apply matching_freq_min / matching_freq_max window —
        if self.matching_freq_min is not None or self.matching_freq_max is not None:
            # build a boolean mask, one entry per peak
            mask = np.ones_like(peaks, dtype=bool)
            # lower bound
            if self.matching_freq_min is not None:
                mask &= (freq[peaks] >= self.matching_freq_min)
            # upper bound
            if self.matching_freq_max is not None:
                mask &= (freq[peaks] <= self.matching_freq_max)
            # apply mask to peaks & width arrays
            peaks        = peaks[mask]
            widths       = widths[mask]
            width_heights= width_heights[mask]
            left_ips     = left_ips[mask]
            right_ips    = right_ips[mask]
        
        # 5) convert to frequency units and store only the filtered peaks
        self.peak_freqs       = freq[peaks]
        self.peak_heights     = amp[peaks]   # raw amplitude
        self.peak_widths_freq = (
            freq[np.round(right_ips).astype(int)]
          - freq[np.round(left_ips) .astype(int)]
        )
    
        # optional: inspect what remains
        print(f"Detected {len(peaks)} peaks after filtering by threshold profile, "
              f"prominence>={np.nanmax(min_prom):.2g}, width>={min_w} samples.")

    def load_peaks_summary(self, filepath):
        """
        载入 all_peaks_summary 格式的峰汇总文件（.txt），
        读取第 3~5 列（频率/峰高/FWHM）到 self.peak_freqs / peak_heights / peak_widths_freq，
        同时将频率分布转换为柱状图存入 self.experimental_data = (bin_centers, counts)。

        文件格式（制表符分隔）:
            总序号\t峰序号\t频率[MHz]\t峰高\tFWHM[MHz]\tstart[s]\tend[s]\t...
        第 3 列 = 频率 (MHz)，第 4 列 = 峰高，第 5 列 = FWHM (MHz)

        柱状图参数（freq_min / freq_max / bins）通过 self 上的同名属性传入，
        可由 GUI 对话框预先设置。
        """
        freq_list = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('总序号') or line.startswith('---'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    freq_list.append(float(parts[2]) * 1e6)   # MHz → Hz
                except (ValueError, IndexError):
                    continue

        raw_freqs = np.array(freq_list)

        # load_peaks_summary 只构建柱状图，不设置峰数据（峰由 Find Peaks 按钮检测）
        # self.peak_freqs / peak_heights / peak_widths_freq 保持 __init__ 中的空列表

        # 从 self 读取柱状图参数（可由 GUI 对话框设置）
        freq_min = getattr(self, 'hist_freq_min', None)
        freq_max = getattr(self, 'hist_freq_max', None)
        bins = self.hist_bins if self.hist_bins is not None else 200

        # 如果设置了范围，过滤频率数据
        if freq_min is not None or freq_max is not None:
            mask = np.ones_like(raw_freqs, dtype=bool)
            if freq_min is not None:
                mask &= (raw_freqs >= freq_min)
            if freq_max is not None:
                mask &= (raw_freqs <= freq_max)
            filtered = raw_freqs[mask]
            print(f"  频率范围过滤: {len(raw_freqs)} → {len(filtered)} 个峰 "
                  f"({freq_min/1e6 if freq_min else 'min':.3f} – {freq_max/1e6 if freq_max else 'max':.3f} MHz)")
            raw_freqs = filtered

        # 构建柱状图：counts per bin
        hist_range = (freq_min, freq_max) if freq_min is not None or freq_max is not None else None
        counts, bin_edges = np.histogram(raw_freqs, bins=bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 存入 experimental_data，格式 (频率, 幅度) 与原有频谱数据兼容
        self.experimental_data = (bin_centers, counts.astype(float))

        print(f"✅ 已从 {filepath} 载入 {len(raw_freqs)} 个峰 → 柱状图 {bins} bins "
              f"(频率范围 {bin_centers.min()/1e6:.3f} – {bin_centers.max()/1e6:.3f} MHz)")

    def _save_experimental_data(self, data=None):
        if data is not None:
            frequency, amplitude_avg = data
            np.savez_compressed(self.cache_file, frequency=frequency, amplitude_avg=amplitude_avg)
        elif self.experimental_data is not None:
            frequency, amplitude_avg = self.experimental_data
            np.savez_compressed(self.cache_file, frequency=frequency, amplitude_avg=amplitude_avg)

    def apply_baseline_once(self):
        """
        Apply baseline removal to the experimental data ONCE and replace
        self.experimental_data with the baseline-removed version.
        The result is saved to the cache file so subsequent loads skip baseline removal.

        Returns:
            tuple (freq, baseline) if successful, None otherwise.
        """
        if self.experimental_data is None:
            return None
        try:
            freq, psd = self.experimental_data
            print("Applying baseline removal once...")
            baseline = NONPARAMS_EST(psd).pls('BrPLS',
                                              l=self.psd_baseline_removed_l,
                                              ratio=self.psd_baseline_removed_ratio)
            psd_removed = psd - baseline

            if np.any(np.isnan(psd_removed)):
                print("Baseline removal produced NaN values — aborting.")
                return None

            # Replace experimental_data with baseline-removed version
            self.experimental_data = (freq, psd_removed)
            self.psd_baseline = (freq, baseline)
            self.psd_baseline_removed = (freq, psd_removed)

            # Save to cache so subsequent loads use baseline-removed data
            self._save_experimental_data()
            # Mark baseline as already applied so GUI knows
            self._baseline_applied = True

            print("Baseline removal applied and cached ONCE. "
                  "Subsequent runs will load baseline-removed data directly.")
            return self.psd_baseline
        except Exception as e:
            print(f"Baseline removal failed: {e}")
            return None
            
    def _load_experimental_data(self):
        if os.path.exists(self.cache_file):
            data = np.load(self.cache_file, allow_pickle=True)
            frequency = data['frequency']
            amplitude_avg = data['amplitude_avg']
            self.experimental_data = (frequency, amplitude_avg)
        else:
            raise FileNotFoundError("Cached data file not found. Please set reload_data to True to generate it.")

    def _set_particles_to_simulate_from_file(self, filep, verbose=None):
        # import ame from barion: # This would be moved somewhere else
        self.ame = AMEData()
        self.ame_data = self.ame.ame_table
        # Read with lise reader  # Extend lise to read not just lise files? 
        lise = LISEreader(filep)
        self.particles_to_simulate = lise.get_info_all(verbose=verbose)
        
    def _calculate_moqs(self, particles = None):
        # Calculate the  moq from barion of the particles present in LISE file or of the particles introduced
        print("chenrj _calculate_moqs  ..." )
        if particles:
             for particle in particles:
                 ion_name = f'{particle.tbl_aa}{particle.tbl_name}{particle.qq}+'
                 m_q = particle.get_ionic_moq_in_u()
                 self.moq[ion_name] = m_q
                 self.total_mass[ion_name] = m_q * particle.qq  # Calculate and store the total mass
                 print("chenrj _calculate_moqs 2...",ion_name,"\t total mass",self.total_mass[ion_name] )
        else:
             for particle in self.particles_to_simulate:
                 ion_name = f'{particle[1]}{particle[0]}{particle[4][-1]}+'
                 print("chenrj _calculate_moqs 3...",ion_name,"\t total mass")
                 for ame in self.ame_data:                   
                     if particle[0] == ame[6] and particle[2] == ame[4] and particle[3] == ame[3]:
                         pp = Particle(particle[0], particle[2], particle[3], self.ame, self.ring)
                         pp.qq = particle[4][-1]
                         m_q = pp.get_ionic_moq_in_u()
                         self.protons[ion_name] = ame[4]
                         self.moq[ion_name] = m_q
                         self.total_mass[ion_name] = m_q * pp.qq  # Calculate and store the total mass
                         break  # ✅ Exit the for-loop once a match is found
                         
    def _calculate_srrf(self, moqs = None, fref = None, brho = None, ke = None, gam = None, correct = None):
        if moqs:
            self.moq = moqs
        self.ref_mass = AMEData.to_mev(self.moq[self.ref_ion] * self.ref_charge)
        self.ref_frequency = self.reference_frequency(fref = fref, brho = brho, ke = ke, gam = gam)
        # Simulated relative revolution frequencies (respect to the reference particle)
        self.srrf = array([1 - self.alphap * (self.moq[name] - self.moq[self.ref_ion]) / self.moq[self.ref_ion]
                           for name in self.moq])
        if correct:
            self.srrf = self.srrf + polyval(array(correct), self.srrf * self.ref_frequency) / self.ref_frequency
            
    def build_ion_name(self,p):
        return f"{int(p[1])}{p[0]}{int(p[4][-1])}+"
        
    def _simulated_data(self, brho = None, harmonics = None, particles = False,mode = None, sim_scalingfactor = None, nions = None):
       self.harmonics = harmonics
       self.mode = mode  # 保存模式供 GUI 显示
       for harmonic in harmonics:
           ref_moq = self.moq[self.ref_ion]
           if mode == 'Bρ':
               ref_frequency =  self.ref_frequency*harmonic
               self.brho = brho
           else:
               ref_frequency =  self.ref_frequency
               self.brho = self.calculate_brho_relativistic(ref_moq, ref_frequency, self.ring.circumference, harmonic) #improve this line
       # Dictionary with the simulated meassured frecuency and expected yield, for each harmonic
       self.simulated_data_dict = dict()
       moq_keys = list(self.moq.keys())
       
       for key in moq_keys:
           found = False
           
           for p in self.particles_to_simulate:
               ion_name = self.build_ion_name(p)
               
               if ion_name == key:
                   yield_val = p[5]
                   self.yield_data.append(yield_val)
                   found = True
                   break
                   
       self.nuclei_names = array(moq_keys)
       # We normalize the yield to avoid problems with ranges and printing
       # If a scaling factor is provided, multiply yield_data by scalingfactor
       self.yield_data = np.array(self.yield_data, dtype=float)
       if sim_scalingfactor is not None:
           self.yield_data *= sim_scalingfactor
       # Get nuclei name for labels
       # Simulate the expected measured frequency for each harmonic:
       for harmonic in harmonics:
           simulated_data = array([])
           array_stack = array([])
       
           # get srf data
           if mode == 'Frequency':
               # If ref_harmonic is given, compute the true revolution frequency f0,
               # then multiply by the current harmonic to get the frequency at that harmonic.
               if self.ref_harmonic is not None and self.ref_harmonic != 0:
                   f0 = self.ref_frequency / self.ref_harmonic     # revolution frequency (harmonic = 1)
                   harmonic_frequency = self.srrf * f0 * harmonic  # frequency at this harmonic
               else:
                   harmonic_frequency = self.srrf * self.ref_frequency
           elif mode == 'Bρ':
                #harmonic_frequency = self.srrf * self.ref_frequency * harmonic
                print("self.ref_frequency = ",self.ref_frequency)
                harmonic_freq_list = []
                # ───────────── Bρ 固定模式：每個離子獨立計算自己的頻率 ─────────────
                for i, ion_name in enumerate(moq_keys):
                    try:
                        # 解析出該離子的質量數 A、元素、電荷 q
                        A, elem, q = self._parse_ion_name(ion_name)
                        
                        # 該離子的總質量（以 u 為單位） ≈ moq * q
                        
                        #mass_u = self.moq[ion_name] * q
                        mass_u = AMEData.to_mev(self.moq[ion_name] * q)
                        
                        # 使用現有的靜態方法計算這個離子的革命頻率
                        f_rev = ImportData.calc_ref_rev_frequency(
                            ref_mass=mass_u,                    # 注意單位是 u
                            ring_circumference=self.ring.circumference,
                            brho=brho,
                            ref_charge=q                        # 這個離子的電荷
                        )
                        
                        # 第 h 次諧波的頻率
                        this_harmonic_freq = harmonic * f_rev
                        harmonic_freq_list.append(this_harmonic_freq)
                    
                    except Exception as e:
                        print(f"計算離子 {ion_name} 在 Bρ 模式下的頻率失敗：{e}")
                        continue
                harmonic_frequency = np.array(harmonic_freq_list)
               
           # attach harmonic, frequency, yield data and ion properties together:
           print(f"harmonic_frequency shape: {harmonic_frequency.shape}")
           print(f"yield_data shape: {self.yield_data.shape}")
           print(f"nuclei_names shape: {self.nuclei_names.shape}")
           
           array_stack = stack((harmonic_frequency, self.yield_data, self.nuclei_names), axis=1)  # axis=1 stacks vertically
           simulated_data = append(simulated_data, array_stack)
           simulated_data = simulated_data.reshape(len(array_stack), 3)
           name = f'{harmonic}'
           self.simulated_data_dict[name] = simulated_data
        
    def calculate_brho_relativistic(self, moq, frequency, circumference, harmonic):
        """
            Calculate the relativistic magnetic rigidity (Bρ) of an ion.
            
            Parameters:
            moq (float): mass-to-charge ratio (m/q) of the ion
            frequency (float): frequency of the ion in Hz
            circumference (float): circumference of the ring in meters
            harmonic (float): harmonic number
            
            Returns:
            float: magnetic rigidity (Bρ) in T*m (Tesla meters)
            """
        # Speed of light in m/s
        #c = 299792458.0
        
        # Calculate the actual frequency of the ion
        actual_frequency = frequency / harmonic
        
        # Calculate the velocity of the ion
        v = actual_frequency * circumference
        
        # Calculate the Lorentz factor gamma
        gamma = 1 / np.sqrt(1 - (v / AMEData.CC) ** 2)
        
        # Calculate the momentum p = γ m v
        p = moq *AMEData.UU* gamma *  (v/AMEData.CC)/AMEData.CC
        
        # Calculate the magnetic rigidity (Bρ)
        brho = p*1e6
        return brho
                                                            
    def reference_frequency(self, fref = None, brho = None, ke = None, gam = None):
        
        # If no frev given, calculate frev with brho or with ke, whatever you wish
        if fref:
            return fref
        elif brho:
            return ImportData.calc_ref_rev_frequency(self.ref_mass, self.ring.circumference,
                                                     brho = brho, ref_charge = self.ref_charge)
        elif ke:
            return ImportData.calc_ref_rev_frequency(self.ref_mass, self.ring.circumference,
                                                     ke = ke, aa = self.ref_aa)
        elif gam:
            return ImportData.calc_ref_rev_frequency(self.ref_mass, self.ring.circumference,
                                                     gam = gam)
            
        else: sys.exit('None frev, brho, ke or gam')
        
    @staticmethod
    def calc_ref_rev_frequency(ref_mass, ring_circumference, brho = None, ref_charge = None, ke = None, aa = None, gam = None):
        
        if brho:
            gamma = ImportData.gamma_brho(brho, ref_charge, ref_mass)
        elif ke:
            gamma = ImportData.gamma_ke(ke, aa, ref_mass)
            
        elif gam:
            gamma = gam
        
        beta = ImportData.beta(gamma)
        velocity = ImportData.velocity(beta)
        
        return ImportData.calc_revolution_frequency(velocity, ring_circumference)
        
    @staticmethod
    def gamma_brho(brho, charge, mass):
        # 1e6 necessary for mass from mev to ev.
        return sqrt(pow(brho * charge * AMEData.CC / (mass * 1e6), 2)+1)
    
    @staticmethod
    def gamma_ke(ke, aa, ref_mass):
        # ke := Kinetic energy per nucleon
        return (ke * aa) / (ref_mass) + 1
    
    @staticmethod
    def beta(gamma):
        return sqrt(gamma**2 - 1) / gamma

    @staticmethod
    def velocity(beta):
        return AMEData.CC * beta
    
    @staticmethod
    def calc_revolution_frequency(velocity, ring_circumference):
        return velocity / ring_circumference
