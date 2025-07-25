from numpy import polyval, array, stack, append, sqrt, genfromtxt
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
    def __init__(self, refion, highlight_ions, remove_baseline, psd_baseline_removed_l,alphap, filename = None, reload_data = None, circumference = None, peak_threshold_pct=None,min_distance=None,matching_freq_min=None,matching_freq_max=None):
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
        self.peak_heights = []
        self.gammats = []
        self.yield_data=[]
        # Data cache file path
        self.cache_file = self._get_cache_file_path(filename)
        self.chi2= 0
        self.match_count=0
        self.min_distance=min_distance
        self.matching_freq_min=matching_freq_min
        self.matching_freq_max=matching_freq_max
        self.remove_baseline = remove_baseline
        self.psd_baseline_removed_l=psd_baseline_removed_l
        self.psd_baseline_removed_ratio=1e-6
        
        # Get the experimental data
        if filename is not None:
            if reload_data:
                self._get_experimental_data(filename)
                self._save_experimental_data()
            else:
                self._load_experimental_data()
        else:
            print("No experimental data file provided. Using default or simulated data.")
            self.experimental_data = None  # Set empty or simulated data here
        
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
                        match_frequency_max=None):
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
        # Build list of (frequency, ion_name) from simulated_data_dict
        sim_items = []
        for sdata in self.simulated_data_dict.values():
            for row in sdata:
                sim_items.append((float(row[0]), row[2]))
        sim_freqs = np.array([freq for freq, _ in sim_items])
    
        # Initialize accumulators
        chi2 = 0.0
        match_count = 0
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
                chi2 += diff**2
                match_count += 1
                matched_ions.append(sim_items[idx][1])
                matched_sim_items.append(sim_items[idx])
                matched_sim_freqs.append(sim_freqs[idx])
                matched_exp_freqs.append(exp_freq)
                matched_peak_widths.append(width)
                matched_peak_heights.append(height)
    
        # Finalize chi²
        chi2 = chi2 / match_count if match_count > 0 else float('inf')
    
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
            for ion, sim_f, exp_f, w, h, gt in zip(
                self.matched_ions,
                self.matched_sim_freqs,
                self.matched_exp_freqs,
                self.matched_peak_widths,
                self.matched_peak_heights,
                self.gammats
            ):
                moq = self.moq[ion]
                HN = self.harmonics[0]
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

    def _get_cache_file_path(self, filename):
        base, _ = os.path.splitext(filename)
        return f"{base}_cache.npz"
    
    def _get_experimental_data(self, filename):
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
                
         # remove baseline or not.
        if self.remove_baseline:
            try:
                if self.experimental_data is not None and len(self.experimental_data) == 2:
                    freq, psd = self.experimental_data
                    baseline = NONPARAMS_EST(psd).pls('BrPLS', l=self.psd_baseline_removed_l, ratio=self.psd_baseline_removed_ratio)
                    psd_baseline_removed = psd - baseline
                    self.psd_baseline_removed = (freq, psd_baseline_removed)
                    self.experimental_data = (freq, psd_baseline_removed)
                    print("✅ Baseline removal completed. Result stored in self.psd_baseline_removed.")
                else:
                    print("⚠️ Invalid format of self.experimental_data. Skipping baseline removal.")
            except Exception as e:
                print("❌ Baseline removal failed:", e)
                traceback.print_exc()

    
        self.detect_peaks_and_widths()
        
    def detect_peaks_and_widths(self):
        if self.experimental_data is None:
            return
    
        freq, amp = self.experimental_data
    
        # 2) set up your thresholds
        rel_height = max(0.0, min(self.peak_threshold_pct, 1.0))
        height_thresh = np.max(amp) * rel_height
        min_dist    = float(self.min_distance)
        min_prom    = height_thresh * 0.3      # e.g. at least 30% of your threshold
        min_w       = 1                         # in samples, adjust to reject narrow spikes
        
        # 3) call find_peaks with prominence and minimum width
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
        print(f"Detected {len(peaks)} peaks after filtering by height={height_thresh:.2g}, "
              f"prominence>={min_prom:.2g}, width>={min_w} samples.")

    def _save_experimental_data(self):
        if self.experimental_data is not None:
            frequency, amplitude_avg = self.experimental_data
            np.savez_compressed(self.cache_file, frequency=frequency, amplitude_avg=amplitude_avg)                        
            
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
        if particles:
             for particle in particles:
                 ion_name = f'{particle.tbl_aa}{particle.tbl_name}{particle.qq}+'
                 m_q = particle.get_ionic_moq_in_u()
                 self.moq[ion_name] = m_q
                 self.total_mass[ion_name] = m_q * particle.qq  # Calculate and store the total mass
        else:
             for particle in self.particles_to_simulate:
                 ion_name = f'{particle[1]}{particle[0]}{particle[4][-1]}+'
                 for ame in self.ame_data:
                     #print("chenrj ame= ",ame)
                     if particle[0] == ame[6] and particle[1] == ame[5]:
                         pp = Particle(particle[2], particle[3], self.ame, self.ring)
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
               harmonic_frequency = self.srrf * self.ref_frequency
           else:
               harmonic_frequency = self.srrf * self.ref_frequency * harmonic
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
