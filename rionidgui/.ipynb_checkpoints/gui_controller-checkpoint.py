from numpy import argsort, where, append
from loguru import logger
from rionid.importdata import ImportData
from barion.amedata import AMEData
import time
import numpy as np
from PyQt5.QtWidgets import QMessageBox

def import_controller(datafile=None, filep=None, remove_baseline = None, psd_baseline_removed_l = None,psd_baseline_removed_ratio = None, alphap=None, refion=None, highlight_ions=None, harmonics = None, nions = None, amplitude=None, circumference = None, mode=None, sim_scalingfactor=None, value=None, reload_data=None,peak_threshold_pct = None,min_distance=None,output_results=None,saved_data = None,matching_freq_min=None,matching_freq_max=None,simulation_result=None):
    try:
        start_time = time.time()  # Record start time for each test_alphap iteration
        # initializations
        if float(alphap) > 1: alphap = 1/float(alphap)**2 # handling alphap and gammat
        fref = brho = ke = gam = None
        if mode == 'Frequency': fref = float(value)
        elif mode == 'Bρ': brho = float(value)
        elif mode == 'Kinetic Energy': ke = float(value)
        elif mode == 'Gamma': gam = float(value)
        # Calculations | ImportData library
        mydata = ImportData(refion, highlight_ions, remove_baseline, float(psd_baseline_removed_l),float(psd_baseline_removed_ratio), float(alphap), filename = datafile, reload_data = reload_data, circumference = circumference,peak_threshold_pct=peak_threshold_pct,min_distance=min_distance,matching_freq_min=matching_freq_min,matching_freq_max=matching_freq_max)
        end_time1 = time.time()  # Record end time after each iteration
        elapsed_time1 = end_time1 - start_time  # Calculate elapsed time for this iteration
        if reload_data: 
            mydata._set_particles_to_simulate_from_file(filep,verbose=output_results)
            mydata._calculate_moqs()
        else:
            mydata.ame = saved_data.ame
            mydata.ame_data = saved_data.ame_data
            mydata.particles_to_simulate = saved_data.particles_to_simulate
            mydata.protons = saved_data.protons
            mydata.moq = saved_data.moq
            mydata.total_mass = saved_data.total_mass
            mydata.peak_freqs = saved_data.peak_freqs
            mydata.peak_widths_freq = saved_data.peak_widths_freq
            mydata.peak_heights = saved_data.peak_heights
        mydata._calculate_srrf(fref = fref, brho = brho, ke = ke, gam = gam, correct = False)
        harmonics = [float(h) for h in harmonics.split()]
        mydata._simulated_data(brho = brho, harmonics = harmonics, mode = mode, sim_scalingfactor = sim_scalingfactor, nions = nions) # -> simulated frecs
        # "Outputs"
        if nions:
            display_nions(int(nions), mydata.yield_data, mydata.nuclei_names, mydata.simulated_data_dict, refion, harmonics)
        if output_results:
            logger.info(f'Simulation results (ordered by frequency) will be saved to simulation_result.out')
        sort_index = argsort(mydata.srrf)
        # Save the results if output_results is True
        if output_results:
            save_simulation_results(mydata,mode, harmonics, sort_index,simulation_result)
            logger.info(f'Succesfully saved!')

        return mydata # Returns the simulated spectrum data 
    except Exception as e:
        print(f"Error during calculations: {str(e)}")
        # ✅ Show error dialog
        QMessageBox.critical(None,"Error",f"An error occurred: {str(e)}")
        return None

def display_nions(nions, yield_data, nuclei_names, simulated_data_dict, ref_ion, harmonics):
    sorted_indices = argsort(yield_data)[::-1][:nions]
    ref_index = where(nuclei_names == ref_ion)[0]
    if ref_index not in sorted_indices:
        sorted_indices = append(sorted_indices, ref_index)
    nuclei_names = nuclei_names[sorted_indices]
    
    for harmonic in harmonics: # for each harmonic
        name = f'{harmonic}'
        simulated_data_dict[name] = simulated_data_dict[name][sorted_indices]

def save_simulation_results(mydata, mode, harmonics, sort_index=None, filename='simulation_result.out'):
    """
    Saves the simulation results to a specified file.
    
    Parameters:
    - mydata:        object containing simulated_data_dict, moq, total_mass, etc.
    - mode:          'Frequency' or 'Bρ'
    - harmonics:     list of harmonic numbers
    - sort_index:    (optional) global sorted indices — 如果不傳就每個 harmonic 單獨排序
    - filename:      output filename
    """
    with open(filename, 'w') as file:
        brho = getattr(mydata, 'brho', None)   # Bρ mode 才會有
        
        # 先寫整體資訊
        file.write(f"Simulation mode: {mode}\n")
        if brho is not None:
            file.write(f"Reference Bρ: {brho:.6f} Tm\n")
        file.write(f"Harmonics: {harmonics}\n")
        file.write("=" * 80 + "\n\n")
        
        for harmonic in harmonics:
            key = f'{harmonic}'   # 你當初是用 str(harmonic) 當 key
            if key not in mydata.simulated_data_dict:
                file.write(f"Harmonic {harmonic} : no data\n\n")
                continue
                
            data = mydata.simulated_data_dict[key]   # shape: (n_ions, 3)
            # data[:,0] → frequency
            # data[:,1] → yield
            # data[:,2] → ion name (string array)
            
            # 建議：每個 harmonic 獨立排序（頻率由小到大）
            if sort_index is None:
                # 就地排序索引
                sort_idx = np.argsort(data[:, 0].astype(float))
            else:
                # 如果堅持用外部傳入的全局排序索引（較少見）
                sort_idx = sort_index
            
            sorted_data = data[sort_idx]
            
            # 標頭
            header = f"Harmonic: {harmonic}"
            if brho is not None:
                header += f"    Bρ: {brho:.6f} Tm"
            file.write(header + "\n")
            
            file.write(
                f"{'ion':<18} "
                f"{'frequency [Hz]':<22} "
                f"{'yield [pps]':<16} "
                f"{'m/q [u]':<18} "
                f"{'mass [MeV/c²]':<18}\n"
            )
            file.write("-" * 90 + "\n")
            
            for row in sorted_data:
                ion_name   = row[2]
                freq       = float(row[0])
                yield_pps  = float(row[1])
                
                moq        = mydata.moq.get(ion_name, np.nan)
                mass_u     = mydata.total_mass.get(ion_name, np.nan)
                mass_mev   = AMEData.to_mev(mass_u) if not np.isnan(mass_u) else np.nan
                
                line = (
                    f"{ion_name:<18} "
                    f"{freq:<22.10f} "
                    f"{yield_pps:<16.4e} "
                    f"{moq:<18.12f} "
                    f"{mass_mev:<18.3f}"
                )
                file.write(line + "\n")
            
            file.write("\n" + "=" * 80 + "\n\n")

    print(f"Simulation results saved to: {filename}")