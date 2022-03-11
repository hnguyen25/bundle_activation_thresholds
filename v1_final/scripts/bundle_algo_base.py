# LAB LIBRARIES
import bin2py
import visionloader as vl
import sta_utils as su
import old_labview_data_reader as oldlv
import eilib_new as eil

# IO LIBRARIES
import os
import sys
import pickle
import multiprocessing
import numpy as np
import time

# ==================================================
#          PREPROCESSING - AXON STATISTICS
# ==================================================

def axonorsomaRatio(wave : np.array, uppBound=1.6, lowBound=0.05) -> str:
    """based on a given wave, classify it as being characteristic of a soma, dendrite, or axon propogation
    
    Args:
        wave (1D numpy array): input signal
        uppBound: max value for which signal is not considered axonal
        lowBound: min value for which signal is not considered somatic
        
    Returns:
        (str) type of wave, either "axon", "soma", "mixed", or "error"
    
    """
    
    try:
        # Get index and value of (negative) min {only one}
        minind = np.argmin(wave)
        minval = np.min(wave)
        
        # Get max vals to either side of min
        maxvalLeft = np.max(wave[0:minind])
        maxvalRight = np.max(wave[minind:])
        
        if np.abs(minval) < max(maxvalLeft,maxvalRight):
            rCompt = 'dendrite'
            
        else:
            if maxvalRight == 0:
                ratio = 0
            else:
                ratio = maxvalLeft/maxvalRight
            if ratio > uppBound:
                rCompt = 'axon'
            elif ratio < lowBound: #FUDGED
                rCompt = 'soma'
            else:
                rCompt = 'mixed'
    
    except ValueError:
        rCompt = 'error' # wave is abnormally shaped (usually has min at leftmost or rightmost point)
    return rCompt


def get_axon_signal_dist_per_channel(vcd, channel_noise : np.array) -> list:
    """for all cells detected during visual stimulation, compile a list of all axonal signals
    
    Args:
        vcd: vision table, see .data_utils.get_vision_table()
        channel_noise: average mean noise on all elec channels
        
    Returns:
        (list) of lists of signals classified as "axonal", first classified according to which electrode the cell was identified from
    
    """
    
    cell_types = vcd.get_all_present_cell_types()   
    
    # keep all cell types that are true cells
    pruned_cell_types = []
    for cell_type in cell_types:
        if 'bad' in cell_type or 'dup' in cell_type:
            continue
        else:
            pruned_cell_types.append(cell_type)
    
    single_axon_list = [[] for i in range(519)]
    
    # get cells from all legit cell types
    for cell_type in pruned_cell_types:
        cellid_list = vcd.get_all_cells_of_type(cell_type)
        for cell in cellid_list:
            # get the electrical image of particular cell on array
            ei = vcd.get_ei_for_cell(cell).ei
            
            # check all elecs on array
            NUM_ELECS = 519
            for i in range(NUM_ELECS):
                wave = ei[i]
                max_signal = np.max(ei[i])
                
                if max_signal > channel_noise[i] or channel_noise[i] == 0:   
                    waveform_classification = axonorsomaRatio(wave)
                    
                    # keep signals that are classified as axons ("mixed" means a combination of somatic and axonal)
                    if waveform_classification == 'axon' or waveform_classification == 'mixed':
                        single_axon_list[i].append(max_signal)

    return single_axon_list


def get_single_axon_amp_statistics(dataset_path_dict : dict, dataset : str, plot=False) -> tuple:
    """calculate the statistics for the amplitude of an average single axon signal from 
    a given dataset, for use in thresholding for the bundle algorithm
    
    Args
        dataset_path_dict: see .data_utils.get_dataset_dict()
        dataset: name of dataset
        plot: whether or not to plot the distribution of axon amplitudes
    
    Returns
        (median, mean, std_dev) of all axon statistics
    """
    
    from .data_utils import get_vision_data
    
    # pull dataset info from dataset_path_dict
    vis_run = dataset_path_dict[dataset]['eidir']
    elec_run = dataset_path_dict[dataset]['seldir']
    data_folder_path = dataset_path_dict[dataset]['eidir2']
    
    # load visual stim data
    ANALYSIS_PARENT = '/Volumes/Analysis/'
    analysis_path = os.path.join(ANALYSIS_PARENT,dataset)
    vision_data = get_vision_data(dataset, vis_run, data_folder_path)
    channel_noise = vision_data.channel_noise
    
    vl_analysis_path = ANALYSIS_PARENT + dataset + '/' + vis_run
    
    # get all eis from visual stim data
    vcd = vl.load_vision_data(vl_analysis_path, data_folder_path,
                             include_params=True,
                             include_ei=True,
                             include_noise=True)
    
    single_axon_list = get_axon_signal_dist_per_channel(vcd, channel_noise)
    
    # combine all axons together
    composite_single_axon_list = []
    for channel_list in single_axon_list:
        for axon in channel_list:
            composite_single_axon_list.append(axon)

    # calculate signal statistics
    median = np.median(composite_single_axon_list)
    mean = np.mean(composite_single_axon_list)
    std_dev = np.std(composite_single_axon_list)
    
    if plot is True:
        fig, ax = plt.subplots()
        ax.set_title('Axon Dist for Dataset: ' + dataset)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Amplitude')

        names = ["median", "mean", "95th percentile"]
        colors = ['green', 'blue', 'orange']
        
        measurements = [median, mean, mean + 2*std_dev]
        for measurement, name, color in zip(measurements, names, colors):
            ax.axvline(x=measurement, linestyle='--', linewidth=2, label='{0}'.format(name), c=color)
        
        ax.legend();
        fig.show()
    
    return median, mean, std_dev

# ==================================================
#       PREPROCESSING - GENERATING PARAMS
# ==================================================

def pregen_dataset_params(dataset_path_dict : dict) -> dict:
    """preprocess and generate general dataset parameters for each dataset and compile into a dictionary.
    speeds up generation of parameters for multiprocessing overall.
    
    Args
        dataset_path_dict: see .data_utils.get_dataset_dict()
        
    Returns
        (dict) of dictionaries of general dataset parameters for each dataset in dataset_path_dict
    """
    pregenned_params = {}

    for dataset in dataset_path_dict.keys():
        print("Generating parameters for dataset " + dataset)
        start_time = time.time()
        
        vis_run = dataset_path_dict[dataset]['eidir']
        elec_run = dataset_path_dict[dataset]['seldir']
        data_folder_path = dataset_path_dict[dataset]['eidir2']

        ANALYSIS_PARENT = '/Volumes/Analysis/'
        analysis_path = os.path.join(ANALYSIS_PARENT,dataset,elec_run)
        vision_data = get_vision_data(dataset, vis_run, data_folder_path)

        threshold = None
        SIGMAS_BEYOND_NOISE = 4
        median, mean, std_dev = get_single_axon_amp_statistics(dataset_path_dict, dataset, plot=False)

        dataset_params = {
            'dataset':dataset,
            'vis_run':vis_run,
            'elec_run':elec_run,
            'data_folder_path':data_folder_path,
            'analysis_path': analysis_path,
            'channel_noise': vision_data.channel_noise,
            'threshold': threshold,
            'SIGMAS_BEYOND_NOISE': SIGMAS_BEYOND_NOISE,
            'axon_stats_median': median,
            'axon_stats_mean': mean,
            'axon_stats_std_dev': std_dev
        }

        pregenned_params[dataset] = dataset_params
        print("--- %s seconds ---" % (time.time() - start_time))
    return pregenned_params

from .bundle_criterions import *
def pregen_elec_params(tests : np.array , pregenned_dataset_params : dict, dataset_path_dict : dict, 
                       verbose=False, criterion={'strict_line': True}) -> list:
    
    """generate parameters for each electrode-dataset pair specified in tests
    
    Args
        tests: array with shape Nx2, with N being the number of tests cases and the first column
            consisting of the dataset name, and the second column consisting of the number of the
            stimulating electrode for each test case
        pregenned_dataset_params: for every dataset which contains test cases, pass through a
            dictionary containing useful parameters to know about the dataset
            see: pregen_dataset_params()
        dataset_path_dict: see .data_utils.get_dataset_dict()
        verbose: param to specificy whether or not to print debugging statements within bundle algorithm
        criterion: dictionary of criterions to use, each entry is a function within .criterion.py
    
    Returns
        (list) of params for all electrode-dataset pairs specified by tests that should have thresholds generated for
    """
    all_params = []
    
    for i in range(len(tests)):
        # pull out electrode-dataset info for each test case
        dataset = tests[i][0]
        stim_elec = tests[i][1]
        
        # add electrode-specific parameters
        one_elec_params = pregenned_dataset_params[dataset].copy()
        analysis_path = os.path.join('/Volumes/Analysis/',dataset,dataset_path_dict[dataset]['seldir'])
        
        one_elec_params['switching_artifact'] = np.mean(oldlv.get_oldlabview_pp_data(analysis_path,stim_elec,0),0) # DC offset
        one_elec_params['p_elec'] = int(stim_elec)
        one_elec_params['criterion'] = criterion
        one_elec_params['verbose'] = verbose
        one_elec_params['threshold'] = pregenned_dataset_params[dataset]['axon_stats_median']
            
        all_params.append(one_elec_params)
    return all_params

# ==================================================
#       BUNDLE ALGORITHM - WRAPPER FUNCTIONS
# ==================================================

# dictionary of electrodes that are on the edge of the array (CORNERS ARE EXCLUDED -> the commented electrodes)
top_left = [261, 259, 258, 257, 255, 250, 242, 235, 225, 215, 205] #264, 195
top_right = [185, 175, 165, 155, 148, 140, 135, 133, 132, 130, 129] #195, 126
right = [119, 110, 101, 92, 83, 74, 65, 56, 47, 38, 29, 20, 11] #126, 4
bot_right = [1, 519, 518, 516, 514, 509, 501, 464, 474, 484, 494]; #4, 455
bot_left = [445, 435, 425, 415, 408, 400, 395, 393, 391, 390, 389]; #455, 386
far_left = [379, 370, 361, 352, 343, 334, 325, 316, 307, 298, 289, 280, 271] #386, 264
edges = [top_left, top_right, right, bot_right, bot_left, far_left]


def generate_thresholds_for_given_test_cases(tests : np.array , all_params : list, 
                                             multiprocess=True, verbose=False, processes=24) -> np.array:
    """run the bundle algorithm for given test cases
    
    Args
        tests: array with shape Nx2, with N being the number of tests cases and the first column
            consisting of the dataset name, and the second column consisting of the number of the
            stimulating electrode for each test case
        all_params: parameters passed to bundle algorithm generated by pregen_elec_params()
        multiprocess: whether or not to run bundle algorithm in single-threaded or multiprocessing mode
        verbose: whether or not to print debugging statements within bundle algorithm (works best for single-threaded version)
        processes: number of CPU threads to use for multiprocessed version of algorithm
        
    Returns
        (np.array) containing thresholds generated by algorithm with shape Nx2 (col 0: index of elec || col 1: bundle threshold)
    """
    print("Running bundle algorithm...")
    
    if multiprocess is True:
        if verbose: print("<multiprocessed version>")
        pool = multiprocessing.Pool(processes)
        vals = pool.imap(generate_one_stimelec_bundle_threshold__multiprocessed, all_params)
    else:
        if verbose: print("<not multiprocessed>")
        vals = np.zeros((len(tests), 2))
        
        for i in range(len(all_params)):
            if verbose: print('processing ' + all_params[i]['analysis_path'] + ' ' + str(all_params[i]['p_elec']))
            p, threshold = generate_one_stimelec_bundle_threshold__multiprocessed(all_params[i])
            vals[i, 0] = p
            vals[i, 1] = threshold
    
    # Unpack multi-processing data into generated thresholds
    algo_thresholds = np.zeros((len(tests), 2))
    for idx, i in enumerate(vals):
        # column 0: index of electrode
        algo_thresholds[idx, 0] = i[0]
        # column 1: algorithm's bundle thresholds
        algo_thresholds[idx, 1] = i[1]
 
    return algo_thresholds

def generate_bundle_thresholds_for_dataset(dataset : str, vis_run : str, elec_run : str, data_folder_path : str, 
                        analyzable_elecs : list, threshold=None, SIGMAS_BEYOND_NOISE=4,
                        verbose=False, criterion={'strict_line': True}, multiprocess=True) -> np.array:
    """for a specified dataset, return the bundle threshold, if any, for every stimulating electrode on array
    
    Args
        dataset, vis_run, elec_run, data_folder_path: specifies dataset
        analyzable_elecs: list of electrode indices that should be analyzed (usually 1-519)
        threshold: if None, used as threshold for each corresponding recording elec as a threshold 
            for it to be considered "active" when calculating bundle threholds
        SIGMAS_BEYOND_NOISE: if threshold not specified, the number of std dev above channel noise 
            electrode activity must be to be considered "active" for purposes of thresholdling
        verbose: whether or not to print debugging statements within bundle algorithm (works best for single-threaded version)
        criterion: dictionary of criterions to use, each entry is a function within .criterion.py
        multiprocess: whether or not to run bundle algorithm in single-threaded or multiprocessing mode

    Returns
        (np.array) containing thresholds generated by algorithm with shape Nx2 (col 0: index of elec || col 1: bundle threshold)    
    """
    
    # path directory
    ANALYSIS_PARENT = '/Volumes/Analysis/'
    analysis_path = os.path.join(ANALYSIS_PARENT,dataset,elec_run)
    
    # load useful data
    vision_data = get_vision_data(dataset, vis_run, data_folder_path)

    pool = multiprocessing.Pool(processes=24)
    
    # args to pass through MP for each stimulating electrode to analyze
    all_params = []
    
    print("Generating params for elecs...")
    for i in analyzable_elecs:      
        # take the mean, used to subtract off the DC offset
        try:
            switching_artifact = np.mean(oldlv.get_oldlabview_pp_data(analysis_path,i,0),0) 
        except:
            continue
            
        one_elec_params = {
            'p_elec': int(i),
            'analysis_path': analysis_path,
            'switching_artifact': switching_artifact,
            'channel_noise': vision_data.channel_noise,
            'threshold': threshold,
            'SIGMAS_BEYOND_NOISE': SIGMAS_BEYOND_NOISE,
            'criterion':criterion,
            'verbose':verbose
        }
        all_params.append(one_elec_params)

    print("Running bundle algorithm...")
    start_time = time.time()
    
    if multiprocess is True:
        if verbose: print("<multiprocessed version>")
        pool = multiprocessing.Pool(processes=24)
        vals = pool.imap(generate_one_stimelec_bundle_threshold__multiprocessed, all_params)
        
        # Unpack multi-processing data into generated thresholds
        algo_thresholds = np.zeros((len(analyzable_elecs), 2))

        for idx, i in enumerate(vals):
            # column 0: index of electrode
            algo_thresholds[idx, 0] = i[0]
            # column 1: algorithm's bundle thresholds
            algo_thresholds[idx, 1] = i[1]
    else:
        if verbose: print("<not multiprocessed>")
        vals = np.zeros((len(tests), 2))
        for i in range(len(all_params)):
            if verbose: print('processing ' + all_params[i]['analysis_path'] + ' ' + str(all_params[i]['p_elec']))
            p, threshold = generate_one_stimelec_bundle_threshold__multiprocessed(all_params[i])
            vals[i, 0] = p
            vals[i, 1] = threshold
        
    print("--- Finished in %s seconds ---\n" % (time.time() - start_time))
    
    return algo_thresholds


def generate_one_stimelec_bundle_threshold__multiprocessed(info : dict, MAX_AMP=39) -> tuple:
    """for a specified electrode+dataset (specified in info dict), calculate its bundle threshold
    (wrapper function for binary search, multi-processed)
    
    Args
        info: dictionary containing all necessary params
            (see keys specified in generate_bundle_thresholds_for_dataset)
    
    Returns
        (elec, bundle_threshold) returns electrode number and the amplitude threshold of bundle threshold
    
    """
    return info['p_elec'], check_amps__binarysearch(0, MAX_AMP, info)

# ==================================================
#       BUNDLE ALGORITHM - INTERNAL
# ==================================================

def check_amps__binarysearch(min_amp : int, max_amp : int, info : dict) -> int:
    """performs binary search to find bundle threshold, i.e. the lowest amplitude at which a bundle event is detected

    Args
        info: dictionary containing all necessary params
            (see keys specified in generate_bundle_thresholds_for_dataset)
            
    Returns
        (int) amplitude index if bundle event detected, or -1 if no bundle event found at any amplitude
    """ 
    from .estim_viz import plot_ei
    
    # Base case: If there are no more amplitudes to search, there is no threshold
    if max_amp <= min_amp:
        if min_amp >= 38:
            return -1
        else:
            return max_amp
    else:
        # Next midpoint amplitude to check
        mid_amp = (min_amp + max_amp) // 2
         
        if info['verbose']: print("Checking mid amp " + str(mid_amp))
        
        try:       
            # Get estim mean trace data for this amplitude
            raw_data = oldlv.get_oldlabview_pp_data(info['analysis_path'], info['p_elec'], mid_amp)
            mean_trace = np.mean(raw_data[:,:,:],0) - info['switching_artifact']
            mean_trace[mean_trace > 0] = 0
            mean_trace = mean_trace[:, 7:50]
            mean_trace = np.abs(mean_trace) # we want to only check negative amplitudes

            # No edge array activity -> don't bother checking all electrodes
            if quick_amp_check(mean_trace, info['channel_noise'], info['threshold'], info['SIGMAS_BEYOND_NOISE']):
                # Check midpoint for bundle activity, using specified criterions
                if full_amp_check(mean_trace, info):
                    # If mid_amp has bundle activity, then check lower amplitudes
                    return check_amps__binarysearch(min_amp, mid_amp, info)

            # Else if mid_amp does not have bundle activity, check higher amplitudes
            return check_amps__binarysearch(mid_amp + 1, max_amp, info)
        except:
            print('no 38 movie index')
            return -1
    
def full_amp_check(mean_trace : np.array, info : dict) -> bool:
    """for a specific amplitude/stimelec, check if there exists bundle activity 
    according to criterion passed through info dict

    Args
        info: dictionary containing all necessary params
            (see keys specified in generate_bundle_thresholds_for_dataset)
    Returns
        (bool) True if bundle event detected, false otherwise    
    """
    criterion = info['criterion']
    
    if criterion['strict_line']:
        if criterion__strict_line(mean_trace, info['threshold'], info['channel_noise'], info['SIGMAS_BEYOND_NOISE']) is False:
            return False
        # can be extended to add more criterions in .bundle_criterions.py

    # if trace passes all given criterions, then we classify this as a bundle event
    return True
    

def quick_amp_check(mean_trace : np.array, channel_noise : list, threshold : float, NUM_SIGMAS_BEYOND_NOISE : int):
    """quick check for active electrode existing at two non-adjacent edges of 
    the array (necessary condition for bundle event)

    Args
        mean_trace: estim recorded data with DC offset subtracted (shape channels x time)
        threshold: threshold set for each electrode to be considered "active"
        NUM_SIGMAS_BEYOND_NOISE: if threshold is None, use number of std devs 
            above channel noise for electrode as method for considering if electrode
            is considered "active"
        channel_noise: average noise recorded before stimulation for each channel

    Returns
        True if there is edge activity at two-non adjacent edges, False if there is not
    """
    
    isEdgeActivity = False
    max_trace = np.max(mean_trace, axis=1)
    
    # check each electrode at edge for activity
    activity_at_edge = np.zeros((6))
    for i in range(len(edges)):
        for elec in edges[i]:
            elec_corrected = elec - 1 # just a list indexing change
            
            if is_active_electrode(elec_corrected, max_trace, threshold, channel_noise, NUM_SIGMAS_BEYOND_NOISE):
                activity_at_edge[i] = 1
    
    # check if non-adjacent edges have axon activity
    for i in range(6):
        if int(activity_at_edge[i]) == 1:
            for j in range(2,5): # X1X 2, 3, 4, 5 X6X
                if int(activity_at_edge[(i + j) % 6]) == 1:
                    isEdgeActivity = True
                        
    return isEdgeActivity

# ==================================================
#                 VALIDATION CHECKS
# ==================================================

def validate_bundle_algorithm(dataset : str, vis_run : str, elec_run : str, data_folder_path : str, 
    plot=True, save=True, overwrite=False, root_path='/Volumes/Scratch/Users/huy/bundle-analysis/',
    SIGMAS_BEYOND_NOISE=4, SIZE_SCALAR=1.5, FIG_SCALE=7, verbose=True) -> list:
    
    """wrapper function which compares manual threshold data against generated bundle 
    threshold data, with the capability to plot and save generate data if necessary

    note: manual threshold data must be in a .csv file located at root_path
    automatic and manual threshold data in array with dimensions (number_of_analyzable_elecs, 2)
    where 1st col -> id of electrode (out of 519) and 2nd col -> threshold indices (0-39, or -1 if DNE)
    
    Args
        dataset: dataset to analyze in /Volumes/Analysis (ex. '2021-05-27-0')
        vis_run: visual stim folder (ex. 'kilosort_data001/data001')
        elec_run: estim folder (ex. 'data002')
        data_folder_path: visual stim folder (ex.'data001')
        plot: whether or not to produce plots that show accuracy of bundle 
            algorithm in comparison to manual analysis
        save: save generated thresholds in a .csv folder located in root_path
        overwrite: whether or not, if thresholds already exist in a saved file, 
            to overwrite them with new thresholds (only do if algorithm has been changed)
        root_path: ex. '/Volumes/Scratch/Users/huy/bundle-analysis/'
    
    Returns
        (list) with the difference in amplitudes for each electrode analyzed through manual analysis
        and by the bundle algorithm
    """
    
    # (1) load manual analysis
    manual_thresholds_path = root_path + dataset + '_bundle_thresholds.csv'
    
    
    if os.path.isfile(manual_thresholds_path) is False:
        #print ("No manual analysis found for this dataset!")
        return None
    
    manual_thresholds =  np.genfromtxt(manual_thresholds_path, delimiter=',')
    manual_thresholds = manual_thresholds[1:,:2] # remove first row which only has labels 'p, thresholds'
    NUM_ANALYZABLE_ELECS = manual_thresholds.shape[0]
    
    # (2) load prior bundle algorithm analysis, if available
    thresholdsAlreadyGenerated = False
    auto_thresholds_path = root_path + dataset + '_auto_thresholds'
    auto_thresholds = None

    
    if os.path.isfile(auto_thresholds_path + ".npy") and overwrite is False:
        print("Getting previously generated thresholds for dataset " + dataset)
        auto_thresholds = np.load(auto_thresholds_path + ".npy")
        thresholdsAlreadyGenerated = True
    # (3) else, run bundle algorithm and generate thresholds
    else:
        print("Generating new thresholds for dataset " + dataset)
        analyzable_elecs = list(manual_thresholds[:,0])
        auto_thresholds = generate_bundle_thresholds_for_dataset(dataset, vis_run, elec_run, data_folder_path, analyzable_elecs,
            SIGMAS_BEYOND_NOISE=SIGMAS_BEYOND_NOISE, verbose=verbose)
    
    # (4) compare difference between manual and auto bundle thresholds
    threshold_differences = []
    for i in range(len(manual_thresholds)):
        if (manual_thresholds[i,1] == -1):
            continue
        else:
            if (auto_thresholds[i,1] == -1):
                continue
            else:
                diff = auto_thresholds[i,1] - manual_thresholds[i,1]    
            threshold_differences.append(diff)
        #print(auto_thresholds[i, 1])
        #print(manual_thresholds[i, 1])
    
    # (5) plot figures which show differences between manual and auto threshold
    if plot is True:
        print('\n\n================== STATS ====================')
        # for electrodes that have bundle events, plot the difference in amplitudes between manual and auto
        fig, ax = plt.subplots()
        bin_edges = np.arange(-5.5, 5.5+1, 1) # center bin around 0
        ax.hist(threshold_differences, bins=bin_edges, color='b')
        plt.title('Difference in manual and auto thresholds for ' + dataset)
        plt.xlabel('Difference in bundle threshold amplitude indices')
        plt.ylabel('Frequency')
        
        # confusion matrix (bundle event vs no bundle event)
        plot_confusion_matrix(manual_thresholds[:,1], auto_thresholds[:,1], threshold_differences)
        
    # (6) save auto bundle thresholds to file for later use
    if save is True:
        if thresholdsAlreadyGenerated is False or overwrite is True:
            print("Saving generated threshold data to path " + auto_thresholds_path)
            np.save(auto_thresholds_path, auto_thresholds)
    
    return threshold_differences

def plot_confusion_matrix(manual_original : list, generated_original : list, diff : list) -> None:
    """plot confusion matrix between manual analysis thresholds and bundle generated thresholds
    
    Args
        manual_original: thresholds from manual analysis
        generated_original: thresholds from bundle algorithm
        diff: pre-calculated difference between manual and generated thresholds from validate_bundle_algorithm
    """
    
    manual = manual_original.copy()
    generated = generated_original.copy()
    
    manual[manual!=-1] = 1
    manual[manual==-1] = 0

    generated[generated!=-1] = 1
    generated[generated==-1] = 0
    tt = 0 #good
    tf = 0 # no bundle found -> make detection less strict
    ft = 0 # no actual bundle -> make bundle detection more strict
    ff = 0 #good
    
    for i in range(len(manual)):
        if manual[i] == 1:
            if generated[i] == 1: tt += 1
            else: tf += 1
        else:
            if generated[i] == 1: ft += 1
            else: ff += 1
    
    num_thresholds = tt + tf + tf + ff
    num_true_thresholds = tt + tf
    
    diff = np.abs(diff)
    zero_diff = np.count_nonzero(diff == 0)
    one_diff = np.count_nonzero(diff == 1)
    two_diff = np.count_nonzero(diff == 2)
    three_diff = np.count_nonzero(diff == 3)
    weighted_accuracy = (zero_diff + one_diff * 0.75 + two_diff * 0.5 + three_diff * 0.25)/(tt+tf)
    
    print("Confusion Matrix: Manual Analysis Bundle Events (top) vs. Auto-Generated Bundle Events (Left)")
    print("  |TRUE|FALSE")
    print("T |", "{:03d}".format(tt), "{:03d}".format(ft))
    print("F |", "{:03d}".format(tf), "{:03d}".format(ff), "\n") 
    print("Accuracy:", "{0:.0%}".format((weighted_accuracy*(tt+tf)+ff)/(num_thresholds)), "(overall weighted accuracy)")
    print("Sensitivity:", "{0:.0%}".format((tt)/(tt+tf)), "(proportion of true bundle events classified as true)")
    print("----> 0 diff.", "{0:.0%}".format(zero_diff/num_true_thresholds))
    print("----> 1 diff.", "{0:.0%}".format((one_diff + zero_diff)/num_true_thresholds))
    print("----> 2 diff.", "{0:.0%}".format((two_diff + one_diff + zero_diff)/num_true_thresholds))
    print("----> 3 diff.", "{0:.0%}".format((three_diff + two_diff + one_diff + zero_diff)/num_true_thresholds))
    print("----> Weighted Accuracy:", "{0:.0%}".format(weighted_accuracy))
    print("Specifity:", "{0:.0%}".format((ff)/(ff+ft)), "(proportion of non-bundle events classified as false)")
    print("\n")