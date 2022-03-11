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

def get_bundle_test_cases(include_periphery=True, include_raphe=True, 
                          include_noevent_cases=True, include_event_cases=True,
                          custom_tests=False) -> list:
    """compile a set of test cases to test and validate bundle algorithm.
    each test case is specified as a specific dataset-electrode pair.
    
    Args
        include_periphery: include test cases from periphery datasets
        include_raphe: include test cases from raphe datasets
        include_event_nocases: include test cases where there was no bundle events
        include_event_cases: include test cases where there are bundle events
        custom_tests: include tests specified in "custom" variable
    
    Returns
        list of test cases
    """
    
    custom = [('2020-10-18-0', 178)]
    
    tests_periphery = [('2016-06-13-0',28),('2016-06-13-0',278),('2016-06-13-0',306),
                       ('2016-06-13-0',483),('2016-06-13-8',70),('2016-06-13-8',232),
                       ('2016-06-13-8',489),('2017-11-20-9',17),('2017-11-20-9',197),
                       ('2017-11-20-9',295),('2017-11-20-9',421),('2020-09-12-4',62),
                       ('2020-09-12-4',167),('2020-09-12-4',341),('2020-09-12-4',500),
                       ('2020-10-06-5',220),('2020-10-06-5',354),('2020-10-06-5',477),
                       ('2020-10-06-7',149),('2020-10-06-7',275),('2020-10-06-7',430),
                       ('2020-10-06-7',489),('2020-10-18-5',8),('2020-10-18-5',41),
                       ('2020-10-18-5',54),('2020-10-18-5',275),('2020-10-18-5',315),
                       ('2020-10-18-5',364),('2020-10-18-5',489)
                      ]
    
    tests_periphery_noevent = [('2016-06-13-0',89),('2016-06-13-0',202),('2016-06-13-8',330),
                               ('2016-06-13-9',8),('2020-10-06-5',122), ('2016-06-13-9',102),
                               ('2016-06-13-9',335),('2016-06-13-9',506)]
    
    tests_raphe = [('2018-03-01-1',178),('2018-03-01-1',221),('2018-03-01-1',448),
                   ('2019-06-20-0',115),('2019-11-07-2',58),('2020-01-30-1',203),
                   ('2020-01-30-1',320),('2020-01-30-1',430),('2020-01-30-1',498),
                   ('2020-02-27-2',17),('2020-02-27-2',63),('2020-02-27-2',203),
                   ('2020-02-27-2',313),('2020-02-27-2',467),('2020-09-29-2',145),
                   ('2020-09-29-2',286),('2020-09-29-2',405),('2020-10-18-0',18),
                   ('2020-10-18-0',108),('2020-10-18-0',178),('2020-10-18-0',240),
                   ('2020-10-18-0',278),('2020-10-18-0',339),('2020-10-18-0',493),
                   ('2021-05-27-0',95),('2021-05-27-0',159),('2021-05-27-0',203),
                   ('2021-05-27-0',305),('2021-05-27-0',339),('2021-05-27-0',348)]
    
    tests_raphe_noevent = [('2018-03-01-1',88),('2019-06-20-0',94),('2019-06-20-0',353),
                           ('2019-06-20-0',506),('2019-11-07-2',182),('2019-11-07-2',460),
                           ('2019-11-07-2',468),('2020-01-30-1',33),('2020-09-29-2',505),
                           ('2021-05-27-0',22)]
    
    tests = []
    if custom_tests: return custom
    if include_periphery and include_noevent_cases:
        tests += tests_periphery_noevent
    if include_periphery and include_event_cases:
        tests += tests_periphery
    if include_raphe and include_noevent_cases:
        tests += tests_raphe_noevent
    if include_raphe and include_event_cases:
        tests += tests_raphe
    return tests

def get_dataset_dict(include_raphe=True, include_periphery=True) -> dict:
    """get list of datasets to run bundle algorithm on
    
    Args
        include_raphe: include raphe datasets
        include_periphery: include periphery datasets
        
    Returns
        dictionary of dictionaries specifying each dataset
    """
    dataset_path_dict = {}
    dataset_path_dict = dict()
    
    if include_periphery:
        dataset_path_dict['2016-06-13-0'] = dict()
        dataset_path_dict['2016-06-13-0']['location'] = 'periphery'
        dataset_path_dict['2016-06-13-0']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2016-06-13-0']['eidir2'] = 'data000'
        dataset_path_dict['2016-06-13-0']['seldir'] = 'data001'

        dataset_path_dict['2016-06-13-8'] = dict()
        dataset_path_dict['2016-06-13-8']['location'] = 'periphery'
        dataset_path_dict['2016-06-13-8']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2016-06-13-8']['eidir2'] = 'data000'
        dataset_path_dict['2016-06-13-8']['seldir'] = 'data001'

        dataset_path_dict['2016-06-13-9'] = dict()
        dataset_path_dict['2016-06-13-9']['location'] = 'periphery'
        dataset_path_dict['2016-06-13-9']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2016-06-13-9']['eidir2'] = 'data000'
        dataset_path_dict['2016-06-13-9']['seldir'] = 'data001'

        dataset_path_dict['2017-11-20-9'] = dict()
        dataset_path_dict['2017-11-20-9']['location'] = 'periphery'
        dataset_path_dict['2017-11-20-9']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2017-11-20-9']['eidir2'] = 'data000'
        dataset_path_dict['2017-11-20-9']['seldir'] = 'data002'

        dataset_path_dict['2020-09-12-4'] = dict()
        dataset_path_dict['2020-09-12-4']['location'] = 'periphery'
        dataset_path_dict['2020-09-12-4']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2020-09-12-4']['eidir2'] = 'data000'
        dataset_path_dict['2020-09-12-4']['seldir'] = 'data001'

        dataset_path_dict['2020-10-06-5'] = dict()
        dataset_path_dict['2020-10-06-5']['location'] = 'periphery'
        dataset_path_dict['2020-10-06-5']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2020-10-06-5']['eidir2'] = 'data000'
        dataset_path_dict['2020-10-06-5']['seldir'] = 'data001'

        dataset_path_dict['2020-10-06-7'] = dict()
        dataset_path_dict['2020-10-06-7']['location'] = 'periphery'
        dataset_path_dict['2020-10-06-7']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2020-10-06-7']['eidir2'] = 'data000'
        dataset_path_dict['2020-10-06-7']['seldir'] = 'data001'

        dataset_path_dict['2020-10-18-5'] = dict()
        dataset_path_dict['2020-10-18-5']['location'] = 'periphery'
        dataset_path_dict['2020-10-18-5']['eidir'] = 'kilosort_data002/data002'
        dataset_path_dict['2020-10-18-5']['eidir2'] = 'data002'
        dataset_path_dict['2020-10-18-5']['seldir'] = 'data001'
    
    if include_raphe:
        dataset_path_dict['2019-06-20-0'] = dict()
        dataset_path_dict['2019-06-20-0']['location'] = 'raphe'
        dataset_path_dict['2019-06-20-0']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2019-06-20-0']['eidir2'] = 'data000'
        dataset_path_dict['2019-06-20-0']['seldir'] = 'data001'

        dataset_path_dict['2018-03-01-1'] = dict()
        dataset_path_dict['2018-03-01-1']['location'] = 'raphe'
        dataset_path_dict['2018-03-01-1']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2018-03-01-1']['eidir2'] = 'data000'
        dataset_path_dict['2018-03-01-1']['seldir'] = 'data001'

        dataset_path_dict['2019-11-07-2'] = dict()
        dataset_path_dict['2019-11-07-2']['location'] = 'raphe'
        dataset_path_dict['2019-11-07-2']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2019-11-07-2']['eidir2'] = 'data000'
        dataset_path_dict['2019-11-07-2']['seldir'] = 'data001'

        dataset_path_dict['2020-01-30-1'] = dict()
        dataset_path_dict['2020-01-30-1']['location'] = 'raphe'
        dataset_path_dict['2020-01-30-1']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2020-01-30-1']['eidir2'] = 'data000'
        dataset_path_dict['2020-01-30-1']['seldir'] = 'data001'

        dataset_path_dict['2020-02-27-2'] = dict()
        dataset_path_dict['2020-02-27-2']['location'] = 'raphe'
        dataset_path_dict['2020-02-27-2']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2020-02-27-2']['eidir2'] = 'data000'
        dataset_path_dict['2020-02-27-2']['seldir'] = 'data001'

        dataset_path_dict['2020-09-29-2'] = dict()
        dataset_path_dict['2020-09-29-2']['location'] = 'raphe'
        dataset_path_dict['2020-09-29-2']['eidir'] = 'kilosort_data002/data002'
        dataset_path_dict['2020-09-29-2']['eidir2'] = 'data002'
        dataset_path_dict['2020-09-29-2']['seldir'] = 'data003'

        dataset_path_dict['2020-10-18-0'] = dict()
        dataset_path_dict['2020-10-18-0']['location'] = 'raphe'
        dataset_path_dict['2020-10-18-0']['eidir'] = 'kilosort_data000/data000'
        dataset_path_dict['2020-10-18-0']['eidir2'] = 'data000'
        dataset_path_dict['2020-10-18-0']['seldir'] = 'data001'

        dataset_path_dict['2021-05-27-0'] = dict()
        dataset_path_dict['2021-05-27-0']['location'] = 'raphe'
        dataset_path_dict['2021-05-27-0']['eidir'] = 'kilosort_data001/data001'
        dataset_path_dict['2021-05-27-0']['eidir2'] = 'data001'
        dataset_path_dict['2021-05-27-0']['seldir'] = 'data002'
    
    return dataset_path_dict

# ========================================================================================

def get_vision_table(dataset : str, vis_run : str, elec_run : str, data_folder_path : str):
    """retrieve vision table using lab functions
    
    Args
        dataset: name of dataset
        vis_run, elec_run, data_folder_path: path to dataset
        
    Returns
        vision table for given dataset
    
    """
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
    
    return vcd, channel_noise

def get_vision_data(dataset : str, vis_datarun : str, data_folder_path : str):
    """get vision data container for a given dataset using lab tools
    
    Args
        dataset, vis_datarun, data_folder_path: 
        
    Returns
        vision data container
    """
    # Constants.
    PARENT_ANALYSIS = '/Volumes/Analysis/' # Where analysis lives.
    PARENT_DATA = '/Volumes/Data/' # Where data lives.

    # Set paths.
    analysis_path = os.path.join(PARENT_ANALYSIS,dataset,vis_datarun)    

    # Get Vision container with various things.
    vision_data = vl.load_vision_data(analysis_path,
                                      data_folder_path,
                                      include_params=True,
                                      include_ei=True,
                                      include_sta=False, 
                                      include_runtimemovie_params=True,
                                      include_neurons=False,
                                      include_noise=True)

    return vision_data

def get_raw_data(analysis_path : str, pattern_no : int, amp_ind : int) -> np.array:
    """get the raw, electrical stim data from the array using lab tools
    
    Args
        analysis_path: path to dataset data
        pattern_no: stimulating electrode to pull raw data for
        amp_ind: amplitude of stimulation to pull raw data for
    
    Returns
        raw data of all electrodes from array
    """
    raw_data = oldlv.get_oldlabview_pp_data(analysis_path, pattern_no, amp_ind)
    
    # grab raw data at first amplitude to zero everything by subtracting off the DC offset / average over 25 trials
    switching_artifact = np.mean(oldlv.get_oldlabview_pp_data(analysis_path, pattern_no, 0), axis=0)
    raw_data = np.mean(raw_data, axis=0) - switching_artifact
    return raw_data