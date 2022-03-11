# LAB LIBRARIES
import bin2py
import visionloader as vl
import sta_utils as su
import old_labview_data_reader as oldlv
import eilib_new as eil
from .estim_viz import *

# IO LIBRARIES
import os
import sys
import pickle
import multiprocessing
import numpy as np

# edge dictionary
top_left = [264, 261, 259, 258, 257, 255, 250, 242, 235, 225, 215, 205, 195]
top_right = [195, 185, 175, 165, 155, 148, 140, 135, 133, 132, 130, 129, 126]
right = [126, 119, 110, 101, 92, 83, 74, 65, 56, 47, 38, 29, 20, 11, 4]
bot_right = [4, 1, 519, 518, 516, 514, 509, 501, 464, 474, 484, 494, 455];
bot_left = [455, 445, 435, 425, 415, 408, 400, 395, 393, 391, 390, 389, 386];
far_left = [386, 379, 370, 361, 352, 343, 334, 325, 316, 307, 298, 289, 280, 271, 264]
edges = [top_left, top_right, right, bot_right, bot_left, far_left]

def is_active_electrode(elec : int, max_trace : list, threshold : float, channel_noise : list, SIGMAS_BEYOND_NOISE : int) -> bool:
    """for a given signal recorded from an electrode, check if it is considered "electrode", as defined by whether or not its
    max amplitude is larger than the threshold, or if none given, a certain number of sigmas beyond the channel noise
    
    Args
        elec: electrode from which to get signal
        max_trace: max amplitude for each electrode
        threshold: threshold to compare electrode against, most likely the median axon amplitude
        channel_noise: amount of baseline channel noise on each electrode
        SIGMAS_BEYOND_NOISE: if threshold not defined, number of std devs beyond channel noise 
            to be considered an active electrode
    
    Returns
        True if electrode is active, False if otherwise
        
    """
    # check for dead elecs
    if channel_noise[elec] == 0.0:
        return False

    # activity detected only if significant
    if threshold is not None:
        if max_trace[elec] > threshold:
            return True
    else:
        if max_trace[elec] > SIGMAS_BEYOND_NOISE * channel_noise[elec]:
            return True
    return False

def criterion__strict_line(mean_trace, threshold, channel_noise, SIGMAS_BEYOND_NOISE):
    """criterion for which to check that there exists a bundle event.
    A bundle event exists on an array if there exists active electrodes which touch at least two non-adjacent edges
    of the array, with a strictly connected path of active electrodes between them.
    
    Args
        mean_trace: signals for every electrode
        threshold: threshold to compare electrode against, most likely the median axon amplitude
        channel_noise: amount of baseline channel noise on each electrode
        SIGMAS_BEYOND_NOISE: if threshold not defined, number of std devs beyond channel noise 
            to be considered an active electrode
        
    Returns
        True if bundle event exists, False if otherwise
    
    """
    max_trace = np.max(mean_trace, axis=1)
    
    sig_elecs = []
    # loop through all elecs this time
    NUM_ELECS = 519
    for i in range(NUM_ELECS):
        if is_active_electrode(i, max_trace, threshold, channel_noise, SIGMAS_BEYOND_NOISE):
            sig_elecs.append(i)
            
    G, pos = construct_path_graph(sig_elecs)
    
    sig_edge_elecs = [[], [], [], [], [], []] # 6 edges
    
    # check each electrode at edge for activity
    for i in range(len(edges)):
        for elec in edges[i]:

            elec_corrected = elec - 1 # just a list indexing change
            if is_active_electrode(elec_corrected, max_trace, threshold, channel_noise, SIGMAS_BEYOND_NOISE):
                sig_edge_elecs[i].append(elec_corrected)
    
    # check that there exists a strictly connected line between the edges
    for i in range(6):
        for elec1 in sig_edge_elecs[i]: 
            for j in range(2, 5): # X1X 2, 3, 4, 5 X6X
                for elec2 in sig_edge_elecs[(i+j) % 6]:
                    if (nx.has_path(G, elec1, elec2)): # indexng     
                        path, length = array_shortest_path(G, elec1, elec2)
                        # make sure the two elecs are not too close
                        if (length > 7):
                            
                            # if it satisfies all these conditions, then it is a bundle event
                            return True
    
    # if conditions not satisfied, then there is no bundle event
    return False
