# LAB LIBRARIES
import bin2py
import visionloader as vl
import sta_utils as su
import old_labview_data_reader as oldlv
import eilib_new as eil

from .data_utils import *


# IO LIBRARIES
import os
import sys
import pickle
import multiprocessing
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.patches import Ellipse
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.pyplot import cm


def get_electrode_coords(type : str) -> tuple:
    """get xcoords and ycoords of elecs on array for plotting purposes
    
    Args
        type: accomodate different array shapes, currently only one -- '519'
    
    Returns:
        (tuple) containing xcoords and ycoords for every elec of the array
    
    """
    xcoords = []
    ycoords = []
    if type == '519':
        # choose a random dataset with given array type
        vision_data = get_vision_data('2021-05-27-0', 'kilosort_data001/data001', 'data001')
        electrode_map = vision_data.electrode_map
        xcoords = electrode_map[:,0]
        ycoords = electrode_map[:,1]
    return xcoords, ycoords

def plot_ei(dataset : str, vis_run : str, elec_run : str, data_folder_path : str, pattern_no : int, amp_ind : int, threshold : float,
           start_frame=7, end_frame=50, NUM_SIGMAS=4) -> None:
    """plot a cell's still electrical image
    
    Args
        dataset, vis_run, elec_run, data_folder_path: info about dataset paths
        pattern_no: stimulating electrode at which ei is found
        amp_ind: stimulating amplitude index at which ei is found
        threshold: threshold at which an electrode is considered to part of the cell's ei
        start_frame: number of frames in to count for threshold, start at 7 to avoid artifacting at the beginning
        end_frame: end time for counting as ei
        NUM_SIGMAS: num sigmas above channel noise to consider, if threshold is None
    """    
     # get electrode map
    ANALYSIS_PARENT = '/Volumes/Analysis/'
    analysis_path = os.path.join(ANALYSIS_PARENT,dataset,elec_run)
    vision_data = get_vision_data(dataset, vis_run, data_folder_path)
    electrode_map = vision_data.electrode_map
    xcoords = electrode_map[:,0]
    ycoords = electrode_map[:,1]
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5.75))
    
    # draw red circle around stim pattern
    ax.scatter(xcoords[pattern_no-1], ycoords[pattern_no-1], 
               facecolors='none', edgecolors='r', s=100)
    ax.axis('off')
    ax.set_title('Amp ' + str(amp_ind), fontsize=5)
        
    # draw hexagon at border
    corner_elecs = [264, 195, 126, 4, 455, 386]
    for i in range(6):
        a_loc = corner_elecs[i]
        b_loc = corner_elecs[(i+1) % 6]
        ax.plot([xcoords[a_loc-1], xcoords[b_loc-1]], 
                [ycoords[a_loc-1], ycoords[b_loc-1]], 
                color='0.6')
    
    raw_data = get_raw_data(analysis_path, pattern_no, amp_ind)
    raw_data = raw_data[:, start_frame:end_frame]
    raw_data[raw_data > 0] = 0
    raw_data = np.abs(raw_data)
    max_data = np.max(raw_data, axis=1)

    if threshold is None:
        sig_inds = np.argwhere(max_data >= NUM_SIGMAS * vision_data.channel_noise).flatten()
        nonsig_inds = np.argwhere(max_data < NUM_SIGMAS * vision_data.channel_noise).flatten()
    else:
        sig_inds = np.argwhere(max_data >= threshold).flatten()
        nonsig_inds = np.argwhere(max_data < threshold).flatten()

    ax.scatter(xcoords[sig_inds], ycoords[sig_inds], c='b', s=0.5*max_data[sig_inds])
    ax.scatter(xcoords[nonsig_inds], ycoords[nonsig_inds], c='0.9', s=0.5*max_data[nonsig_inds])

def plot_five_ei(dataset : str, vis_run : str, elec_run : str, data_folder_path : str, 
                 pattern_no : int, amp_start : int, threshold : float, algo_amp : int,
                 start_frame=7, end_frame=50, NUM_SIGMAS=4) -> None:
    """plot the ei at which the bundle algorithm classifies a bundle event to exist, and the adjacent
    four stimulating amplitude indices, to validate bundle algorithm accuracy
    
    Args
        dataset, vis_run, elec_run, data_folder_path: info about dataset paths
        pattern_no: stimulating electrode at which ei is found
        amp_start: stimulating amplitude index at which ei is found
        threshold: threshold at which an electrode is considered to part of the cell's ei
        algo_amp: amplitude given by algorithms -> will be the center amplitude
        
        start_frame: number of frames in to count for threshold, start at 7 to avoid artifacting at the beginning
        end_frame: end time for counting as ei
        NUM_SIGMAS: num sigmas above channel noise to consider, if threshold is None
    """
     # get electrode map
    ANALYSIS_PARENT = '/Volumes/Analysis/'
    analysis_path = os.path.join(ANALYSIS_PARENT,dataset,elec_run)
    vision_data = get_vision_data(dataset, vis_run, data_folder_path)
    electrode_map = vision_data.electrode_map
    xcoords = electrode_map[:,0]
    ycoords = electrode_map[:,1]
    
    corner_elecs = [264, 195, 126, 4, 455, 386]
    fig, ax = plt.subplots(5, 1, figsize=(3, 5*4))    
    fig.suptitle('Generated threshold:' + str(algo_amp))

    for i in range(5):
        # draw red circle around stim pattern     
        ax[i].scatter(xcoords[pattern_no-1], ycoords[pattern_no-1], 
                   facecolors='none', edgecolors='r', s=100)
        ax[i].axis('off')
    
        # draw hexagon at border
        for j in range(6):
            a_loc = corner_elecs[j]
            b_loc = corner_elecs[(j+1) % 6]
            ax[i].plot([xcoords[a_loc-1], xcoords[b_loc-1]], 
                    [ycoords[a_loc-1], ycoords[b_loc-1]], 
                    color='0.6')
    
    amp_ind = amp_start
    for i in range(5):
        ax[i].set_title('Amp ' + str(amp_ind), fontsize=15)
        raw_data = get_raw_data(analysis_path, pattern_no, amp_ind)
        raw_data = raw_data[:, start_frame:end_frame]
        raw_data[raw_data > 0] = 0
        raw_data = np.abs(raw_data)
        max_data = np.max(raw_data, axis=1)

        if threshold is None:
            sig_inds = np.argwhere(max_data >= NUM_SIGMAS * vision_data.channel_noise).flatten()
            nonsig_inds = np.argwhere(max_data < NUM_SIGMAS * vision_data.channel_noise).flatten()
        else:
            sig_inds = np.argwhere(max_data >= threshold).flatten()
            nonsig_inds = np.argwhere(max_data < threshold).flatten()

        ax[i].scatter(xcoords[sig_inds], ycoords[sig_inds], c='b', s=0.5*max_data[sig_inds])
        ax[i].scatter(xcoords[nonsig_inds], ycoords[nonsig_inds], c='0.9', s=0.5*max_data[nonsig_inds])
        
        amp_ind += 1
        if amp_ind > 38: amp_ind = 38
    return fig, ax

def plot_twentyfive_ei(dataset : str, vis_run : str, elec_run : str, data_folder_path : str, 
                       pattern_no : int, threshold : float,
                       start_frame=7, end_frame=50, NUM_SIGMAS=4, amp_start=14):
    """plot eis for a given dataset and stimulating electrode, from amplitudes indices #13-38
    
    Args      
        dataset, vis_run, elec_run, data_folder_path: info about dataset paths
        pattern_no: stimulating electrode at which ei is found   
        start_frame: number of frames in to count for threshold, start at 7 to avoid artifacting at the beginning
        end_frame: end time for counting as ei
        NUM_SIGMAS: num sigmas above channel noise to consider, if threshold is None
    """
    
     # get electrode map
    ANALYSIS_PARENT = '/Volumes/Analysis/'
    analysis_path = os.path.join(ANALYSIS_PARENT,dataset,elec_run)
    vision_data = get_vision_data(dataset, vis_run, data_folder_path)
    electrode_map = vision_data.electrode_map
    xcoords = electrode_map[:,0]
    ycoords = electrode_map[:,1]
    
    corner_elecs = [264, 195, 126, 4, 455, 386]
    
    fig, ax = plt.subplots(5, 5, figsize=(5*3, 5*3.6))
    
    fig.suptitle(dataset + ', p' + str(pattern_no))
    
    for i in range(5):
        for k in range(5):
            # draw red circle around stim pattern

            ax[i,k].scatter(xcoords[pattern_no-1], ycoords[pattern_no-1], 
                       facecolors='none', edgecolors='r', s=100)
            ax[i,k].axis('off')
        
            # draw hexagon at border
            for j in range(6):
                a_loc = corner_elecs[j]
                b_loc = corner_elecs[(j+1) % 6]
                ax[i,k].plot([xcoords[a_loc-1], xcoords[b_loc-1]], 
                        [ycoords[a_loc-1], ycoords[b_loc-1]], 
                        color='0.6')

    for i in range(5):
        for k in range(5):
            amp_ind = amp_start + i*5+k
            ax[i,k].set_title('Amp ' + str(amp_ind), fontsize=15)
            raw_data = get_raw_data(analysis_path, pattern_no, amp_ind)
            raw_data = raw_data[:, start_frame:end_frame]
            raw_data[raw_data > 0] = 0
            raw_data = np.abs(raw_data)
            max_data = np.max(raw_data, axis=1)

            if threshold is None:
                sig_inds = np.argwhere(max_data >= NUM_SIGMAS * vision_data.channel_noise).flatten()
                nonsig_inds = np.argwhere(max_data < NUM_SIGMAS * vision_data.channel_noise).flatten()
            else:
                sig_inds = np.argwhere(max_data >= threshold).flatten()
                nonsig_inds = np.argwhere(max_data < threshold).flatten()

            ax[i,k].scatter(xcoords[sig_inds], ycoords[sig_inds], c='b', s=0.5*max_data[sig_inds])
            ax[i,k].scatter(xcoords[nonsig_inds], ycoords[nonsig_inds], c='0.9', s=0.5*max_data[nonsig_inds])
        
        amp_ind += 1
        if amp_ind > 38: amp_ind = 38
    return fig, ax

def construct_path_graph(elec_list : list) -> tuple:
    """use NetworkX tools to create a graph where the nodes represent each electrode on the array and also
    on elec_list, as well as edges representing connections between adjacent electrodes, if they exist
    
    Args
        elec_list: number of acceptable electrodes to generate a path graph from
        
    Returns
        the NetworkX path graph container
    """
    
    xcoords, ycoords = get_electrode_coords('519')
    
    # get coords
    points = np.stack((xcoords, ycoords), axis=0)
    points = list(zip(points[0], points[1])) # converts 2D numpy array to list of 2-tuples
    
    #pos = {index:point for index, point in enumerate(points)}
    pos = {}
    for i in range(len(points)):
        if i in elec_list:
            pos[i] = points[i] 
    
    # generate nodes
    G = nx.Graph()   
    for i in pos:
        G.add_node(i)
    
    # generate edges
    for i in pos.keys():
        for j in pos.keys():
            if is_neighbor(xcoords, ycoords, i, j):
                G.add_edge(i, j, weight=1)

    return G, pos


def visualize_graph(G, pos, scale=20, 
                    SIZE_SCALAR=1.5, annotation=False, title="") -> None:
    """simple way to visualize generated path graph
    
    Args
        G, pos: NetworkX graph containers
        scale: size of the plot
        SIZE_SCALAR: size of the elements within the plot
        annotation: whether or not electrodes are numbered with their elec nums
        title: title of the plot
    """
    
    fig, ax = plt.subplots(figsize=(scale,scale*1.1))
    ax.axis('off')
    nx.draw(G, pos=pos, node_size=600, ax=ax)  # draw nodes and edges
    nx.draw_networkx_labels(G, pos=pos, font_color='w')  # draw node labels/names
    fig.suptitle(title)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()
    
def array_shortest_path(G, elec1 : int, elec2 : int) -> tuple:
    """find the shortest path between two electrodes, given a particular path graph
    
    Args
        G: graph container
        elec1: start point
        elec2: end point
    
    Return
        the path of electrodes to traverse from elec1 to elec2 given graph G, 
        and the length of the path
    """
    path = nx.shortest_path(G, elec1, elec2)
    length = len(path) - 1
    return path, length

def is_neighbor(xcoords, ycoords, a, b):
    p = [xcoords[a], ycoords[a]]
    q = [xcoords[b], ycoords[b]]
    return np.linalg.norm(np.array(p) - np.array(q)) < 34

def visualizeMultipleStimAmps(dataset, vis_run, elec_run, data_folder_path, pattern_no, amp_start=27, 
                              scale=20, playback_interval=200, annotation=False, threshold=None, NUM_SIGMAS=4,
                              SIZE_SCALAR=1.5, FRAME_START=7, NUM_FRAMES=25, NUM_COLS=5, NUM_ROWS=2):
    """create animated figure showing electrical stimulation at multiple amplitudes for a particular dataset-stimelec pair
    
    Args
        dataset, vis_run, elec_run, data_folder_path: info about dataset paths
        pattern_no: stimulating electrode at which ei is found
        amp_start: lowest amplitude indices to show
        
        scale: size of entire figure
        playback_interval: how fast animation is played
        annotation: whether or not each electrode is annotated with its id, can only be done for a still image
        threshold: threshold at which an electrode is considered to part of the cell's ei
        
        NUM_SIGMAS: num sigmas above channel noise to consider, if threshold is None
        SIZE_SCALAR: size of elements within figure
        FRAME_START: number of frames in to count for threshold, start at 7 to avoid artifacting at the beginning
        NUM_FRAMES: number of frames to make animation for
        NUM_COLS: number of columns of figures
        NUM_ROWS: number of rows of figures
        
        note: total number of amplitudes shown is equal to NUM_COLS * NUM_ROWS
    """
    if NUM_FRAMES - FRAME_START > 100:
        print('the frame range is too large, specify different FRAME_START and NUM_FRAMES')
        return
    
    # get electrode map
    ANALYSIS_PARENT = '/Volumes/Analysis/'
    analysis_path = os.path.join(ANALYSIS_PARENT,dataset,elec_run)
    vision_data = get_vision_data(dataset, vis_run, data_folder_path)
    electrode_map = vision_data.electrode_map
    xcoords = electrode_map[:,0]
    ycoords = electrode_map[:,1]
    
    # 1 x 5 figure, each subplot represents animation of raw data at a single amplitude index
    fig, ax = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(scale,scale * NUM_ROWS/NUM_COLS * 1.05))
    fig.suptitle(analysis_path + ', p' + str(pattern_no) + ', t=0', fontsize=scale)

    # initialize plots
    scat_sig = []
    scat_nonsig = []
    raw_data = []
    
    for a in range(NUM_ROWS * NUM_COLS):    
        
        amp = a + amp_start
        raw_data.append(get_raw_data(analysis_path, pattern_no, amp))
        
        if NUM_ROWS == 1:
            
            scat_sig.append(ax[a].scatter(xcoords, ycoords, c='b', s=SIZE_SCALAR*1))
            scat_nonsig.append(ax[a].scatter(xcoords, ycoords, c='0.9', s=SIZE_SCALAR*1))
            ax[a].axis('off')
            ax[a].set_title('Amp ' + str(amp), fontsize=scale*0.75)
        
            # draw red circle around stim pattern
            ax[a].scatter(xcoords[pattern_no-1], ycoords[pattern_no-1], facecolors='none', edgecolors='r', s=SIZE_SCALAR*100)

            # draw hexagon at border
            corner_elecs = [264, 195, 126, 4, 455, 386]
            for i in range(6):
                a_loc = corner_elecs[i]
                b_loc = corner_elecs[(i+1) % 6]
                ax[a].plot([xcoords[a_loc-1], xcoords[b_loc-1]], [ycoords[a_loc-1], ycoords[b_loc-1]], color='0.6')
            
        else:    
            x = a // NUM_COLS
            y = a % NUM_ROWS

            scat_sig.append(ax[x, y].scatter(xcoords, ycoords, c='b', s=SIZE_SCALAR*1))
            scat_nonsig.append(ax[x, y].scatter(xcoords, ycoords, c='0.9', s=SIZE_SCALAR*1))
            ax[x, y].axis('off')
            ax[x, y].set_title('Amp ' + str(amp), fontsize=scale*0.75)

            # draw red circle around stim pattern
            ax[x, y].scatter(xcoords[pattern_no-1], ycoords[pattern_no-1], facecolors='none', edgecolors='r', s=SIZE_SCALAR*100)

            # draw hexagon at border
            corner_elecs = [264, 195, 126, 4, 455, 386]
            for i in range(6):
                a_loc = corner_elecs[i]
                b_loc = corner_elecs[(i+1) % 6]
                ax[x, y].plot([xcoords[a_loc-1], xcoords[b_loc-1]], [ycoords[a_loc-1], ycoords[b_loc-1]], color='0.6')

    def animate(i):
        i += FRAME_START
        fig.suptitle(analysis_path + ', p' + str(pattern_no) + ', t=' + str(i), fontsize=scale)
        
        for a in range(NUM_ROWS * NUM_COLS):
            amp = a + amp_start
            
            raw_data_frame = raw_data[a][:,i]
                        
            sig_inds, nonsig_inds = get_significant_elecs(vision_data.channel_noise, raw_data_frame, NUM_SIGMAS, threshold=threshold)
            
            sig_coords = np.stack((xcoords[sig_inds], ycoords[sig_inds]), axis=0).T
            nonsig_coords = np.stack((xcoords[nonsig_inds], ycoords[nonsig_inds]), axis=0).T
            
            raw_data_frame[raw_data_frame > 0] = 0
            
            scat_sig[a].set_offsets(sig_coords)
            scat_sig[a].set_sizes(SIZE_SCALAR * np.abs(raw_data_frame[sig_inds]))
            
            scat_nonsig[a].set_offsets(nonsig_coords)
            scat_nonsig[a].set_sizes(SIZE_SCALAR * np.abs(raw_data_frame[nonsig_inds]))
            
            
    anim = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=playback_interval)
    return anim


def get_significant_elecs(noise : list, raw_data_frame : np.array, num_sigmas : int, threshold=None) -> tuple:
    """for given raw data, get electrodes to exceed a certain threshold, if threshold is not specified, then
        choose electrodes that are beyond a certain number of std devs (num_sigmas) above the channel noise
    
    Args
        noise: channel noise, for each elec
        raw_data_frame: raw data
        num_sigmas: std devs above noise to be considered significant
        threshold: set value beyond which elecs are considered significant
        
    Returns
        list of significant elecs, and list of non-significant elecs
    
    """
    
    if threshold is None: 
        # zero out positive values (trying to find the most negative values), flip sign for graphing purposes
        raw_data_frame[raw_data_frame>0] = 0
        raw_data_frame = np.abs(raw_data_frame)

        sig_channel_inds = np.argwhere(raw_data_frame > num_sigmas * noise).flatten()
        non_sig_channel_inds = np.argwhere(raw_data_frame <= num_sigmas * noise).flatten()

        num_channels = 519
        assert np.all(np.union1d(sig_channel_inds, non_sig_channel_inds) == np.arange(0, num_channels))
    else:
        # zero out positive values (trying to find the most negative values), flip sign for graphing purposes
        raw_data_frame[raw_data_frame>0] = 0
        raw_data_frame = np.abs(raw_data_frame)

        sig_channel_inds = np.argwhere(raw_data_frame > threshold).flatten()
        non_sig_channel_inds = np.argwhere(raw_data_frame <= threshold).flatten()

        num_channels = 519
        assert np.all(np.union1d(sig_channel_inds, non_sig_channel_inds) == np.arange(0, num_channels))
        
    return sig_channel_inds, non_sig_channel_inds