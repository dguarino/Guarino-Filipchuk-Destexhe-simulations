"""
Copyright (c) 2022, Domenico GUARINO
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite Paris Saclay nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL GUARINO BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import glob
import gc
import json
import pickle
import sys
import resource
import collections
import random
import math
import warnings
import itertools
from functools import cmp_to_key
from itertools import zip_longest # analysis
import numpy as np

################################
import matplotlib
matplotlib.use('Agg') # to be used when DISPLAY is undefined as in Docker
################################

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import LineCollection
import matplotlib.colors as mpcolors
import matplotlib.cm as mpcm
from matplotlib import image as pltimg
from matplotlib.patches import PathPatch
import matplotlib.patches as patches

from neo.core import AnalogSignal # analysis
import quantities as pq

# SciPy related
import scipy.signal as signal
from scipy.signal import savgol_filter
import scipy.cluster.hierarchy as hierarchy
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import pdist, jaccard, hamming
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import shapiro
from scipy.stats import hypergeom
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy
import scipy.linalg
import scipy.signal as signal


# ------------------------------------------------
# Additional classes and functions
from helper_functions import *
# ------------------------------------------------


def analyse(params, folder, addon='', removeDataFile=False):
    print("\nAnalysing data...")

    # populations key-recorders match
    populations = {}
    for popKey,popVal in params['Populations'].items():
        # if popKey != 'ext': # if is not the drive, to be dropped
        #     if popKey in params['Recorders']:
        #         populations[popKey] = list(params['Recorders'][popKey].keys())
        # we do the analysis on what we recorded
        if popKey in params['Recorders']:
            populations[popKey] = list(params['Recorders'][popKey].keys())


    ###################################
    # iteration over populations and selective plotting based on params and available recorders
    for key,rec in populations.items():
        print("\n\nAnalysis for:",key)

        # assuming 2D structure to compute the edge N
        n = 0
        if isinstance(params['Populations'][key]['n'], dict):
            n = int(params['Populations'][params['Populations'][key]['n']['ref']]['n'] * params['Populations'][key]['n']['ratio'])
        else:
            n = int(params['Populations'][key]['n'])
        edge = int(np.sqrt(n))

        # state
        for trial_id,trial in enumerate(params['state']):
            print("\n"+trial['name'])

            for itrial in range(trial['count']):
                print("trial #",itrial)
                timeslice_start = params['run_time'] * trial_id + params['Analysis']['transient'] # to avoid initial transient
                timeslice_end   = params['run_time'] * (trial_id+itrial+1)
                print("trial-based slicing window (ms):", timeslice_start, timeslice_end)

                # Get cell indexes and ids
                cell_coords = []
                cell_indexes = []
                cell_ids = []
                with open(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+'_positions.txt', 'r') as posfile:
                    print("... getting cell indexes and ids")
                    lines = posfile.readlines()
                    posfile.close()
                    for line in lines:
                        cell_coords.append( [int(float(i)) for i in line.split(' ')[:4]] ) # not including 4
                    # print(cell_coords) # id, idx, x, y
                    #[ ..., [13287, 4064, 63, 32], [13288, 4065, 63, 33], ... ]
                    cell_coords = np.array(cell_coords)
                    cell_ids = cell_coords[:,0]
                    cell_indexes = range(0,len(cell_coords))

                # get data
                print("from file:",key+addon+'_'+trial['name']+str(itrial))
                neo = pickle.load( open(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+'.pkl', "rb") )
                data = neo.segments[0]
                # getting and slicing data
                # continuous variables (Vm, Gsyn, W) are sliced just according to the dt
                if 'w' in rec:
                    w = data.filter(name = 'w')[0]#[timeslice_start:timeslice_end]
                if 'v' in rec:
                    vm = data.filter(name = 'v')[0]#[timeslice_start:timeslice_end]
                if 'gsyn_exc' in rec:
                    gexc = data.filter(name = 'gsyn_exc')[0]#[timeslice_start:timeslice_end]
                if 'gsyn_inh' in rec:
                    ginh = data.filter(name = 'gsyn_inh')[0]#[timeslice_start:timeslice_end]
                # discrete variables (spiketrains) are sliced according to their time signature
                if 'spikes' in rec:
                    spiketrains = []
                    if 'subsampling' in params['Analysis']:
                        cell_indexes = np.random.choice(len(data.spiketrains), params['Analysis']['subsampling'], replace=False)
                        cell_coords = cell_coords[cell_indexes]
                        cell_ids = cell_ids[cell_indexes]
                        for spiketrain in [data.spiketrains[i] for i in cell_indexes]:
                            spiketrains.append(spiketrain[ (spiketrain>=timeslice_start) & (spiketrain<=timeslice_end) ])
                        cell_indexes = range(0,len(cell_indexes)) # go to 0 to tot instead of jumping indexes, to be consistent with the spiketrains from now on
                    else:
                        for spiketrain in data.spiketrains:
                            spiketrains.append(spiketrain[ (spiketrain>=timeslice_start) & (spiketrain<=timeslice_end) ])

                # return list for param search
                scores = []

                ###################################
                if 'Events_Clustering' in params['Analysis'] and params['Analysis']['Events_Clustering'] and 'spikes' in rec:
                    if not key in params['Analysis']['Events_Clustering']:
                        continue
                    if trial['name'] not in params['Analysis']['Events_Clustering'][key]['state']:
                        continue
                    print('Events_Clustering')

                    # spiketrains = select_spikelist( spiketrains=spiketrains, edge=edge, limits=params['Analysis']['Events_Clustering'][key]['limits'] )
                    print("number of spiketrains:", len(spiketrains))
                    # print("spiketrains interval:", spiketrains[0])

                    frame_duration= 1 # ms
                    sub_sampling= 1

                    # consider additional populations
                    if 'add' in params['Analysis']['Events_Clustering'][key]:
                        for added_pop in params['Analysis']['Events_Clustering'][key]['add']:
                            # get data
                            print("add spiketrains from file:",added_pop+addon+'_'+trial['name']+str(itrial)+'.pkl')
                            add_neo = pickle.load( open(folder+'/'+added_pop+addon+'_'+trial['name']+str(itrial)+'.pkl', "rb") )
                            add_data = add_neo.segments[0]
                            for added_spiketrain in add_data.spiketrains:
                                spiketrains.append(added_spiketrain[ (added_spiketrain>=timeslice_start) & (added_spiketrain<=timeslice_end) ])
                            print("number of spiketrains with "+added_pop+":", len(spiketrains))

                    # Calcium imaging-like analysis
                    #
                    # this analysis is globally as in MillerAyzenshtatCarrilloYuste2014
                    # but contains optimisations suggested by Anton Filipchuk in ... :

                    print("time:", params['run_time'], "bin size:",params['Analysis']['Events_Clustering'][key]['bin'] , "bins:", (params['run_time']/(params['Analysis']['Events_Clustering'][key]['bin']) ) )
                    event_bin = params['Analysis']['Events_Clustering'][key]['bin']

                    print("1. Compute population instantaneous firing rate (bin)")
                    print("    timeslice:",timeslice_start, timeslice_end)
                    fr = firinghist(timeslice_start, timeslice_end, spiketrains, bin_size=event_bin) # ms
                    print("    population firing: {:1.2f}±{:1.2f} sp/frame".format(np.mean(fr),np.std(fr)) )
                    # Smoothed firing rate: Savitzky-Golay 1D filter
                    smoothed_fr = savgol_filter(fr, window_length=7, polyorder=3) # large window (smoother), higher polyorder (closer)
                    smoothed_fr[smoothed_fr<0.] = 0. # filtering can bring the firing below zero
                    # Baseline: Eilers-Boelens 1D filter
                    baseline_fr = baseline(smoothed_fr, l=10**9, p=0.05)  # high l (less details), high p (closer to the mean)
                    # plot firingrates
                    fig, ax = plt.subplots(figsize=(20,5))
                    x = np.arange(0,len(smoothed_fr))
                    ax.plot(smoothed_fr,linewidth=0.5, color='k', zorder=2)
                    plt.plot(baseline_fr, linewidth=0.5, color='orange', zorder=3)
                    fig.savefig(folder+'/population_firingrate_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', dpi=300, transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()

                    print("... generating surrogates to establish population event threshold")
                    # computing ISIs of the original spiketrains
                    spiketrainsISI = []
                    cell_firing_rate = []
                    for st in spiketrains:
                        cell_firing_rate.append(len(st)/int(timeslice_end-timeslice_start)) # cell firing rate
                        spiketrainsISI.append( np.diff( st ) )
                    print("    cell firing rate: {:1.5f}±{:1.5f} sp/s".format(np.mean(cell_firing_rate),np.std(cell_firing_rate)) )

                    # reshuffle ISIs (100) times
                    surrogate_fr = []
                    for isur in range(100):
                        # build surrogate rasterplot
                        surrogate_spiketrains = []
                        for isi in spiketrainsISI:
                            random.shuffle(isi) # in-place function
                            surrogate_spiketrains.append( np.cumsum(isi) )
                        # 2.3 compute the population instantaneous firing rate for each surrogate timebinned rasterplot
                        surrogate_fr.append( firinghist(timeslice_start, timeslice_end, surrogate_spiketrains, bin_size=event_bin) )

                    print("3. Find population events in the trial")

                    # instantaneous threshold is the 95% of the surrogate population instantaneous firing rate
                    event_threshold = np.percentile(np.array(surrogate_fr), 95) + baseline_fr # 95% significance threshold

                    # the maximal extrema (beyond threshold) are peaks of population events
                    peaks = []
                    for peak in signal.argrelextrema(smoothed_fr, np.greater)[0]:
                        if smoothed_fr[peak] < event_threshold[peak]:
                            continue # ignore peaks below threshold
                        peaks.append(peak)

                    # the minimal extrema are potential limits of population events
                    minima = signal.argrelextrema(smoothed_fr, np.less, order=2)[0]
                    # there can be periods when the firing rate is 0, to be considered minima as well
                    zerominima = np.where(smoothed_fr == 0)[0]
                    minima = np.concatenate((minima,zerominima))
                    minima.sort(kind='mergesort')

                    # the minimum before and after a peak are taken as start and end times of the population event
                    events = []
                    low_toBsaved = []
                    # each peak (beyond threshold) is part of an event
                    for peak in peaks:
                        event = {'start':0, 'end':0} # init
                        # the minima (either below or above threshold) before and after the peak are the real limits of the event
                        # start
                        for minimum in minima:
                            if minimum < peak: # continue until peak
                                event['start'] = minimum
                                continue
                            break # last assigned before peak is taken
                        low_toBsaved.append(event['start'])
                        # end
                        for minimum_idx,minimum in enumerate(minima):
                            if minimum <= peak: # continue to peak
                                continue
                            event['end'] = minima[minimum_idx] # take the one beyond
                            break
                        low_toBsaved.append(event['end'])
                        if (event['start']>=0 and event['end']>0) and (event['end']-event['start']>1):
                            events.append(event)
                    # remove duplicates due to minima above threshold
                    evti = 0
                    while evti < len(events):
                        evtj = evti + 1
                        while evtj < len(events):
                            if events[evti] == events[evtj]:
                                del events[evtj]
                            else:
                                evtj += 1
                        evti += 1

                    # removal of unnecessary minima
                    minima = sorted(low_toBsaved)

                    # plot everything so far, surrogates, original, threshold, maxima and minima, event
                    fig, ax = plt.subplots(figsize=(20,5))
                    x = np.arange(0,len(smoothed_fr))
                    for event in events:
                        ax.axvspan(event['start'], event['end'], alpha=0.1, color='green', zorder=1)
                    ax.plot(smoothed_fr,linewidth=0.5, color='k', zorder=2)
                    ax.plot(event_threshold, linewidth=0.5, color='magenta', zorder=3)
                    plt.plot(baseline_fr, linewidth=0.5, color='orange', zorder=3)
                    ax.plot(x[minima], smoothed_fr[minima], 'wo', markersize=1, markeredgewidth=0., zorder=4)
                    ax.plot(x[peaks], smoothed_fr[peaks], 'rs', markersize=.5, zorder=4)
                    fig.savefig(folder+'/Events_population_firingrate_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', dpi=300, transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()

                    if len(events)<4:
                        print("... not enough events to perform clustering")
                        continue

                    print("... signatures of population events", len(events))
                    # produce a cell id signature of each population event event: string of cell ids firing during the event event
                    # get the cell ids for each event
                    events_signatures = [] # N x M, N events and all M idx for each event (0s and 1s)
                    events_spectrums = [] # N x Z, N events and only Z cids for each event
                    toBremoved = []
                    for event in events:
                        # signature
                        signature = [0 for i in range(len(spiketrains))] # init
                        spectrum = []
                        # start end of population event index along the smoothed fr are converted to ms in order to search cell ids being active in that window
                        tstart = event['start'] * (frame_duration*sub_sampling) + timeslice_start
                        tend = event['end'] * (frame_duration*sub_sampling) + timeslice_start
                        for idx,(id,spiketrain) in enumerate(zip(cell_ids,spiketrains)):
                            # take the idx if there are spikes within event start and end
                            spiketrain = np.array(spiketrain)
                            # choose idx based on event limits
                            if np.any(np.logical_and(spiketrain>=tstart, spiketrain<tend)):
                                spectrum.append( id ) # to store the actual cid and not just the index
                                signature[idx] = 1 # the signature goes with the spiketrain
                        # check that the event signature has more than 1 cell
                        if np.count_nonzero(spectrum)>1:
                            events_spectrums.append( spectrum )
                        if np.count_nonzero(signature)>1:
                            events_signatures.append( signature )
                        else:
                            toBremoved.append(events.index(event))
                    # removing events with just one cell
                    for index in sorted(toBremoved, reverse=True):
                        del events[index]

                    print("    number of events:",len(events))

                    events_signatures = np.array(events_signatures)
                    events_spectrums = np.array(events_spectrums)
                    # print(events_signatures[0].tolist())

                    # Population events number of events per sec
                    events_sec = len(events_signatures)/timeslice_end

                    # events durations and intervals statistics
                    events_durations = []
                    events_intervals = []
                    last_event = None
                    for event in events:
                        events_durations.append(event['end']-event['start'])
                        if last_event: # only from the second event on
                            events_intervals.append(event['start']-last_event['end'])
                        last_event = event

                    # Population events median+std Duration
                    print("    event durations")
                    events_durations_f = np.array(events_durations, dtype=np.float)
                    events_durations_f = events_durations_f*(0.001*event_bin) # events are computed over the binned fr, itself in ms
                    np.save(folder+'/events_durations_'+key+addon+'_'+trial['name']+str(itrial)+'.npy', events_durations_f)
                    fig = plt.figure()
                    # ev_weights = np.ones_like(events_durations_f)/float(len(events_durations_f))
                    bin_heights, bin_borders, _ = plt.hist(events_durations_f)
                    bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2
                    curve_centers=[]
                    curve_heights=[]
                    for center,height in zip(bin_centers,bin_heights):
                        if height>0:
                            curve_centers.append(center)
                            curve_heights.append(height)
                    plt.plot(curve_centers,curve_heights)
                    plt.xlim([0.01,1])
                    plt.yscale('log')
                    # plt.xscale('log')
                    lims = plt.ylim()
                    plt.vlines([np.median(events_durations_f)], ymin=lims[0], ymax=lims[1], linestyles='dashed', colors='k')
                    plt.title("Overall events duration: %.3f (%.3f)" % (np.median(events_durations_f), np.std(events_durations_f)) )
                    plt.ylabel('event occurrences')
                    plt.xlabel('event duration (s)')
                    fig.savefig(folder+'/Events_durations_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    fig.savefig(folder+'/Events_durations_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()

                    # Population events size (actual number of cells per event)
                    print("    event size (actual)")
                    events_size = []
                    for esignature in events_signatures:
                        events_size.append(len(np.nonzero(esignature)[0]))
                    np.save(folder+'/events_size_'+key+addon+'_'+trial['name']+str(itrial)+'.npy', events_size)
                    fig = plt.figure()
                    bin_heights, bin_borders, _ = plt.hist(events_size, bins='auto')
                    bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2
                    curve_centers=[]
                    curve_heights=[]
                    for center,height in zip(bin_centers,bin_heights):
                        if height>=1:
                            curve_centers.append(center)
                            curve_heights.append(height)
                    plt.plot(curve_centers,curve_heights)
                    plt.yscale('log')
                    plt.xscale('log')
                    plt.xlim([1,10000])
                    lims = plt.ylim()
                    plt.vlines([np.median(events_size)], ymin=lims[0], ymax=lims[1], linestyles='dashed', colors='k')
                    plt.title("Events size: %.3f (%.3f)" % (np.median(events_size), np.std(events_size)) )
                    plt.ylabel('occurrences')
                    plt.xlabel('number of cells per event')
                    fig.savefig(folder+'/Events_size_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    fig.savefig(folder+'/Events_size_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()

                    print("    event size (ratio to total number of cells)")
                    # ratio of cells participating to events with respect to total
                    events_size_ratio = np.array(events_size)/len(spiketrains)
                    np.save(folder+'/events_size_ratio_'+key+addon+'_'+trial['name']+str(itrial)+'.npy', events_size_ratio)
                    fig = plt.figure()
                    # bin_heights, bin_borders, _ = plt.hist(events_size_ratio, bins='auto')
                    bin_heights, bin_borders, _ = plt.hist(events_size_ratio, bins=40, range=(0.01,1))
                    bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2
                    curve_centers=[]
                    curve_heights=[]
                    for center,height in zip(bin_centers,bin_heights):
                        if height>=1:
                            curve_centers.append(center)
                            curve_heights.append(height)
                    plt.plot(curve_centers,curve_heights)
                    plt.yscale('log')
                    # plt.xscale('log')
                    plt.xlim([0.01,1])
                    lims = plt.ylim()
                    plt.vlines([np.median(events_size_ratio)], ymin=lims[0], ymax=lims[1], linestyles='dashed', colors='k')
                    plt.title("Events size ratio: %.3f (%.3f)" % (np.median(events_size_ratio), np.std(events_size_ratio)) )
                    plt.ylabel('occurrences')
                    plt.xlabel('ratio of cells per event')
                    fig.savefig(folder+'/Events_size_ratio_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    fig.savefig(folder+'/Events_size_ratio_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()

                    # ------------------------------------------------------------------

                    print("... Similarity of events")
                    # Raw Pearson's correlation matrix over events signatures (their patterns of cells)
                    SimilarityMap = np.corrcoef(events_signatures)
                    plt.set_cmap(viridis)
                    fig = plt.figure()
                    plt.pcolormesh(SimilarityMap)
                    cbar = plt.colorbar()
                    fig.savefig(folder+'/Events_CorrMatrix_'+key+addon+'_'+trial['name']+str(itrial)+'.png', dpi=600, transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()

                    # 4.2 perform clustering linkage by complete cross-correlation of event signatures
                    print("... clustering")
                    print("    linkage")
                    Z = linkage(events_signatures, method='complete', metric='correlation') #
                    Z[ Z<0 ] = 0 # for very low correlations, negative values can result
                    cut_off = 0.9*max(Z[:,2]) # generic cutoff more allowing than matlab (0.7), but we bootstrap below

                    print("    surrogate events signatures for clustering threshold")
                    # threshold for cluster significance
                    # generate 100 surrogates events_signatures
                    # correlate and cluster...
                    # there will be clusters, happening just by chance due to the finite number of cells
                    # but their internal correlation should not be high
                    # the 95% correlation of this random cluster will be the threshold
                    # for a cluster to be significantly correlated
                    surrogate_reproducibility_list = []
                    for csur in range(100):
                        surrogate_events_signatures = []
                        for evsig in events_signatures:
                            surrogate_signature = np.array([0 for i in cell_indexes]) # init
                            surrogate_signature[ np.random.choice(cell_indexes, size=np.count_nonzero(evsig), replace=False) ] = 1
                            surrogate_events_signatures.append(surrogate_signature.tolist())
                        # similarity
                        surrogate_similaritymap = np.corrcoef(surrogate_events_signatures)
                        # clustering
                        surrogate_Z = linkage(surrogate_events_signatures, method='complete', metric='correlation') #
                        surrogate_Z[ surrogate_Z<0 ] = 0 # for very low correlations, negative values can result
                        surrogate_events_assignments = fcluster(surrogate_Z, t=cut_off, criterion='distance')
                        # sorting by cluster
                        surrogate_permutation = [x for _, x in sorted(zip(surrogate_events_assignments, range(len(surrogate_events_signatures))))]
                        clustered_surrogate_similaritymap = surrogate_similaritymap[surrogate_permutation] # x
                        clustered_surrogate_similaritymap = clustered_surrogate_similaritymap[:,surrogate_permutation] # y
                        # cluster reproducibility
                        surrogate_events_cluster_sequence = sorted(surrogate_events_assignments) # [ 1 1 1 1 2 2 3 3 3 3 3 ...]
                        starti = 0
                        last_sass = surrogate_events_cluster_sequence[0]
                        for endi,sass in enumerate(surrogate_events_cluster_sequence):
                            if sass!=last_sass or endi==len(surrogate_events_cluster_sequence)-1:
                                # get sub-array
                                surrogate_cluster_subarray = np.array(clustered_surrogate_similaritymap[ starti:endi-1, starti:endi-1 ])
                                if surrogate_cluster_subarray.size:
                                    np.fill_diagonal(surrogate_cluster_subarray, 0.0)
                                    # compute the subarray average
                                    if len(surrogate_cluster_subarray)>0:
                                        surrogate_reproducibility_list.append( np.nanmean(surrogate_cluster_subarray) )
                                starti = endi
                                last_sass = sass

                    # statistically significant reproducibility
                    cluster_reproducibility_threshold = np.percentile(np.array(surrogate_reproducibility_list), 95)
                    print("    cluster reproducibility threshold:",cluster_reproducibility_threshold)
                    # number of events in a cluster, even small clusters as long as they pass the reproducibility threshold
                    cluster_size_threshold = 2 # minimum requirement
                    print("    cluster size threshold:",cluster_size_threshold)

                    # clusters
                    events_assignments = fcluster(Z, t=cut_off, criterion='distance')
                    # print(events_assignments)
                    # print(len(events_assignments))
                    # [4 4 4 4 2 2 34 4 7 7 7 7 5 5 5 5 4 ... ]

                    # count clusters to ensure each cluster has its own color
                    nevents_clusters = np.unique(events_assignments, return_counts=True)[1]
                    # print("    events/clusters:", nevents_clusters)
                    # [ 39  19  11  70  15  74  45  13  10  45 ...]
                    nclusters = len(nevents_clusters)
                    print("    #clusters:",nclusters)

                    # color map of the clustered events
                    cmap = mpcm.get_cmap('gist_rainbow')
                    cluster_color_array = [mpl.colors.rgb2hex(rgba) for rgba in cmap(np.linspace(0.0, 1.0, nclusters))]
                    random.shuffle(cluster_color_array)
                    cluster_color_array = np.array(cluster_color_array)
                    # print("cluster colors:",len(cluster_color_array))

                    threshold_map = nevents_clusters < cluster_size_threshold
                    print("    below size threshold:", np.count_nonzero(threshold_map))
                    cluster_color_array[threshold_map] = 'gray' # or 'none'
                    print("    #clusters:",len(np.unique(cluster_color_array))-1)

                    color_array = []
                    for cluidx in events_assignments:
                        color_array.append(cluster_color_array[cluidx-1]) # fcluster returns numeric labels from 0 to nclusters-1
                    color_array = np.array(color_array)
                    # color_array = ['#304030', '#008c85', '#008c85', '#005955', 'gray', 'gray', '#2db3ac', ...]
                    events_color_assignments = np.copy(color_array)

                    print("    sorting events signatures by cluster")
                    permutation = [x for _, x in sorted(zip(events_assignments, range(len(events_signatures))))]
                    # print(len(permutation))
                    # print(permutation)
                    clustered_signatures = events_signatures[permutation]
                    clustered_spectrums = events_spectrums[permutation]
                    color_array = color_array[permutation] # new color array
                    # print(color_array.tolist())
                    events = np.array(events)
                    clustered_events = events[permutation]
                    # print(clustered_events)
                    np.save(folder+'/clustered_signatures_'+key+addon+'_'+trial['name']+str(itrial)+'.npy', clustered_signatures)

                    # reordering SimilarityMap
                    clustered_SimilarityMap = SimilarityMap[permutation] # x
                    clustered_SimilarityMap = clustered_SimilarityMap[:,permutation] # y

                    # Pattern reproducibility - Cluster self-similarity
                    reproducibility_list = [ 0. for elc in cluster_color_array ]
                    core_reproducibility = {elc:1. for elc in cluster_color_array}
                    events_cluster_sequence = sorted(events_assignments) # [ 1 1 1 1 2 2 3 3 3 3 3 ...]
                    starti = 0
                    last_sass = events_cluster_sequence[0]
                    for endi,sass in enumerate(events_cluster_sequence):
                        if last_sass!=sass or endi==len(events_cluster_sequence)-1:
                            # get sub-array
                            cluster_subarray = np.array(clustered_SimilarityMap[ starti:endi-1, starti:endi-1 ])
                            if cluster_subarray.size:
                                np.fill_diagonal(cluster_subarray, 0.0)
                                # compute the subarray average
                                reproducibility_list[last_sass-1] = np.nanmedian(cluster_subarray)
                                # overwrite color
                                if np.nanmean(cluster_subarray) < cluster_reproducibility_threshold:
                                    cluster_color_array[last_sass-1] = "gray"
                                else:
                                    # Stimulus-free method to detect core neurons:
                                    # within each cluster of events,
                                    # cores are those participating to more than 95% of cluster events
                                    core_reproducibility[cluster_color_array[last_sass-1]] = np.percentile(cluster_subarray, 95)
                            starti = endi
                            last_sass = sass
                    # remove NaN
                    reproducibility_list = np.array(reproducibility_list)[~np.isnan(reproducibility_list)]
                    np.save(folder+'/reproducibility_list_'+key+addon+'_'+trial['name']+str(itrial)+'.npy', reproducibility_list)

                    # plot all
                    plt.set_cmap(viridis)
                    fig, ax = plt.subplots()
                    plt.pcolormesh(clustered_SimilarityMap)
                    # loop over zip(clustered_events,color_array) or over events_assignments
                    clcoord = 0
                    for csize,ccolor,reproval in zip(nevents_clusters,cluster_color_array,reproducibility_list):
                        rect = patches.Rectangle((clcoord, clcoord), csize, csize, linewidth=0.5, edgecolor=ccolor, facecolor='none')
                        ax.add_patch(rect)
                        # ax.text(clcoord+1, clcoord+1, "{:1.2f}".format(reproval), color=ccolor, fontsize=2)
                        clcoord += csize
                    cbar = plt.colorbar()
                    cbar.outline.set_visible(False)
                    cbar.set_ticks([]) # remove all ticks
                    for spine in plt.gca().spines.values(): # get rid of the frame
                        spine.set_visible(False)
                    plt.xticks([]) # remove all ticks
                    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                    fig.savefig(folder+'/Events_CorrClustered_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True, dpi=600)
                    plt.close()
                    fig.clf()

                    # Pattern reproducibility by Cluster
                    fig = plt.figure()
                    for repri, (reprv, reprc) in enumerate(zip(reproducibility_list,cluster_color_array)):
                        plt.bar(repri, reprv, 1., facecolor=reprc)
                    plt.title('Pattern reproducibility')
                    plt.ylabel('Auto-correlation')
                    plt.xlabel('Clusters')
                    fig.savefig(folder+'/Pattern_reproducibility_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    fig.savefig(folder+'/Pattern_reproducibility_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()
                    # Self-similarity index
                    fig = plt.figure()
                    # bin_heights, bin_borders, _ = plt.hist(reproducibility_list, bins='auto')
                    bin_heights, bin_borders, _ = plt.hist(reproducibility_list, bins=10)
                    bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2
                    curve_centers=[]
                    curve_heights=[]
                    for center,height in zip(bin_centers,bin_heights):
                        if height>0:
                            curve_centers.append(center)
                            curve_heights.append(height)
                    np.save(folder+'/reproducibility_curve_'+key+addon+'_'+trial['name']+str(itrial)+'.npy', curve_heights)
                    plt.plot(curve_centers,curve_heights)
                    lims = plt.ylim()
                    plt.vlines([np.mean(reproducibility_list)], ymin=lims[0], ymax=lims[1], linestyles='dashed', colors='k')
                    plt.title("Self-Similarity Index: %.3f (%.3f)" % (np.mean(reproducibility_list), np.std(reproducibility_list)) )
                    plt.xlabel('Auto-correlation')
                    plt.ylabel('Cluster occurrences')
                    fig.savefig(folder+'/Pattern_reproducibility_'+key+addon+'_'+trial['name']+str(itrial)+'_SelfSimilarityIndex.png', transparent=True)
                    fig.savefig(folder+'/Pattern_reproducibility_'+key+addon+'_'+trial['name']+str(itrial)+'_SelfSimilarityIndex.svg', transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()

                    print("... finding cluster cores")
                    # print(color_array.shape)
                    # print(clustered_spectrums.shape)
                    clusters_cores = []
                    clusters_cores_by_color = {ecolor:[] for ecolor in color_array}
                    cluster_color_cores = [[] for clsp in clustered_spectrums]
                    currentcl = color_array[0]
                    cluster_events_list = []
                    for cl_idlist, cl_color in zip(clustered_spectrums, color_array):
                        if cl_color=='gray':
                            continue
                        # when the color changes, plot the map and reset containers
                        if currentcl != cl_color:
                            # find common subset of cells in a clusters
                            cid_counter = {}
                            for event_cids in cluster_events_list:
                                for cidc in event_cids:
                                    if cidc in cid_counter:
                                        cid_counter[cidc] += 1/len(cluster_events_list)
                                    else: # create
                                        cid_counter[cidc] = 1/len(cluster_events_list)
                            cluster_core = []
                            for cidkey,cidprob in cid_counter.items():
                                # Cores identification, independent of stimuli
                                # A cell is considrered 'core' of multiple events if it has
                                # a frequence of occurrence > 95% of its cluster reproducibility
                                if cidprob > core_reproducibility[currentcl]:
                                    cluster_core.append(cidkey)
                            clusters_cores.append(cluster_core)
                            clusters_cores_by_color[currentcl] = cluster_core
                            # reset containers
                            cluster_events_list = []
                            currentcl = cl_color
                        # while the color is the same, append the idx to the current ensemble
                        cluster_events_list.append( cl_idlist )

                    np.save(folder+'/clusters_cores_'+key+addon+'_'+trial['name']+str(itrial)+'.npy', clusters_cores)

                    print("    gathering cores from all clusters ...")
                    core_indexes = []
                    other_indexes = []
                    for dyn_core in clusters_cores:
                        core_indexes.extend( [cell_ids.tolist().index(strid) for strid in dyn_core] )
                    core_indexes = np.unique(core_indexes)
                    print("    # cores:",len(core_indexes))
                    other_indexes = [i for i in cell_indexes if i not in core_indexes]
                    print("    # non-cores:",len(other_indexes))

                    print("    plotting single events rasterplots ...")
                    source_target_cidx = []
                    source_target_color = []
                    core_time_positions = [0, 0, 0] # beginning, middle, end
                    other_time_positions = [0, 0, 0]
                    cores_counts = []
                    others_counts = []
                    for clidx, (cluster_cids, ecolor) in enumerate(zip(clustered_spectrums, color_array)):
                        if ecolor=='gray':
                            continue

                        event_id = events_color_assignments.tolist().index(ecolor) # take the first event of this cluster
                        event = events[event_id]
                        # take start and end of the ensemble
                        estart = (event['start'] * frame_duration )
                        eend = (event['end'] * frame_duration )
                        # print(ecolor,':', estart,eend)
                        # print(cluster_cids)
                        # slice cluster cells spiketrains to the start,end interval
                        event_spiketrains = []
                        event_cidxs = []
                        for cid in cluster_cids:
                            cidx = cell_ids.tolist().index(cid)
                            train = spiketrains[cidx]
                            if np.array(train[(train>=estart)*(train<=eend)]).size>0:
                                event_spiketrains.append( train[(train>=estart)*(train<=eend)] )
                                event_cidxs.append( cidx )
                        # print(event_spiketrains)
                        if len(event_spiketrains) < np.mean(event_threshold):
                            continue

                        # sort them based on the fitst element of each and sasve also the last for flow analysis
                        sorted_event_cidx = [cidx for _,cidx in sorted(zip(event_spiketrains, event_cidxs), key=lambda ez: ez[0][0])] # sort based on the first element of zip
                        source_target_cidx.append([sorted_event_cidx[0], sorted_event_cidx[-1]]) # take beginning and end cidx
                        source_target_color.append(ecolor)
                        # sort them based on the fitst element of each
                        event_spiketrains = sorted(event_spiketrains, key=lambda etrain: etrain[0])
                        # print(event_spiketrains)

                        # print("    plotting spike rasterplot for event from cluster ",ecolor,':', estart,eend)
                        core_count = 0
                        other_count = 0
                        fig = plt.figure()
                        for row,train in enumerate(event_spiketrains):
                            ccol = 'gray'
                            # beginning is the first 1/3rd of the event
                            if np.array(train[(train>=estart)*(train<estart+(eend-estart)/3)]).size>0:
                                other_time_positions[0]+=1
                            # middle is everything in between
                            if np.array(train[(train>estart+(eend-estart)/3)*(train<eend-(eend-estart)/3)]).size>0:
                                other_time_positions[1]+=1
                            # end is the last 1/3rd of the event
                            if np.array(train[train>=eend-(eend-estart)/3]).size>0:
                                other_time_positions[2]+=1
                            # Cores
                            if row in core_indexes:
                                core_count +=1
                                ccol = 'g'
                                if np.array(train[(train>=estart)*(train<estart+(eend-estart)/3)]).size>0:
                                    core_time_positions[0]+=1
                                if np.array(train[(train>estart+(eend-estart)/3)*(train<eend-(eend-estart)/3)]).size>0:
                                    core_time_positions[1]+=1
                                if np.array(train[train>=eend-(eend-estart)/3]).size>0:
                                    core_time_positions[2]+=1
                            else:
                                other_count +=1
                            plt.scatter( train, [row]*len(train), marker='|', facecolors=ccol, s=150, linewidth=3 )
                        cores_counts.append(core_count)
                        others_counts.append(other_count)
                        plt.ylabel("cell IDs")
                        plt.xlabel("time (s)")
                        fig.savefig(folder+'/rasterplot_'+str(clidx)+'.svg', transparent=False, dpi=300)
                        plt.close()
                        fig.clear()
                        fig.clf()
                    np.save(folder+'/source_target_cidx_'+key+'.npy', source_target_cidx)
                    np.save(folder+'/source_target_color_'+key+'.npy', source_target_color)

                    print("... Cores occurrence position in time during the event")
                    # plot
                    # print(core_time_positions)
                    # print(other_time_positions)
                    fig = plt.figure()
                    plt.bar(np.array(range(len(core_time_positions)))+0.15, other_time_positions, color='gray')
                    plt.bar(range(len(core_time_positions)), core_time_positions, color='green')
                    plt.ylabel('count')
                    plt.xlabel('core positions')
                    fig.savefig(folder+'/core_time_positions.svg', transparent=True)
                    plt.close()
                    fig.clear()
                    fig.clf()


                # -----------------------------------------------------
                # for systems with low memory :)
                if removeDataFile:
                    os.remove(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+'.pkl')

                print("scores",key,":",scores)

    return scores # to fix: is returning only the last scores!








###############################################
# ADDITIONAL FUNCTIONS

# Finds baseline in the firing rate
# Params:
#   l for smoothness (λ)
#   p for asymmetry
# Both have to be tuned to the data at hand.
# We found that generally is a good choice (for a signal with positive peaks):
#   10^2 ≤ l ≤ 10^9
#   0.001 ≤ p ≤ 0.1
# but exceptions may occur.
# In any case one should vary l on a grid that is approximately linear for log l
def baseline(y, l, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = l * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def select_spikelist( spiketrains, edge=None, limits=None ):
    new_spiketrains = []
    for i,st in enumerate(spiketrains):

        # reject cells outside limits
        if limits:
            # use st entries as indexes to put ones
            x = int(i % edge)
            y = int(i / edge)
            # print(i, x,y)
            # if (x>10 and x<50) and (y>10 and y<54):
            if (x>limits[0][0] and x<limits[0][1]) and (y>limits[1][0] and y<limits[1][1]):
                # print('taken')
                new_spiketrains.append( st )
        else:
            new_spiketrains.append( st )
    return new_spiketrains


def firingrate( start, end, spiketrains, bin_size=10 ):
    """
    Population instantaneous firing rate
    as in https://neuronaldynamics.epfl.ch/online/Ch7.S2.html
    """
    if spiketrains == [] :
        return NaN
    # create bin edges based on start and end of slices and bin size
    bin_edges = np.arange( start, end, bin_size )
    # print("bin_edges",bin_edges.shape)
    # binning total time, and counting the number of spike times in each bin
    hist = np.zeros( bin_edges.shape[0]-1 )
    for spike_times in spiketrains:
        hist = hist + np.histogram( spike_times, bin_edges )[0]
    return ((hist / len(spiketrains) ) / bin_size ) * 1000 # average over population; result in ms *1000 to have it in sp/s


def firinghist( start, end, spiketrains, bin_size=10 ):
    if len(spiketrains)==0:
        return NaN
    # create bin edges based on start and end of slices and bin size
    bin_edges = np.arange( start, end, bin_size )
    # binning total time, and counting the number of spike times in each bin
    hist = np.zeros( bin_edges.shape[0]-1 )
    for spike_times in spiketrains:
        hist += np.histogram( spike_times, bin_edges )[0]
    return hist # no average over population and bin
