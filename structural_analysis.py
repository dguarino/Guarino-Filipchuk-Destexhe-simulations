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
import warnings
import itertools
from functools import cmp_to_key
from itertools import zip_longest # analysis
import numpy as np
import inspect

################################
import matplotlib
matplotlib.use('Agg') # to be used when DISPLAY is undefined as in Docker
################################
# matplotlib.rc('image', cmap='viridis')
# matplotlib.rc('image', cmap='Reds')
matplotlib.rc('image', cmap='summer')
################################

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import LineCollection
import matplotlib.colors as mpcolors
import matplotlib.cm as mpcm

from neo.core import AnalogSignal # analysis
import quantities as pq

# SciPy related
import scipy.signal as signal
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import cophenet
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import curve_fit

import igraph as ig
from igraph import *


# ------------------------------------------------
# Additional classes and functions
from helper_functions import *
# ------------------------------------------------


# frame_duration = 1
frame_duration = 30 # as in 2photon

def analyse(params, folder, addon='', removeDataFile=False):
    print("\nAnalysing data...")

    # populations key-recorders match
    populations = {}
    for popKey,popVal in params['Populations'].items():
        if popKey in params['Recorders']:
            populations[popKey] = list(params['Recorders'][popKey].keys())


    ###################################
    if 'Structure' in params['Analysis']:

        print('Structural Analysis')
        print("\nRemember that the connections should be created on a population with fill_order='sequential' to plot meaningful images.\n")

        print("... getting cell indexes and ids")
        for key,strpar in params['Analysis']['Structure'].items():
            # Get saved cell coordinates in the network
            cell_coords = []
            cell_indexes = []
            cell_ids = []
            with open(folder+'/'+key+addon+'_default0_positions.txt', 'r') as posfile: # forced reading py conns
                lines = posfile.readlines()
                posfile.close()
                for line in lines:
                    cell_coords.append( [int(float(i)) for i in line.split(' ')[:4]] ) # id, index, x, y (not including z (4))
                    # print(cell_coords) # id, idx, x, y
                    #[ ..., [13287, 4064, 63, 32], [13288, 4065, 63, 33], ... ]
                cell_coords = np.array(cell_coords)
                cell_indexes = (cell_coords[:,1]).tolist()
                cell_ids = cell_coords[:,0]

            connskey = strpar['conns'] # restrict analysis to py as in 2photon

            # # older NEST versions
            # adjacency_matrix = np.nan_to_num(np.load(folder+'/connections_'+connskey+'.npy'))
            # print("    ", connskey, "conns, shape", adjacency_matrix.shape)
            # # newer NEST version
            conns = np.load(folder+'/connections_'+connskey+'.npy')
            print("    ", connskey, "conns, shape", conns.shape)
            conns[np.isnan(conns)] = 0.0 # to avoid NaN
            # print(conns) # [idx_source, idx_target, weight]
            edges = [ [int(i[0]),int(i[1])] for i in conns ]
            print("    nodes:",np.max(edges)+1)
            adjacency_matrix = np.zeros( (np.max(edges)+1,np.max(edges)+1) )
            for conn in conns:
                adjacency_matrix[int(conn[0]),int(conn[1])] = conn[2]
            # print(adjacency_matrix.tolist())

            # print("    plotting adjacency matrix")
            # fig = plt.figure()
            # plt.pcolormesh(adjacency_matrix)
            # cbar = plt.colorbar()
            # cbar.set_label('synaptic weight', rotation=270)
            # fig.savefig(folder+'/Connections_'+connskey+'.png', transparent=True, dpi=1000)
            # plt.close()
            # fig.clear()

            # distances distribution
            # cell_coords
            print("    computing inter-soma distances")
            connection_distance = []
            for prei,synpre in enumerate(adjacency_matrix):
                pre_coord = cell_coords[prei]
                for posti,synpost in enumerate(adjacency_matrix[prei]):
                    if synpost>0.:
                        post_coord = cell_coords[posti]
                        connection_distance.append( np.linalg.norm(np.array(pre_coord)-np.array(post_coord)) ) # distance
            # print(connection_distance)
            connection_distance = np.array(connection_distance)*5*1000 # connversion to EM data (5um to nm)
            print("    connection distances:",stats.describe(connection_distance))
            # plot
            # res = stats.bootstrap((connection_distance,), np.std, confidence_level=0.95, n_resamples=10000)
            bin_heights, bin_borders, _ = plt.hist(connection_distance, bins=np.logspace(2,6,10,endpoint=True))
            fig = plt.figure()
            xs = np.logspace(2,6,len(bin_heights),endpoint=True)
            plt.plot(xs, bin_heights, c='g')
            # plt.fill_between(xs, (bin_heights-res.confidence_interval[0]), (bin_heights+res.confidence_interval[1]), color='r', alpha=.2)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('norm occurrences')
            plt.xlabel('distance (nm)')
            fig.savefig(folder+'/distance_distribution_'+connskey+'.png', transparent=True)
            fig.savefig(folder+'/distance_distribution_'+connskey+'.svg', transparent=True)
            plt.close()
            fig.clear()
            fig.clf()

            print("... loading cores")
            corefile = folder+"/clusters_cores_py_default0.npy"
            print("    reading file: " + corefile)
            clusters_cores = np.load(corefile)

            core_indexes = []
            other_indexes = []
            for dyn_core in clusters_cores:
                core_indexes.extend( [cell_ids.tolist().index(strid) for strid in dyn_core] )
            core_indexes = np.unique(core_indexes)
            print("    # cores:",len(core_indexes))
            other_indexes = [i for i in range(cell_ids.size) if i not in core_indexes]
            print("    # non-cores:",len(other_indexes))

            print('... creating network')
            # dgraph = ig.Graph.Adjacency(adjacency_matrix.tolist()) # old NEST version
            dgraph = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist(), mode='directed')

            # # graw the graph
            # ig.plot(dgraph, folder+'/ring_'+key+'.png', layout=dgraph.layout("circle"), edge_curved=0., edge_color='#000', edge_width=0.5, edge_arrow_size=0.1, vertex_size=5, vertex_color='#000', margin=50)
            # # ig.plot(dgraph, folder+'/ring_'+key+'.svg', layout=dgraph.layout("circle"), edge_curved=0., edge_color='#000', edge_width=0.5, edge_arrow_size=0.1, vertex_size=5, vertex_color='#000', margin=50)

            print("    number of vertices:", dgraph.vcount())
            print("    number of edges:", dgraph.ecount())
            print("    density of the graph:", 2*dgraph.ecount()/(dgraph.vcount()*(dgraph.vcount()-1)))

            print('... Network nodes degrees')
            degrees = np.array(dgraph.degree())
            print(degrees)
            np.save(folder+'/degrees_'+connskey+'.npy', degrees)

            # res = stats.bootstrap((degrees,), np.std, confidence_level=0.95, n_resamples=10000)
            bin_heights, bin_borders, _ = plt.hist(degrees, bins=np.logspace(0,5.,10,endpoint=True))
            fig = plt.figure()
            xs = np.logspace(0,5,len(bin_heights),endpoint=True)
            plt.plot(xs, bin_heights, c='g')
            # plt.fill_between(xs, (bin_heights-res.confidence_interval[0]), (bin_heights+res.confidence_interval[1]), color='r', alpha=.2)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('norm occurrences')
            plt.xlabel('degrees')
            fig.savefig(folder+'/degree_distribution_CI_'+connskey+'.png', transparent=True)
            fig.savefig(folder+'/degree_distribution_CI_'+connskey+'.svg', transparent=True)
            plt.close()
            fig.clear()
            fig.clf()

            # https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#degree
            in_degrees = ig.Graph.degree(dgraph, mode='in', loops=False)
            np.save(folder+'/in_degrees_'+connskey+'.npy', in_degrees)
            out_degrees = ig.Graph.degree(dgraph, mode='out', loops=False)
            np.save(folder+'/out_degrees_'+connskey+'.npy', out_degrees)
            # print("in", in_degrees, len(in_degrees))
            # print("out", out_degrees, len(out_degrees))

            # # Clustering Coefficient of only excitatory cells
            # print('... Local Clustering Coefficient')
            #
            # # from igraph
            # dgraph.to_undirected(mode='mutual', combine_edges='ignore')
            # local_clustering_coefficients = np.array(dgraph.transitivity_local_undirected(vertices=None, mode="zero"))
            # print("min", np.min(local_clustering_coefficients))
            # print("max", np.max(local_clustering_coefficients))
            # print("mean", np.mean(local_clustering_coefficients))
            # np.save(folder+'/local_clustering_coefficients_'+connskey+'.npy', local_clustering_coefficients)
            #
            # # Hierarchical modularity as in SadovskyMacLean2013
            # # demonstrated by a log-log relationship between node degree and node-clustering coefficient.
            # # Nodes are colored from orange to green corresponding to clustering coefficient, and sized according to degree.
            # fig = plt.figure()
            # summer = mpcm.summer
            # for deg,ccoef in zip(degrees,local_clustering_coefficients):
            #     plt.scatter( deg, ccoef, marker='o', facecolor='g', s=10, edgecolors='none', alpha=0.25)
            # plt.yscale('log')
            # plt.xscale('log')
            # # plt.xlim([1,400])
            # # plt.ylim([0.01,1])
            # ax = plt.gca()
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # plt.tick_params(axis='both', bottom='on', top='on', left='off', right='off')
            # plt.tight_layout()
            # fig.savefig(folder+'/hierarchical_modularity.png', transparent=True, dpi=600)
            # fig.savefig(folder+'/hierarchical_modularity.svg', transparent=True)
            # plt.close()
            # fig.clf()
            #
            # print('... Motifs')
            # print('    counting motifs')
            # # number of motifs by class
            # motifs = dgraph.motifs_randesu(size=3, cut_prob=None, callback=None)
            # motifs.extend([np.nan for i in range(16-len(motifs))]) # to be broadcastable with the surrogates
            # motifs = np.array(motifs)
            # motifs[motifs==0.0] = np.nan # to avoid division by zero later
            # print("motifs",motifs)
            # # which vertices belong to which class
            # motif_vertices = {}
            # def cb_motif(graph,vs_motif,isoc):
            #     if not isoc in motif_vertices:
            #         motif_vertices[isoc] = []
            #     motif_vertices[isoc].append(vs_motif)
            # dgraph.motifs_randesu(size=3, cut_prob=None, callback=cb_motif)
            # # print(motif_vertices)
            # print('    generating surrogate networks and counting motifs')
            # # generate 100 random networks with the same
            # # number of vertices and number of edges as dgraph
            # surrogate_motifs = []
            # for isur,surrogateg in enumerate(range(100)):
            #     random.seed(isur)
            #     # https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#Degree_Sequence
            #     erg = ig.Graph.Degree_Sequence(out_degrees, in_degrees, method='simple') # two ways connections allowed
            #     # erg = ig.Graph.Degree_Sequence(out_degrees, in_degrees, method='no_multiple')
            #     surrogate_motifs.append( erg.motifs_randesu(size=3, cut_prob=None, callback=None) )
            #     # ig.plot(erg, folder+'/ring_erg.svg', layout=erg.layout("circle"), edge_curved=0., edge_color='#000', edge_width=0.1, edge_arrow_size=0.1, vertex_size=1, vertex_color='#000', margin=50)
            # surrogate_motifs = np.percentile(surrogate_motifs, 99, axis=0)
            # # print("surrogate_motifs",surrogate_motifs)
            # surrogate_motifs[surrogate_motifs==0.0] = np.nan # to avoid division by zero later
            # # surrogate motif vertices
            # surrogate_motif_vertices = {}
            # def cb_motif(graph,vs_motif,isoc):
            #     if not isoc in surrogate_motif_vertices:
            #         surrogate_motif_vertices[isoc] = []
            #     surrogate_motif_vertices[isoc].append(vs_motif)
            # # take the last surrogate network (just one for now)
            # erg.motifs_randesu(size=3, cut_prob=None, callback=cb_motif)
            # # plot
            # print(surrogate_motifs)
            # fig = plt.figure()
            # plt.bar(range(len(motifs)), motifs, color='k', label='real', zorder=10)
            # plt.bar(range(len(surrogate_motifs)), surrogate_motifs, color='orange', label='avg surrogates', zorder=1)
            # plt.legend()
            # plt.ylabel('occurrences')
            # plt.xlabel('motifs types')
            # fig.savefig(folder+'/motifs_occurrences.png', transparent=True)
            # fig.savefig(folder+'/motifs_occurrences.svg', transparent=True)
            # plt.close()
            # fig.clear()
            # fig.clf()
            # # plot
            # motifsratio = motifs/surrogate_motifs
            # # print("motifsratio",motifsratio)
            # fig = plt.figure()
            # plt.bar(range(len(motifs)), motifsratio, color='k')
            # plt.ylabel('count relative to random')
            # # plt.ylim([0,25000])
            # plt.xlabel('motifs types')
            # fig.savefig(folder+'/motifs_ratio.png', transparent=True)
            # fig.savefig(folder+'/motifs_ratio.svg', transparent=True)
            # plt.close()
            # fig.clf()
            #
            # global_structural_motif_cores = {k: 0 for k in range(16)}
            # global_structural_motif_others = {k: 0 for k in range(16)}
            #
            # print("... overlap between motifs and cluster cores")
            # # For each set of reproducible cluster cores we count their connectivity motifs.
            # set_indexes = set(cell_indexes)
            # for dyn_core in clusters_cores:
            #     dyn_core_indexes = set([cell_ids.tolist().index(strid) for strid in dyn_core])
            #     dyn_other_indexes = set_indexes.symmetric_difference(dyn_core_indexes)
            #     # print(dyn_core_indexes)
            #     for mclass, mlist in motif_vertices.items():
            #         # print("   motif class:",mclass)
            #         # print(mlist)
            #         for mtriplet in mlist:
            #             intersection_cores = len(list(dyn_core_indexes.intersection(mtriplet)))
            #             intersection_others = len(list(dyn_other_indexes.intersection(mtriplet)))
            #             global_structural_motif_cores[mclass] += intersection_cores
            #             global_structural_motif_others[mclass] += intersection_others
            #
            # fig = plt.figure()
            # plt.bar(global_structural_motif_cores.keys(), global_structural_motif_cores.values(), color='forestgreen')
            # plt.ylabel('cores occurrences')
            # plt.yscale('log')
            # plt.ylim([0.7,plt.ylim()[1]])
            # plt.xlabel('motifs types')
            # fig.savefig(folder+'/global_motifs_cores.png', transparent=True)
            # fig.savefig(folder+'/global_motifs_cores.svg', transparent=True)
            # plt.close()
            # fig.clear()
            # fig.clf()
            #
            # fig = plt.figure()
            # plt.bar(global_structural_motif_others.keys(), global_structural_motif_others.values(), color='silver')
            # plt.ylabel('non-cores occurrences')
            # plt.yscale('log')
            # plt.ylim([0.7,plt.ylim()[1]])
            # plt.xlabel('motifs types')
            # fig.savefig(folder+'/global_motifs_others.png', transparent=True)
            # fig.savefig(folder+'/global_motifs_others.svg', transparent=True)
            # plt.close()
            # fig.clear()
            # fig.clf()

            # dgraph is already defined from the structural_analysis included file
            print("    graph diameter (#vertices):", dgraph.diameter(directed=True, unconn=True, weights=None))
            print("    graph average path length (#vertices):", dgraph.average_path_length(directed=True, unconn=True))

            dgraph.vs["ophys_cell_id"] = cell_ids.tolist()

            is_id_core = np.array( [0] * len(cell_ids) )
            is_id_core[core_indexes] = 1
            dgraph.vs["is_core"] = is_id_core.tolist()

            print('... assortativity')
            # is a preference for a network's nodes to attach to others that are similar in some way
            # biological networks typically show negative assortativity, or disassortative mixing, or disassortativity, as high degree nodes tend to attach to low degree nodes.
            print("    overall:", dgraph.assortativity("is_core", types2=None, directed=True) )
            print("    overall (nominal):", dgraph.assortativity_nominal("is_core", directed=True) )
            # cores degree distro vs others degree distro
            print("    assortativity degree:", dgraph.assortativity_degree(directed=True) )

            print("    reciprocity:", dgraph.reciprocity(ignore_loops=True, mode='default') )
            # the proportion of mutual connections in a directed graph.
            # It is most commonly defined as the probability that the opposite counterpart of a directed edge is also included in the graph.

            # -------------------------------------------------
            # Centrality measures

            print('... degree centrality')
            degree_centrality_cores = dgraph.degree(core_indexes, mode='all', loops=True)
            degree_centrality_others = dgraph.degree(other_indexes, mode='all', loops=True)
            print("    cores: {:1.2f}±{:1.2f} degree".format(np.mean(degree_centrality_cores),np.std(degree_centrality_cores)) )
            print("    others: {:1.2f}±{:1.2f} degree".format(np.mean(degree_centrality_others),np.std(degree_centrality_others)) )
            # D’Agostino and Pearson’s test for normality
            # null hypothesis: x comes from a normal distribution
            alpha = 5e-2 # p=0.05
            p = 0
            for distr in [degree_centrality_cores, degree_centrality_others]:
                _, pdistr = stats.normaltest(distr)
                p += pdistr
            p /= len([degree_centrality_cores, degree_centrality_others]) # arbitrary and simple
            if p < alpha:
                print("    Non-normal distribution: Kruskal-Wallis will be performed.")
                kwstat,pval = stats.kruskal(degree_centrality_cores, degree_centrality_others)
                print("    test results:",kwstat,pval)
            else:
                print("    Normal distribution: One-way ANOVA will be performed.")
                kwstat,pval = stats.f_oneway(degree_centrality_cores, degree_centrality_others)
                print("    test results:",kwstat,pval)
            d,_ = stats.ks_2samp(degree_centrality_cores, degree_centrality_others) # non-parametric measure of effect size [0,1]
            print('    Kolmogorov-Smirnov Effect Size: %.3f' % d)
            # all degree by type
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(degree_centrality_cores))
            plt.scatter(xs, degree_centrality_cores, alpha=0.3, c='forestgreen', edgecolors='none')
            xs = np.random.normal(2, 0.04, len(degree_centrality_others))
            plt.scatter(xs, degree_centrality_others, alpha=0.3, c='silver', edgecolors='none')
            bp = ax.boxplot([degree_centrality_cores,degree_centrality_others], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Degree')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(degree_centrality_cores)), "other\n(n={:d})".format(len(degree_centrality_others))])
            fig.savefig(folder+'/global_cores_others_degree.svg', transparent=True)
            plt.close()
            fig.clf()
            # all degree by type
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(degree_centrality_cores))
            plt.scatter(xs, degree_centrality_cores, alpha=0.3, c='forestgreen', edgecolors='none')
            xs = np.random.normal(2, 0.04, len(degree_centrality_others))
            plt.scatter(xs, degree_centrality_others, alpha=0.3, c='silver', edgecolors='none')
            bp = ax.boxplot([degree_centrality_cores,degree_centrality_others], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yscale('log')
            plt.ylabel('Degree')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(degree_centrality_cores)), "other\n(n={:d})".format(len(degree_centrality_others))])
            fig.savefig(folder+'/global_cores_others_degree_log.svg', transparent=True)
            plt.close()
            fig.clf()

            print('... betweenness')
            betweenness_centrality = np.array(dgraph.betweenness(vertices=None, directed=True, cutoff=None, weights=None))
            # print(betweenness)
            # print(len(betweenness))
            np.save(folder+'/betweenness_centrality.npy', betweenness_centrality)
            core_betweenness = betweenness_centrality[core_indexes]
            other_betweenness = betweenness_centrality[other_indexes]
            core_betweenness[core_betweenness<0.0001] = 0.0001 # for later stats and plotting
            other_betweenness[other_betweenness<0.0001] = 0.0001
            global_structural_cores_betweeness = core_betweenness
            global_structural_others_betweeness = other_betweenness
            print("    cores: {:1.2f}±{:1.2f} edges".format(np.mean(global_structural_cores_betweeness),np.std(global_structural_cores_betweeness)) )
            print("    others: {:1.2f}±{:1.2f} edges".format(np.mean(global_structural_others_betweeness),np.std(global_structural_others_betweeness)) )
            # D’Agostino and Pearson’s test for normality
            # null hypothesis: x comes from a normal distribution
            alpha = 5e-2 # p=0.05
            p = 0
            for distr in [global_structural_cores_betweeness, global_structural_others_betweeness]:
                _, pdistr = stats.normaltest(distr)
                p += pdistr
            p /= len([global_structural_cores_betweeness, global_structural_others_betweeness]) # arbitrary and simple
            if p < alpha:
                print("    Non-normal distribution: Kruskal-Wallis will be performed.")
                kwstat,pval = stats.kruskal(global_structural_cores_betweeness, global_structural_others_betweeness)
                print("    test results:",kwstat,pval)
            else:
                print("    Normal distribution: One-way ANOVA will be performed.")
                kwstat,pval = stats.f_oneway(global_structural_cores_betweeness, global_structural_others_betweeness)
                print("    test results:",kwstat,pval)
            d,_ = stats.ks_2samp(global_structural_cores_betweeness, global_structural_others_betweeness) # non-parametric measure of effect size [0,1]
            print('    Kolmogorov-Smirnov Effect Size: %.3f' % d)
            # all betweenness by type
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(global_structural_cores_betweeness))
            plt.scatter(xs, global_structural_cores_betweeness, alpha=0.3, c='forestgreen', edgecolors='none')
            xs = np.random.normal(2, 0.04, len(global_structural_others_betweeness))
            plt.scatter(xs, global_structural_others_betweeness, alpha=0.3, c='silver', edgecolors='none')
            bp = ax.boxplot([global_structural_cores_betweeness,global_structural_others_betweeness], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Betweenness')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(global_structural_cores_betweeness)), "other\n(n={:d})".format(len(global_structural_others_betweeness))])
            fig.savefig(folder+'/global_cores_others_betweenness.svg', transparent=True)
            plt.close()
            fig.clf()
            # all betweenness by type
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(global_structural_cores_betweeness))
            plt.scatter(xs, global_structural_cores_betweeness, alpha=0.3, c='forestgreen', edgecolors='none')
            xs = np.random.normal(2, 0.04, len(global_structural_others_betweeness))
            plt.scatter(xs, global_structural_others_betweeness, alpha=0.3, c='silver', edgecolors='none')
            bp = ax.boxplot([global_structural_cores_betweeness,global_structural_others_betweeness], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yscale('log')
            plt.ylabel('Betweenness')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(global_structural_cores_betweeness)), "other\n(n={:d})".format(len(global_structural_others_betweeness))])
            fig.savefig(folder+'/global_cores_others_betweenness_log.svg', transparent=True)
            plt.close()
            fig.clf()

            print('... eccentricity')
            # the shortest distance (in edges) to the vertex, to from similar vertices in the graph.
            # is the distance between cores significantly shorter than between cores and others, or between any two vertices in the graph?
            # NO, the distance between cores is longer (naturally following edge direction "in").
            # and when we consider directed edges it is significantly longer
            in_cores_eccentricity = dgraph.eccentricity(vertices=core_indexes, mode='in')
            in_others_eccentricity = dgraph.eccentricity(vertices=other_indexes, mode='in')
            print("    in cores: {:1.2f}±{:1.2f} edges".format(np.mean(in_cores_eccentricity),np.std(in_cores_eccentricity)) )
            print("    in others: {:1.2f}±{:1.2f} edges".format(np.mean(in_others_eccentricity),np.std(in_others_eccentricity)) )
            # D’Agostino and Pearson’s test for normality
            # null hypothesis: x comes from a normal distribution
            alpha = 5e-2 # p=0.05
            p = 0
            for distr in [in_cores_eccentricity, in_others_eccentricity]:
                _, pdistr = stats.normaltest(distr)
                p += pdistr
            p /= len([in_cores_eccentricity, in_others_eccentricity]) # arbitrary and simple
            if p < alpha:
                print("    Non-normal distribution: Kruskal-Wallis will be performed.")
                kwstat,pval = stats.kruskal(in_cores_eccentricity, in_others_eccentricity)
                print("    cores vs others:",kwstat,pval)
            else:
                print("    Normal distribution: One-way ANOVA will be performed.")
                kwstat,pval = stats.f_oneway(in_cores_eccentricity, in_others_eccentricity)
                print("    cores vs others:",kwstat,pval)
            d,_ = stats.ks_2samp(in_cores_eccentricity, in_others_eccentricity) # non-parametric measure of effect size [0,1]
            print('    Kolmogorov-Smirnov Effect Size: %.3f' % d)
            # all eccentricity by type
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(in_cores_eccentricity))
            plt.scatter(xs, in_cores_eccentricity, alpha=0.3, c='forestgreen', edgecolors='none')
            xs = np.random.normal(2, 0.04, len(in_others_eccentricity))
            plt.scatter(xs, in_others_eccentricity, alpha=0.3, c='silver', edgecolors='none')
            bp = ax.boxplot([in_cores_eccentricity,in_others_eccentricity], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Eccentricity')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(in_cores_eccentricity)), "other\n(n={:d})".format(len(in_others_eccentricity))])
            fig.savefig(folder+'/global_cores_others_eccentricity.svg', transparent=True)
            plt.close()
            fig.clf()

            print("... authority and hub scores")
            # what is the overlap of cores and hubs?
            # AUTHORITY
            authority_scores = np.array(dgraph.authority_score(weights=None, scale=True, return_eigenvalue=False))
            # print(authority_scores)
            auth_scores_cores = authority_scores[core_indexes]
            auth_scores_others = authority_scores[other_indexes]
            print("    cores authority score: {:1.2f}±{:1.2f} edges".format(np.mean(auth_scores_cores),np.std(auth_scores_cores)) )
            print("    others authority score : {:1.2f}±{:1.2f} edges".format(np.mean(auth_scores_others),np.std(auth_scores_others)) )
            # D’Agostino and Pearson’s test for normality
            # null hypothesis: x comes from a normal distribution
            alpha = 5e-2 # p=0.05
            p = 0
            for distr in [auth_scores_cores, auth_scores_others]:
                _, pdistr = stats.normaltest(distr)
                p += pdistr
            p /= len([auth_scores_cores, auth_scores_others]) # arbitrary and simple
            if p < alpha:
                print("    Non-normal distribution: Kruskal-Wallis will be performed.")
                kwstat,pval = stats.kruskal(auth_scores_cores, auth_scores_others)
                print("    cores vs others:",kwstat,pval)
            else:
                print("    Normal distribution: One-way ANOVA will be performed.")
                kwstat,pval = stats.f_oneway(auth_scores_cores, auth_scores_others)
                print("    cores vs others:",kwstat,pval)
            d,_ = stats.ks_2samp(auth_scores_cores, auth_scores_others) # non-parametric measure of effect size [0,1]
            print('    Kolmogorov-Smirnov Effect Size: %.3f' % d)
            # all eccentricity by type
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(auth_scores_cores))
            plt.scatter(xs, auth_scores_cores, alpha=0.3, c='forestgreen', edgecolors='none')
            xs = np.random.normal(2, 0.04, len(auth_scores_others))
            plt.scatter(xs, auth_scores_others, alpha=0.3, c='silver', edgecolors='none')
            bp = ax.boxplot([auth_scores_cores,auth_scores_others], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Hub score')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(auth_scores_cores)), "other\n(n={:d})".format(len(auth_scores_others))])
            fig.savefig(folder+'/global_cores_others_authority_score.svg', transparent=True)
            plt.close()
            fig.clf()
            # HUB
            hub_scores = np.array(dgraph.hub_score(weights=None, scale=True, return_eigenvalue=False))
            hub_scores_cores = hub_scores[core_indexes]
            hub_scores_others = hub_scores[other_indexes]
            # print(hub_scores)
            print("    cores hub score: {:1.2f}±{:1.2f} edges".format(np.mean(hub_scores_cores),np.std(hub_scores_cores)) )
            print("    others hub score : {:1.2f}±{:1.2f} edges".format(np.mean(hub_scores_others),np.std(hub_scores_others)) )
            # D’Agostino and Pearson’s test for normality
            # null hypothesis: x comes from a normal distribution
            alpha = 5e-2 # p=0.05
            p = 0
            for distr in [hub_scores_cores, hub_scores_others]:
                _, pdistr = stats.normaltest(distr)
                p += pdistr
            p /= len([hub_scores_cores, hub_scores_others]) # arbitrary and simple
            if p < alpha:
                print("    Non-normal distribution: Kruskal-Wallis will be performed.")
                kwstat,pval = stats.kruskal(hub_scores_cores, hub_scores_others)
                print("    cores vs others:",kwstat,pval)
            else:
                print("    Normal distribution: One-way ANOVA will be performed.")
                kwstat,pval = stats.f_oneway(hub_scores_cores, hub_scores_others)
                print("    cores vs others:",kwstat,pval)
            d,_ = stats.ks_2samp(hub_scores_cores, hub_scores_others) # non-parametric measure of effect size [0,1]
            print('    Kolmogorov-Smirnov Effect Size: %.3f' % d)
            # all eccentricity by type
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(hub_scores_cores))
            plt.scatter(xs, hub_scores_cores, alpha=0.3, c='forestgreen', edgecolors='none')
            xs = np.random.normal(2, 0.04, len(hub_scores_others))
            plt.scatter(xs, hub_scores_others, alpha=0.3, c='silver', edgecolors='none')
            bp = ax.boxplot([hub_scores_cores,hub_scores_others], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Hub score')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(hub_scores_cores)), "other\n(n={:d})".format(len(hub_scores_others))])
            fig.savefig(folder+'/global_cores_others_hub_score.svg', transparent=True)
            plt.close()
            fig.clf()

            print('... number of paths between cores')
            # cores activity could be sustained by indirect synaptic feedback but highly connected via secondary paths, then it's enough to have an attractor.
            core_shortestpaths = []
            for coreidx in core_indexes:
                othercores = list(core_indexes)
                othercores.remove(coreidx)
                shrtpth = dgraph.get_shortest_paths(coreidx, to=othercores, weights=None, mode='out', output='vpath')
                for strp in shrtpth:
                    core_shortestpaths.append(len(strp))
            other_shortestpaths = []
            for otheridx in other_indexes:
                otherothers = list(other_indexes)
                otherothers.remove(otheridx)
                shrtpth = dgraph.get_shortest_paths(otheridx, to=otherothers, weights=None, mode='out', output='vpath')
                for strp in shrtpth:
                    other_shortestpaths.append(len(strp))
            print("    cores: {:1.2f}±{:1.2f} shortest paths".format(np.mean(core_shortestpaths),np.std(core_shortestpaths)) )
            print("    others: {:1.2f}±{:1.2f} shortest paths".format(np.mean(other_shortestpaths),np.std(other_shortestpaths)) )
            # D’Agostino and Pearson’s test for normality
            # null hypothesis: x comes from a normal distribution
            alpha = 5e-2 # p=0.05
            p = 0
            for distr in [core_shortestpaths, other_shortestpaths]:
                _, pdistr = stats.normaltest(distr)
                p += pdistr
            p /= len([core_shortestpaths, other_shortestpaths]) # arbitrary and simple
            if p < alpha:
                print("    Non-normal distribution: Kruskal-Wallis will be performed.")
                kwstat,pval = stats.kruskal(core_shortestpaths, other_shortestpaths)
                print("    test results:",kwstat,pval)
            else:
                print("    Normal distribution: One-way ANOVA will be performed.")
                kwstat,pval = stats.f_oneway(core_shortestpaths, other_shortestpaths)
                print("    test results:",kwstat,pval)
            d,_ = stats.ks_2samp(core_shortestpaths, other_shortestpaths) # non-parametric measure of effect size [0,1]
            print('    Kolmogorov-Smirnov Effect Size: %.3f' % d)
            fig, ax = plt.subplots()
            xs = np.random.normal(1, 0.04, len(core_shortestpaths))
            plt.scatter(xs, core_shortestpaths, alpha=0.3, c='forestgreen', edgecolor='none')
            xs = np.random.normal(2, 0.04, len(other_shortestpaths))
            plt.scatter(xs, other_shortestpaths, alpha=0.3, c='silver', edgecolor='none')
            bp = ax.boxplot([core_shortestpaths,other_shortestpaths], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k',linestyle='-'),medianprops=dict(color='orange'))
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Shortest path length')
            plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            plt.xticks([1, 2], ["core\n(n={:d})".format(len(core_shortestpaths)), "other\n(n={:d})".format(len(other_shortestpaths))])
            fig.savefig(folder+'/global_cores_others_shortestpath.png', transparent=True)
            plt.close()
            fig.clf()

            # print('... flow between beginning and end of event cells')
            # # Flow
            # print("... loading source_target")
            # source_target_cidx = np.load(folder+'/source_target_cidx_py.npy')
            # source_target_color = np.load(folder+'/source_target_color_py.npy')
            # # Returns all the cuts between the source and target vertices in a directed graph.
            # # This function lists all edge-cuts between a source and a target vertex. Every cut is listed exactly once.
            # core_edges = []
            # other_edges = []
            # cut_distr_clusters = []
            # dgraph.to_directed() # arbitrary
            # for sts,stscol in zip(source_target_cidx,source_target_color):
            #     cuts = dgraph.all_st_cuts(source=sts[0], target=sts[1])
            #     for cut in cuts:
            #         for edge in cut.es:
            #             source_vertex_id = edge.source
            #             target_vertex_id = edge.target
            #             if source_vertex_id in core_indexes:
            #                 core_edges.append(source_vertex_id)
            #                 cut_distr_clusters.append(stscol)
            #             elif target_vertex_id in core_indexes:
            #                 core_edges.append(target_vertex_id)
            #                 cut_distr_clusters.append(stscol)
            #             else:
            #                 other_edges.append(source_vertex_id)
            #                 other_edges.append(target_vertex_id)
            # # clusters_cores_by_color
            # print("    cores in the edges removed to stop the flow:",np.unique(core_edges, return_counts=True))
            # print("    core edges flow cut by cluster:",np.unique(cut_distr_clusters, return_counts=True))
            # print("    others in the edges removed to stop the flow:",np.unique(other_edges, return_counts=True))
            # print(core_edges)
            # print(cut_distr_clusters)
            #
            #
            # # -------------------------- recursive measure here because lengthy, to be moved up with the others
            #
            # print('... cycles')
            # # https://stackoverflow.com/questions/31034730/graph-analysis-identify-loop-paths
            # # breadth first search of paths and unique cycles
            # def get_cycles(adj, paths, maxlen):
            #     # tracking the actual path length:
            #     maxlen -= 1
            #     nxt_paths = []
            #     # iterating over all paths:
            #     for path in paths['paths']:
            #         # iterating neighbors of the last vertex in the path:
            #         for nxt in adj[path[-1]]:
            #             # attaching the next vertex to the path:
            #             nxt_path = path + [nxt]
            #             if path[0] == nxt and min(path) == nxt:
            #                 # the next vertex is the starting vertex, we found a cycle
            #                 # we keep the cycle only if the starting vertex has the
            #                 # lowest vertex id, to avoid having the same cycles
            #                 # more than once
            #                 paths['cycles'].append(nxt_path)
            #                 # if you don't need the starting vertex
            #                 # included at the end:
            #                 # paths$cycles <- c(paths$cycles, list(path))
            #             elif nxt not in path:
            #                 # keep the path only if we don't create
            #                 # an internal cycle in the path
            #                 nxt_paths.append(nxt_path)
            #     # paths grown by one step:
            #     paths['paths'] = nxt_paths
            #     if maxlen == 0:
            #         # the final return when maximum search length reached
            #         return paths
            #     else:
            #         # recursive return, to grow paths further
            #         return get_cycles(adj, paths, maxlen)
            # # Comparison of core based cycles vs other based cycles
            # maxlen = 10 # the maximum length to limit computation time
            # # creating an adjacency list
            # adj = [[n.index for n in v.neighbors()] for v in dgraph.vs]
            # # recursive search of cycles
            # # for each core vertex as candidate starting point
            # core_cycles = []
            # for start in core_indexes:
            #     core_cycles += get_cycles(adj,{'paths': [[start]], 'cycles': []}, maxlen)['cycles']
            # print("    # core-based cycles:", len(core_cycles) )
            # # count the length of loops involving 1 core
            # core_cycles_lens = [len(cycle) for cycle in core_cycles]
            # print("    core-based cycles length: {:1.2f}±{:1.2f} vertices".format(np.mean(core_cycles_lens),np.std(core_cycles_lens)) )
            # other_cycles = []
            # for start in other_indexes:
            #     other_cycles += get_cycles(adj,{'paths': [[start]], 'cycles': []}, maxlen)['cycles']
            # print("    # other-based cycles:", len(other_cycles) )
            # # count the length of loops involving 1 core
            # other_cycles_lens = [len(cycle) for cycle in other_cycles]
            # print("    other-based cycles length: {:1.2f}±{:1.2f} vertices".format(np.mean(other_cycles_lens),np.std(other_cycles_lens)) )
            #
            # # D’Agostino and Pearson’s test for normality
            # # null hypothesis: x comes from a normal distribution
            # alpha = 5e-2 # p=0.05
            # p = 0
            # for distr in [core_cycles_lens, other_cycles_lens]:
            #     _, pdistr = stats.normaltest(distr)
            #     p += pdistr
            # p /= len([core_cycles_lens, other_cycles_lens]) # arbitrary and simple
            # if p < alpha:
            #     print("    Non-normal distribution: Kruskal-Wallis will be performed.")
            #     kwstat,pval = stats.kruskal(core_cycles_lens, other_cycles_lens)
            #     print("    Core vs Others cycle length test results:",kwstat,pval)
            # else:
            #     print("    Normal distribution: One-way ANOVA will be performed.")
            #     kwstat,pval = stats.f_oneway(core_cycles_lens, other_cycles_lens)
            #     print("    Core vs Others cycle length test results:",kwstat,pval)
            # d,_ = stats.ks_2samp(core_cycles_lens, other_cycles_lens) # non-parametric measure of effect size [0,1]
            # print('    Kolmogorov-Smirnov Effect Size: %.3f' % d)
            # # all cycles by type
            # fig, ax = plt.subplots()
            # xs = np.random.normal(1, 0.04, len(core_cycles_lens))
            # plt.scatter(xs, core_cycles_lens, alpha=0.3, c='forestgreen', edgecolors='none')
            # xs = np.random.normal(2, 0.04, len(other_cycles_lens))
            # plt.scatter(xs, other_cycles_lens, alpha=0.3, c='silver', edgecolors='none')
            # bp = ax.boxplot([core_cycles_lens,other_cycles_lens], notch=0, sym='', showcaps=False, boxprops=dict(color='k'),whiskerprops=dict(color='k'),medianprops=dict(color='orange'))
            # ax.spines['top'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # plt.ylabel('Cycles length')
            # plt.title("t={:.2f} p={:1.5f}".format(kwstat,pval))
            # plt.xticks([1, 2], ["core\n(n={:d})".format(len(core_cycles_lens)), "other\n(n={:d})".format(len(other_cycles_lens))])
            # fig.savefig(folder+'/global_cores_others_cyclelens.svg', transparent=True)
            # fig.savefig(folder+'/global_cores_others_cyclelens.png', transparent=True, dpi=1500)
            # plt.close()
            # fig.clf()

    # --------------------------------------------------------------------------
    return []
