"""
Copyright (c) 2016-2021, Domenico GUARINO
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite Paris-Saclay nor the
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

import math
import numpy as np

from pyNN.utility import Timer
from pyNN.space import Space
from pyNN.space import Grid2D
from pyNN.parameters import Sequence

from functools import partial # to run partial dt callback functions

from scipy import ndimage as scimage
import scipy.io

import scipy.stats as stats
from scipy import signal

################################
import matplotlib
matplotlib.use('Agg') # to be used when DISPLAY is undefined
################################
from matplotlib import image as pltimg
import matplotlib.pyplot as plt


# # TODO: callback function to modify parameters while running the simulation
# def modify_populations(t, params, populations):
#     # populations['re'].set(tau_m=params['Populations']['cell']['cellparams']['tau_m'])
#     if t>5000.0: # ms
#         print(t)
#         # spiketrains = populations['re'].get_data().segments[0]
#         # get all tau_m: populations['re'][:].tau_m # because some of them will be different
#         # for each cell:
#         #   if fired:
#         #       Ih = Ih + 'callback_params''increment'
#         #       populations['re'][start:end].tau_m = params['tau_m'] + Ih
#         #   else:
#         #       exponential decay of Ih
#         if populations['re'][0].tau_m > 12.:
#             populations['re'].set(tau_m=(populations['re'][0].tau_m)-0.5)
#             print( populations['re'][0].tau_m )
#     return t + 50. #params['push_interval']


# https://github.com/btel/python-in-neuroscience-tutorials/blob/master/poisson_process.ipynb
def inhomogeneous_poisson(rate, bin_size, start):
    n_bins = len(rate)
    spikes = np.random.rand(n_bins) < rate * bin_size
    spike_times = np.nonzero(spikes)[0] * bin_size
    return spike_times + start


class SetRate(object):
    """
    A callback which changes the firing rate of a population of poisson
    processes at a fixed interval.
    """
    def __init__(self, simulator, population, rate_generator, interval=30.0):
        assert isinstance(population.celltype, simulator.SpikeSourcePoisson)
        self.population = population
        self.interval = interval
        self.rate_generator = rate_generator
    def __call__(self, t):
        try:
            lra = next(self.rate_generator)
            # print(lra)
            self.population.set( rate = lra ) # list of rates for each unit in the population
            # self.population.set( rate = next(self.rate_generator) ) # list of rates for each unit in the population
        except StopIteration:
            pass
        return t + self.interval # next update


# Stimuli
# All sensory modalities are considered to be in one way or another topologically organised and time varying.
# Therefore, it is expected a time series of 2D arrays, with a sampling rate specification.
# Then, during runtime, a callback function is used to alter the rates of a 2D SpikeSourcePoisson population.
# Each frame image is expected to contain a rightly shaped array of 2Dvalues (in the range .0, 1.)
# used to set the Poisson rates at the corresponding 2D position 2Dvalues * amp

# Video
# An image is provided containing all frames side by side
# All frames are extracted at init time and stored in a 3D array
def load_visual_stimulus(filename, x, y, frames, zoom_x, zoom_y, maxr, fromRGBA=False):
    image = pltimg.imread(filename) # load image as pixel array
    # print(image.shape)
    if fromRGBA:
        image = image[:,:,0] # expected grayscale image with RGB all equal
    image = (image/255)*maxr # convert to 0,1 interval
    # print(image)
    images = np.array( np.hsplit(image, frames) ) #  cut the film into frames
    # print(images.shape)
    # print(images)
    # zoom to fit population
    # from (120, 125, 250) to 120, 64, 64
    images = scimage.zoom(images, [1, zoom_y/y, zoom_x/x] )
    images[images<0] = 0.0 # during the zooming process <0 pixelsmay occur
    # input at the end will be 1D to the cells array
    images = np.reshape(images, (images.shape[0], int(zoom_x*zoom_y)) )
    # print(images.shape)
    return images

# Audio
def load_audio_stimulus(filename, start, factor):
    mat = scipy.io.loadmat(filename)
    bin_size = 0.1 # ms
    channels = []
    for rate in mat['Z']:
        channels.append( inhomogeneous_poisson(factor*rate, bin_size, start) ) # ms
        # channels.append( Sequence(inhomogeneous_poisson(rate, bin_size, start)) ) # ms
    # print(channels)
    print(filename, "channels:", len(channels), "lenght:", mat['X'].shape[1])
    return channels

# spiketrains from file
def load_stimulus(filename, start):
    channels = np.load(filename, allow_pickle=True)
    # print(channels)
    channels += start
    # print(channels)
    print(filename, "channels:", len(channels))
    return channels


# -----------------------------------------------------------------------------

def build_network(sim, params):
    print("\nSetting up the Network ...")
    timer = Timer()
    timer.reset()

    sim.setup( timestep=params['dt'] )

    stimulus_frames = []
    populations = {}
    for popKey,popVal in params['Populations'].items():
        # number of units
        number = popVal['n']
        if isinstance(popVal['n'],dict):
            number = int(params['Populations'][popVal['n']['ref']]['n'] * popVal['n']['ratio'])
        # population structure
        if 'structure' in popVal:
            populations[popKey] = sim.Population( number, popVal['type'], cellparams=popVal['cellparams'], structure=popVal['structure'])
            positions = popVal['structure'].generate_positions(number)
        else:
            populations[popKey] = sim.Population( number, popVal['type'], cellparams=popVal['cellparams'])
        # population creation
        populations[popKey].initialize()

        # complex stimuli
        if 'protocol' in popVal and 'modality' in popVal:

            if popVal['modality']=='video':
                if not 'structure' in popVal:
                    print("ERROR: 2D stimuli can only be used with 2D populations. Population "+popKey+" has no structure.")

                stimulus_frames = load_visual_stimulus(
                    filename=popVal['source']['file'],
                    x=popVal['source']['x'],
                    y=popVal['source']['y'],
                    frames=popVal['source']['f'],
                    zoom_x=math.sqrt(number),
                    zoom_y=math.sqrt(number),
                    maxr=popVal['source']['rate'],
                    fromRGBA=popVal['source']['fromRGBA']
                )
                # print(stats.describe(stimulus_frames)) # to check for stimulus stats
                # print(stats.find_repeats(stimulus_frames)) # to check for "blank" (array([ 0.]), array([1237582], dtype=int32))

            if popVal['modality'] in ['audio','fromfile']:
                spiketimes = []
                for stimulus_source in popVal['protocol']:
                    if stimulus_source['source']=='blank':
                        continue # no spike times
                    else:
                        if popVal['modality']=='audio':
                            astimulus = load_audio_stimulus(
                                filename=stimulus_source['source'],
                                start=stimulus_source['start'],
                                factor=stimulus_source['factor'],
                            )
                        if popVal['modality']=='fromfile':
                            astimulus = load_stimulus(
                                filename=stimulus_source['source'],
                                start=stimulus_source['start'],
                            )
                        # protocol
                        if len(spiketimes)==0:
                            # first (or unique) stimulus
                            spiketimes.extend( astimulus )
                        else:
                            # following stimuli
                            for cellnum in range(len(spiketimes)):
                                spiketimes[cellnum] = np.append(spiketimes[cellnum], astimulus[cellnum])
                # print(spiketimes)
                populations[popKey].set(spike_times=spiketimes) # assigned after creation
                # print(populations[popKey].get('spike_times')) # assigned after creation

    projections = {}
    for projKey,projVal in params['Projections'].items():
        if not 'space' in projVal:
            projVal['space'] = Space()
        projections[projKey] = sim.Projection(
            populations[ projVal['pre'] ],
            populations[ projVal['post'] ],
            connector = projVal['connector'],
            synapse_type = projVal['synapse_type'],
            receptor_type = projVal['receptor_type'],
            space = projVal['space'],
        )
        if 'weight' in projVal:
            if isinstance(projVal['weight'], dict):
                refkey = projVal['weight']['ref'] # get list from dotted string
                projVal['weight'] = params['Projections'][refkey]['weight']
            projections[projKey].set(weight=projVal['weight'])
        if 'delay' in projVal:
            projections[projKey].set(delay=projVal['delay'])

        # printout connectivity stats
        if 'print_statistics' in projVal:
            if projVal['print_statistics']:
                print(projKey, "- total projections:", projections[projKey].size())
                projList = projections[projKey].get(["weight"], format="list")
                # mean_dist
                mean_dist = 0
                for conn in projList:
                    cidx_pre = int(conn[0])
                    cidx_post = int(conn[1])
                    a = np.array([ populations[projVal['pre']].positions[0][cidx_pre], populations[projVal['pre']].positions[1][cidx_pre] ])
                    b = np.array([ populations[projVal['post']].positions[0][cidx_post], populations[projVal['post']].positions[1][cidx_post] ])
                    mean_dist += np.linalg.norm(a-b)
                mean_dist /= len(projList)
                print(projKey, "- mean conn distance:", mean_dist)

    simCPUtime = timer.elapsedTime()
    print("... the simulation took %s s to setup." % str(simCPUtime))
    return populations, projections, stimulus_frames


def modify_populations(sim, modify_params, populations):
    for popKey,modification in modify_params.items():
        # sim.set(populations[popKey], modification)
        # populations[popKey].set(modification)
        # print('... modified population', popKey)
        for key,value in modification.items():
            populations[popKey].set(**{key:value})
            # print('... modified param',key,'(',str(value),') for population', popKey)


def run_simulation(sim, params, populations, stimulus_frames):
    print("\nRunning Network ...")
    timer = Timer()
    timer.reset()

    # modpop = partial(modify_populations, populations=populations)
    # sim.run(run_time, callbacks=[modpop])

    if len(stimulus_frames)>1:
        # print("Stimulus frame mean:",np.mean(stimulus_frames), "   max",np.max(stimulus_frames))
        interval = params['Populations']['stimulus']['interval']
        rate_generator = iter(stimulus_frames)
        sim.run(params['run_time'], callbacks=[ SetRate(sim, populations['stimulus'], rate_generator, interval) ])
    else:
        sim.run(params['run_time']) # default no stimulus population

    simCPUtime = timer.elapsedTime()
    print("... the network took %s s to run." % str(simCPUtime))


def perform_injections(params, populations):
    for modKey,modVal in params['Injections'].items():
        # if isinstance(modVal['spike_times'], (list)):
        #     source = modVal['source'](spike_times=modVal['spike_times'])
        # elif
        if 'mean' in modVal:
            source = modVal['source'](mean=modVal['mean'], stdev=modVal['stdev'], start=modVal['start'], stop=modVal['stop'] )
        elif isinstance(modVal['start'], (list)):
            source = modVal['source'](times=modVal['start'], amplitudes=modVal['amplitude'])
        else:
            source = modVal['source'](amplitude=modVal['amplitude'], start=modVal['start'], stop=modVal['stop'])
        # either single cell injection or population wide
        if 'cellidx' in modVal:
            populations[modKey][modVal['cellidx']].inject( source )
        else:
            populations[modKey].inject( source )


def record_data(params, populations):
    for recPop, recVal in params['Recorders'].items():
        for elKey,elVal in recVal.items():
            #populations[recPop].record( None )

            if elVal == 'all':
                populations[recPop].record( elKey )

            elif 'MUA' in elVal:
                idxs = populations[recPop].id_to_index(populations[recPop].local_cells)
                edge = int(np.sqrt(len(idxs)))
                idxs.shape = (edge, edge)
                assert elVal['x'] < edge, "MUA Recorder definition: x is bigger than the Grid2D edge"
                assert elVal['x']+elVal['size']+1 < edge, "MUA Recorder definition: x+size is bigger than the Grid2D edge"
                mualist = idxs[ elVal['x']:elVal['x']+elVal['size']+1, elVal['y']:elVal['y']+elVal['size']+1 ].flatten()
                populations[recPop][mualist.astype(int)].record( elKey )

            elif 'random' in elVal:
                populations[recPop].sample(elVal['random']).record( elKey )

            else:
                populations[recPop][elVal['start']:elVal['end']].record( elKey )


def save_connections(params, populations, projections, folder, addon=''):
    print("\nSaving connections ...")

    for key,p in projections.items():
        if 'save_connections' in params['Projections'][key] and params['Projections'][key]['save_connections']:
            conns = p.get(["weight"], format="list", with_address=True)
            # print(conns)
            print(key, addon)
            # format: cell id, cell index, x, y
            np.save(folder+'/connections_'+key+'.npy', conns)

    for key,p in populations.items():
        # positions
        with open(folder+'/'+key+'_positions.txt', 'w') as pfile:
            # write
            for cid in p.all_cells:
                cidx = p.id_to_index(cid)
                pfile.write(str(cid)+" " + str(cidx)+" " + str(p.positions[0][cidx])+" " + str(p.positions[1][cidx])+" " + str(p.positions[2][cidx])+"\n")
            pfile.close()


def save_data(populations, folder, stimulus_frames, addon=''):
    print("\nSaving Data ...")
    timer = Timer()
    timer.reset()

    for key,p in populations.items():
        # recordings
        p.write_data(folder+'/'+key+addon+'.pkl')
        # spiketrains
        spiketrains = []
        # data = p.get_data().segments[0]
        data = p.get_data('spikes')
         # for spiketrain in data.spiketrains:
        for spiketrain in data.segments[0].spiketrains:
            spiketrain = spiketrain.magnitude
            spiketrains.append(spiketrain)
        np.save(folder+'/spiketrains_'+key+addon+'.npy', spiketrains)

    # if len(stimulus_frames):
    #     fig = plt.figure()
    #     for imid,image in enumerate(stimulus_frames):
    #         image = image.reshape( (64,64) )
    #         # # frames
    #         # figs = plt.figure()
    #         # plt.pcolormesh(fr)
    #         # figs.savefig('DrivenRandom2D_reduced_pulse/stim/frame_'+str(im)+'.png')
    #         # plt.close()
    #         # figs.clear()
    #         # spectrum
    #         npix = image.shape[0]
    #         fourier_image = np.fft.fftn(image)
    #         fourier_amplitudes = np.abs(fourier_image)**2
    #         kfreq = np.fft.fftfreq(npix) * npix
    #         kfreq2D = np.meshgrid(kfreq, kfreq)
    #         knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    #         knrm = knrm.flatten()
    #         fourier_amplitudes = fourier_amplitudes.flatten()
    #         kbins = np.arange(0.5, npix//2+1, 1.)
    #         kvals = 0.5 * (kbins[1:] + kbins[:-1])
    #         Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
    #         Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    #         plt.loglog(kvals, Abins, color='grey')
    #         # plt.xlabel("cycles/picture")
    #         # plt.ylabel("P")
    #         # plt.tight_layout()
    #         # plt.savefig(folder+'/'+'stimulus_frame_'+str(imid)+'.svg')
    #         # plt.close()
    #     plt.xlabel("cycles/frame")
    #     plt.ylabel("probability")
    #     plt.tight_layout()
    #     plt.savefig(folder+'/'+'stimuli.svg')
    #     plt.close()
    #     fig.clear()

    simCPUtime = timer.elapsedTime()
    print("... the simulation took %s s to save data." % str(simCPUtime))
