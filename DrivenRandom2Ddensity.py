{

    'run_time': 120000, # ms
    'dt': 0.1, # ms

    'Populations' : {
        'drive' : {
            # 'n' : 3*3,
            # 'n' : 6*6,
            # 'n' : 12*12,
            'n' : 25*25,
            # 'n' : 50*50,
            # 'n' : 100*100,
            'type': sim.SpikeSourcePoisson,
            'cellparams' : {
                'start':0.0,
                'rate':4.,
                'duration': 120000.0
            }
        },

       'py' : {
            'n': 100*100, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, fill_order='sequential', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams': {
                'e_rev_I'    : -80,   # mV, reversal potential of excitatory synapses
                'e_rev_E'    : 0,     # mV, reversal potential of inhibitory synapses
                'tau_syn_E'  : 5.0,   # ms, time constant of excitatory synaptic short-term plasticity, YgerBoustaniDestexheFregnac2011
                'tau_syn_I'  : 5.0,   # ms, time constant of excitatory synaptic short-term plasticity, YgerBoustaniDestexheFregnac2011
                'tau_refrac' : 5.0,   # ms, refractory period
                'v_reset'    : -65.0, # mV, reset after spike
                'v_thresh'   : -50.0, # mV, spike threshold (modified by adaptation)
                'delta_T'    : 2.,    # mV, steepness of exponential approach to threshold
                'cm'         : 0.150, # nF, tot membrane capacitance
                'a'          : 4.,    # nS, conductance of adaptation variable
                'tau_m'      : 15.0,  # ms, time constant of leak conductance (cm/gl)
                'v_rest'     : -65.0, # mV, resting potential E_leak
                'tau_w'      : 500.0, # ms, time constant of adaptation variable
                'b'          : .02,   # nA, increment to adaptation variable
            },
        },
        'inh' : {
            'n': 50*50, #{'ref':'py','ratio':0.25},
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=2.0, dy=2.0, fill_order='sequential', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams': {
                'e_rev_I'    : -80,   # mV, reversal potential of excitatory synapses
                'e_rev_E'    : 0,     # mV, reversal potential of inhibitory synapses
                'tau_syn_E'  : 5.0,   # ms, time constant of excitatory synaptic short-term plasticity, YgerBoustaniDestexheFregnac2011
                'tau_syn_I'  : 5.0,   # ms, time constant of inhibitory synaptic short-term plasticity, YgerBoustaniDestexheFregnac2011
                'tau_refrac' : 5.0,   # ms, refractory period
                'v_reset'    : -65.0, # mV, reset after spike
                'v_thresh'   : -50.0, # mV, spike threshold (modified by adaptation)
                'delta_T'    : 0.5,   # mV, steepness of exponential approach to threshold
                'cm'         : 0.150, # nF, tot membrane capacitance
                'a'          : 0.0,   # nS, conductance of adaptation variable
                'tau_m'      : 15.0,  # ms, time constant of leak conductance (cm/gl)
                'v_rest'     : -65.0, # mV, resting potential E_leak
                'tau_w'      : 500.0, # ms, time constant of adaptation variable
                'b'          : 0.0,   # nA, increment to adaptation variable
            },
        },
    },

    'Projections' : {
        'drive_py' : {
            'source' : 'drive',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,100), (0,100), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.01, rng=sim.random.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            # 'weight' : .56, # uS # 3*3 *1000 *.56 = 5040
            # 'weight' : .14, # uS # 6*6 *1000 *.14 = 5040
            # 'weight' : .035, # uS # 12*12 *1000 *.035 = 5040
            'weight' : .008, # uS # 25*25 *1000 *.008 = 5000
            # 'weight' : .002, # uS # 50*50 *1000 *.002 = 5000
            # 'weight' : .0005, # uS # 100*100 *1000 *0.005 = 5000
            'receptor_type' : 'excitatory'
        },
        'drive_inh' : {
            'source' : 'drive',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,100), (0,100), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.01, rng=sim.random.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            # 'weight' : .56, # uS # 3*3 *1000 *.56 = 5040
            # 'weight' : .14, # uS # 6*6 *1000 *.14 = 5040
            # 'weight' : .035, # uS # 12*12 *1000 *.035 = 5040
            'weight' : .008, # uS # 25*25 *1000 *.008 = 5000
            # 'weight' : .002, # uS # 50*50 *1000 *.002 = 5000
            # 'weight' : .0005, # uS # 100*100 *1000 *0.005 = 5000
            'receptor_type' : 'excitatory'
        },

        'py_py' : {
            'source' : 'py',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,100), (0,100), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.005, allow_self_connections=False, rng=sim.random.NumpyRNG(2**32-1, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0005,
            'delay' : .5, # ms,
            'receptor_type' : 'excitatory',
            # 'save_connections':True,
            # 'print_statistics':True,
        },
        'py_inh' : {
            'source' : 'py',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,100), (0,100), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.005, allow_self_connections=False, rng=sim.random.NumpyRNG(2**32-1, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : {'ref':'py_py'}, # µS
            'delay' : .5, # ms,
            'receptor_type' : 'excitatory',
            # 'save_connections':True,
            # 'print_statistics':True,
        },
        'inh_inh' : {
            'source' : 'inh',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,100), (0,100), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.005, allow_self_connections=False, rng=sim.random.NumpyRNG(2**32-1, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0025,
            'delay' : .5, # ms,
            'receptor_type' : 'inhibitory',
            # 'save_connections':True,
            # 'print_statistics':True,
        },
        'inh_py' : {
            'source' : 'inh',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,100), (0,100), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.005, allow_self_connections=False, rng=sim.random.NumpyRNG(2**32-1, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : {'ref':'inh_inh'}, # µS
            'delay' : .5, # ms,
            'receptor_type' : 'inhibitory',
            # 'save_connections':True,
            # 'print_statistics':True,
        },

    },


    'Recorders' : {
        'py' : {
            'spikes' : 'all',
            # 'v' : {
            #     'start' : 0,
            #     'end' : 100,
            # },
            'gsyn_exc' : {
                'start' : 0,
                'end' : 10,
            },
            'gsyn_inh' : {
                'start' : 0,
                'end' : 10,
            },
        },
        'inh' : {
            'spikes' : 'all',
            # 'v' : {
            #     'start' : 0,
            #     'end' : 100,
            # },
            'gsyn_exc' : {
                'start' : 0,
                'end' : 10,
            },
            'gsyn_inh' : {
                'start' : 0,
                'end' : 10,
            },
        },
    },


    'Modifiers' :{
    },

    'Injections' : {
    },

    'Analysis' : {
        'scores' : ['py'],
        # 'scores' : ['py','inh'],

        'subsampling' : 1000, # units
        'transient' : 1000, # ms


        'Structure': {
            'py':{
                'conns': 'py_py',
                'shape': (100**2, 100**2),
            },
        },

        # 'Connections_Clustering': {
        #     'fuse': True, # whether multiple instances of connections containing only 'py' should be merged into one adjacency matrix
        #     'fused_shape': (100**2 + 50**2, 100**2 + 50**2), # shape of the fused population
        #     'super_shape': (100**2, 100**2),
        #     'slices':  {  # slice indexes of the fused array to contain the labelled conns
        #         'py_py':     [(0,     100**2), (0, 100**2)],
        #         'py_inh':    [(0,     100**2), (100**2, 100**2+50**2)],
        #         'inh_py':    [(100**2, 100**2+50**2), (0, 100**2)],
        #         'inh_inh':   [(100**2, 100**2+50**2), (100**2, 100**2+50**2)],
        #         'py_py_ca':  [(0,     100**2), (0, 100**2)],
        #         'py_inh_ca': [(0,     100**2), (100**2, 100**2+50**2)],
        #         'inh_py_ca': [(100**2, 100**2+50**2), (0, 100**2)],
        #         'inh_inh_ca':[(100**2, 100**2+50**2), (100**2, 100**2+50**2)],
        #     },
        # },

        'Events_Clustering': {
            'py':{
                # 'add': ['inh'],
                'limits': [(0,100),(0,100)],  # coords: [(from x, to x), (from y, to y)]
                # 'bin': 5, # ms
                'bin': 15, # ms (at 2photon resolution)
                'trials': ['default'], # for which trials the analysis has to be computed
                'ylim': [0,20],
                'print2Dmap': False,
            },
        },

        'ConductanceBalance' : {
            'py':{
                'trials': ['default'], # for which trials the analysis has to be computed
            },
            'inh':{
                'trials': ['default'], # for which trials the analysis has to be computed
            },
        },

        'FiringRate' : {
            'bin': 10, # ms
            'py':{
                'firing': [0,20],
            },
            'inh':{
                'firing': [0,20],
            },
        },

        'Rasterplot' : {
            'py':{
                'limits': [(0,100),(0,100)], # coords: [(from x, to x), (from y, to y)]
                'color': 'black',
            },
            'inh':{
                'limits': [(0,100),(0,100)], # coords: [(from x, to x), (from y, to y)]
                'color': 'blue',
            },
            'type': '.png',
            # 'type': '.svg',
            'interval': False, # all
            # 'interval': [2000.,3000.], # ms # from 2s to 3s
            'dpi':800,
        },

        # 'ISI' : {
        #     'py':{
        #         'bin': 50, # ms, 20 per second
        #         # 'limits': [(0,100),(0,100)], # coords: [(from x, to x), (from y, to y)]
        #         'limits': [(10,50),(10,50)], # only central ones
        #     },
        # },
    },

}
