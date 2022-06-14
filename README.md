# Guarino-Filipchuk-Destexhe-simulations #

Repository containing everything you need to perform simulations and analyses to reproduce the figure panels of the Guarino, Filipchuk, and Destexhe paper.    

With git installed on your system, download the repository on your machine.   
From a terminal, type
```
git clone https://github.com/dguarino/Guarino-Filipchuk-Destexhe-simulations.git
```

cd into it and follow the instructions below.

## Step 1: Docker Image ##

Everything you need to run the simulations is found in the invaluable docker image *simulationx* maintained by Andrew Davison [here](https://hub.docker.com/r/neuralensemble/simulationx/):

- shell environment with NEST 2.14, NEURON 7.5, and PyNN 0.9 installed.    
- The Python 2.7 version provides Brian 1.4, the Python 3.4 version provides Brian 2.    
- IPython, scipy, matplotlib and OpenMPI are also installed.    

The local `Dockerfile` is simply adding the libraries required to perform the analysis.   

You need to install Docker... from here https://docs.docker.com/get-docker/.    

### Basic Docker use ###
Start docker daemon

```
sudo systemctl restart docker
```

Enable the current user to launch docker images

```
sudo usermod -a -G docker $USER
```

Move to the folder `Guarino-Filipchuck-Destexhe-simulations` checked out from github and build the image (don't forget the dot at the end of next line!)

```
docker build -t neuro .
```

Check the existence of the image

```
docker images
```

Start a container with the *neuro* image

```
docker run -i -t neuro /bin/bash
```

And to allow for code development, bind-mount your local files in the container

```
docker run -v `pwd`:`pwd` -w `pwd` -i -t neuro /bin/bash
```

You will now be inside the Docker container, with the git repository mapped into it.


### How to: figure 3 of the Guarino, Filipchuk, Destexhe paper ###


Enter the repository folder and run the simulation for cortical 2D grid of excitatory and inhibitory Adaptive-Exponential Integrate-and-Fire point neurons, connected using a distance-dependent rule.   

```
python run.py --folder DrivenRange2D --params DrivenRange2Ddensity.py nest
```

You can have help on the parameters by typing `python run.py --help`.   
This simulation will take a while to execute (depending on your machine).   

After the simulation has run, the analyses can be performed independently by specifying which file on the command line.    
For example, to have a look at the global properties of the network (firing rate, conductance balance, ...) run the following:   

```
python run.py --folder DrivenRange2D --params DrivenRange2Ddensity.py --analysis default_analysis.py nest
```

There are files to run the analyses for figure 3 of the paper:   

- `dynamical_analysis.py` routines for the analysis of simulation results
- `structural_analysis.py` additional routines for the analysis of simulation results
- `default_analysis.py` additional routines for the analysis of simulation results

There are also simulation management files:    

- `helpers.py` contains all the routines to drive PyNN to define models and stimuli, run simulations (also for parameter searches), collect state values and save them as results ready to be analysed.   
- `run.py` contains the code to interpret various shell commands.    

These contain helper functions to ease the development of spiking neural networks in [PyNN](http://neuralensemble.org/docs/PyNN/index.html).   
