##############################################################################
# A Docker image for running neuronal network simulations
# Linux usage:
#
# $ docker build -t neuro .
#
# default usage (using github repo master):
# $ docker run --privileged --memory="32g" --oom-kill-disable -e DISPLAY=$DISPLAY -i -t neuro /bin/bash
# development usage (slow):
# $ docker run --privileged --memory="32g" --oom-kill-disable -e DISPLAY=$DISPLAY -v `pwd`:`pwd`:delegated -w `pwd` -i -t neuro /bin/bash
# this command line above contains three optimisation for the osx:
# --privileged, our container requires direct hardware access (beware of escalation)
# --oom-kill-disable, disable out of memory sigkill
# --volume:delegated, outsource the underlying filesystem sync-ing (superseded by privileged)

FROM neuralensemble/simulationx:py3_pynn092

# igraph
RUN apt-get update; apt-get install -y bison flex
RUN apt-get -y install libxml2 libxml2-dev zlib1g-dev
RUN $VENV/bin/pip3 install python-igraph==0.8.2
RUN apt-get -y install pkg-config libcairo2-dev libffi-dev
RUN $VENV/bin/pip3 install cairocffi
