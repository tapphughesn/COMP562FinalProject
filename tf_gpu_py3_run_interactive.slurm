#!/bin/bash

## This is an example of a script to run  tensorflow interactively
## using Singularity to run the tensorflow image.
##
## You will be dropped into an interactive shell with the tensorflow environment.
##
## Make a copy of this script and change reserved resources on the srun command as needed for your job.
##
## Just execute this script on the command line.

unset OMP_NUM_THREADS

# Set SIMG path
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg

# Set SIMG name
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

# Run interactive job to GPU node using Singularity
srun --ntasks=1 --cpus-per-task=1 --mem=20G --time=4:00:00 --partition=volta-gpu --gres=gpu:1 --qos=gpu_access --pty singularity shell --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME  


