#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os.path
import re
from shutil import copyfile

"""
This function generates a three-column csv file that contains the replica-specfic input quantities
needed to run the LLR for the heatbath updates. The layout is the following:
    Column 1: Central energy (aka E0/S0)
    Column 2: Initial value of a_0
    Column 3: Energy interval width (aka dE/dS) 
"""
def initial_an(infofile):
    info_df = pd.read_csv(infofile)

    V = info_df['Lt'][0]*info_df['Ls'][0]**3
    umin, umax = info_df['umin'][0], info_df['umax'][0]
    nreplicas  = info_df['n_replicas'][0]
    init_file  = info_df['an_file'][0]

    init_df = pd.read_csv(init_file)
    beta    = init_df['beta'] 
    S0      = init_df['S0']
    Eks     = np.linspace(umin,umax, nreplicas)* 6 * V
    fit     = np.poly1d(np.polyfit(S0,beta,3))
    aks     = fit(Eks)
    dE      = (Eks[1]-Eks[0])*2

    return Eks, aks, dE, nreplicas
    
def setup_input_files(infile,outfile,infofile):
    info_df = pd.read_csv(infofile)
    
    nreplicas = info_df['n_replicas'][0]
    N_meas = info_df['N_meas'][0]
    N_th = info_df['N_th'][0]
    Lt = info_df['Lt'][0] # temporal length 
    Ls = info_df['Ls'][0] # spatial length
    PX = info_df['PX'][0] # domain decomposition

    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*GLB_T.*$', f'GLB_T = {Lt}', line)
            line = re.sub(r'^.*GLB_X.*$', f'GLB_X = {Ls}', line)
            line = re.sub(r'^.*GLB_Y.*$', f'GLB_Y = {Ls}', line)
            line = re.sub(r'^.*GLB_Z.*$', f'GLB_Z = {Ls}', line)
            line = re.sub(r'^.*NP_X.*$' , f'NP_X = {PX}', line)
            line = re.sub(r'^.*N_REP.*$', f'N_REP = {nreplicas}', line)
            line = re.sub(r'^.*llr:nmc.*$', f'llr:nmc = {N_meas}', line)
            line = re.sub(r'^.*llr:nth.*$', f'llr:nth = {N_th}', line)
            print(line, end='',file=io)

def setup_initial_an(infile,outfile,Emin,Emax,S0,dS,a):
    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*llr:S0.*$'    , f'llr:S0 = {S0:.5f}', line)
            line = re.sub(r'^.*llr:dS.*$'    , f'llr:dS = {dS:.5f}', line)
            line = re.sub(r'^.*llr:starta.*$', f'llr:starta = {a:.5f}', line)
            line = re.sub(r'^.*llr:Smin.*$'  , f'llr:Smin = {Emin:.5f}', line)
            line = re.sub(r'^.*llr:Smax.*$'  , f'llr:Smax = {Emax:.5f}', line)
            print(line, end='',file=io)

def setup_bash_files(infile,outfile,infofile):
    info_df = pd.read_csv(infofile)
    
    nreplicas = info_df['n_replicas'][0]
    N_NR = info_df['N_NR'][0]
    N_RM = info_df['N_RM'][0]
    Lt = info_df['Lt'][0] # temporal length 
    Ls = info_df['Ls'][0] # spatial length

    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*run_name.*$', f'run_name=sp4_{Lt}x{Ls}_{nreplicas}', line)
            line = re.sub(r'^.*n_RM=.*$', f'n_RM={N_RM}', line)
            line = re.sub(r'^.*n_NR=.*$', f'n_NR={N_NR}', line)
            print(line, end='',file=io)

# define ceil division in analogy to floor division
def ceildiv(a, b):
    return -(a // -b)

def setup_batch_files(infile,outfile,infofile,cores_per_node):
    info_df = pd.read_csv(infofile)
    nreplicas = info_df['n_replicas'][0]
    PX = info_df['PX'][0] # domain decomposition
    tasks = nreplicas*PX 
    nodes = ceildiv(tasks, cores_per_node)

    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'SBATCH --nodes=[0-9]+',"SBATCH --nodes="+str(nodes),line)
            line = re.sub(r'SBATCH --ntasks=[0-9]+',"SBATCH --ntasks="+str(tasks),line)
            line = re.sub(r'SBATCH --ntasks-per-node=[0-9]+',"SBATCH --ntasks-per-node="+str(cores_per_node),line)
            line = re.sub(r'-n\s+[0-9]+',"-n "+str(tasks),line)
            line = re.sub(r'-r\s+[0-9]+',"-r "+str(nreplicas),line)
            print(line,end='',file=io)