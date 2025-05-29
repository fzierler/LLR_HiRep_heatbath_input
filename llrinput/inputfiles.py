#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os.path as op
import re
import random
import scipy
from shutil import move

"""
This function generates a three-column csv file that contains the replica-specfic input quantities
needed to run the LLR for the heatbath updates. The layout is the following:
    Column 1: Central energy (aka E0/S0)
    Column 2: Initial value of a_0
    Column 3: Energy interval width (aka dE/dS) 
"""
def initial_an(info_df,file_dir):
    V = info_df['Lt']*info_df['Ls']**3
    umin, umax = info_df['umin'], info_df['umax']
    nreplicas  = info_df['n_replicas']
    init_file  = op.join(file_dir,info_df['an_file'])

    init_df = pd.read_csv(init_file)
    beta    = init_df['beta'] 
    plaq    = init_df['plaq']
    S0      = plaq * 6 * V
    Eks     = np.linspace(umin,umax, nreplicas)* 6 * V
    aks     = interpolate_initial_an(S0,beta,Eks)
    dE      = (Eks[1]-Eks[0])*2

    return Eks, aks, dE, nreplicas

def fit_initial_an(S0,beta,Eks):
    fit = np.poly1d(np.polyfit(S0,beta,3))
    aks = fit(Eks)
    return aks

def interpolate_initial_an(S0,beta,Eks):
    spline = scipy.interpolate.PchipInterpolator(S0, beta)
    aks    = spline(Eks)
    return aks

def setup_input_files_inplace(infile,infofile):
    tmpfile = "tmp"
    setup_input_files(infile,tmpfile,infofile)
    move(tmpfile, infile)

def setup_input_files(infile,outfile,info_df):
    nreplicas = info_df['n_replicas']
    N_meas = info_df['N_meas']
    N_th = info_df['N_th']
    Lt = info_df['Lt'] # temporal length 
    Ls = info_df['Ls'] # spatial length
    PX = info_df['PX'] # domain decomposition
    PY = info_df['PY'] # domain decomposition
    
    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*GLB_T.*$', f'GLB_T = {Lt}', line)
            line = re.sub(r'^.*GLB_X.*$', f'GLB_X = {Ls}', line)
            line = re.sub(r'^.*GLB_Y.*$', f'GLB_Y = {Ls}', line)
            line = re.sub(r'^.*GLB_Z.*$', f'GLB_Z = {Ls}', line)
            line = re.sub(r'^.*NP_X.*$' , f'NP_X = {PX}', line)
            line = re.sub(r'^.*NP_Y.*$' , f'NP_Y = {PY}', line)
            line = re.sub(r'^.*N_REP.*$', f'N_REP = {nreplicas}', line)
            line = re.sub(r'^.*llr:nmc.*$', f'llr:nmc = {N_meas}', line)
            line = re.sub(r'^.*llr:nth.*$', f'llr:nth = {N_th}', line)
            print(line, end='',file=io)

def setup_initial_an_inplace(infile,Emin,Emax,S0,dS,a):
    tmpfile = "tmp"
    setup_initial_an(infile,tmpfile,Emin,Emax,S0,dS,a)
    move(tmpfile, infile)

def setup_initial_an(infile,outfile,Emin,Emax,S0,dS,a):
    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*llr:S0.*$'    , f'llr:S0 = {S0:.5f}', line)
            line = re.sub(r'^.*llr:dS.*$'    , f'llr:dS = {dS:.5f}', line)
            line = re.sub(r'^.*llr:starta.*$', f'llr:starta = {a:.5f}', line)
            line = re.sub(r'^.*llr:Smin.*$'  , f'llr:Smin = {Emin:.5f}', line)
            line = re.sub(r'^.*llr:Smax.*$'  , f'llr:Smax = {Emax:.5f}', line)
            line = re.sub(r'^.*rlx_seed.*$'  , f'rlx_seed = {random.randint(1,32767-1)}', line)
            print(line, end='',file=io)

def setup_bash_files(infile,outfile,info_df):  
    nreplicas = info_df['n_replicas']
    N_NR = info_df['N_NR']
    N_RM = info_df['N_RM']
    Lt = info_df['Lt'] # temporal length 
    Ls = info_df['Ls'] # spatial length

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

def setup_batch_files(infile,outfile,info_df,cores_per_node,args):
    nreplicas = info_df['n_replicas']
    time_limit = info_df['time_limit']
    PX = info_df['PX'] # domain decomposition
    tasks = nreplicas*PX 
    nodes = ceildiv(tasks, cores_per_node)
    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'SBATCH --partition=\S*',"SBATCH --partition="+str(args.partition),line)
            line = re.sub(r'SBATCH --mail-user=\S*',"SBATCH --mail-user="+str(args.email),line)
            line = re.sub(r'SBATCH --nodes=[0-9]*', "SBATCH --nodes="+str(nodes),line)
            line = re.sub(r'SBATCH --ntasks=[0-9]*',"SBATCH --ntasks="+str(tasks),line)
            line = re.sub(r'SBATCH --time=\S*'     ,"SBATCH --time="+str(time_limit),line)
            line = re.sub(r'SBATCH --ntasks-per-node=[0-9]*',"SBATCH --ntasks-per-node="+str(cores_per_node),line)
            # Taking of the case when some SBATCH directives are not needed
            # Manages from the argument list.
            if args.qos is None:
                line = re.sub(r'SBATCH --qos=\S*', "" ,line)
            else:
                line = re.sub(r'SBATCH --qos=\S*', "SBATCH --qos="+str(args.qos),line)
            # [end-if] args.machine
            if args.account is None:
                line = re.sub(r'SBATCH --account=\S*', "" ,line)
            else:
                line = re.sub(r'SBATCH --account=\S*', "SBATCH --account="+str(args.account),line)
            # [end-if] args.machine

            line = re.sub(r'-n\s+[0-9]*',"-n "+str(tasks),line)
            line = re.sub(r'-r\s+[0-9]*',"-r "+str(nreplicas),line)
            line = re.sub(r'module load\S*', "module load "+str(args.modules),line)
            line = re.sub(r'(srun|mpirun|mpiexec)', args.mpi_runner,line)
            line = re.sub(r'\S*llr_hb', op.join(args.path_llr_exec,"llr_hb"),line)
            print(line,end='',file=io)

def setup_fxa_input_inplace(infile):
    tmpfile = "tmp"
    setup_fxa_input(infile,tmpfile)
    move(tmpfile, infile)

def setup_fxa_input(infile,outfile):
    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*llr:nfxa.*$'     , 'llr:nfxa=50', line)
            line = re.sub(r'^.*last conf.*$'    , 'last conf=+0', line)
            line = re.sub(r'^.*llr:N_nr.*$'     , 'llr:N_nr=0', line)
            line = re.sub(r'^.*llr:sfreq_fxa.*$', 'llr:sfreq_fxa=100', line)
            print(line,end='',file=io)

def setup_nr_input_inplace(infile,info_df):
    tmpfile = "tmp"
    setup_nr_input(infile,tmpfile,info_df)
    move(tmpfile, infile)

def setup_nr_input(infile,outfile,info_df):
    n_nr_per_step = info_df['N_NR_per_step']
    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*llr:nfxa.*$' , 'llr:nfxa=0', line)
            line = re.sub(r'^.*last conf.*$', 'last conf=0', line)
            line = re.sub(r'^.*llr:N_nr.*$' ,f'llr:N_nr={n_nr_per_step}', line)
            print(line,end='',file=io)

def setup_rm_input_inplace(infile,info_df):
    tmpfile = "tmp"
    setup_rm_input(infile,tmpfile,info_df)
    move(tmpfile, infile)

def setup_rm_input(infile,outfile,info_df):
    n_rm_per_step = info_df['N_RM_per_step']
    rm_it = info_df['rm_it']
    nor = info_df['n_or']
    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            line = re.sub(r'^.*llr:nfxa.*$' , 'llr:nfxa=0', line)
            line = re.sub(r'^.*last conf.*$',f'last conf=+{n_rm_per_step}', line)
            line = re.sub(r'^.*llr:it(?!_).*$',f'llr:it = {rm_it}', line)
            line = re.sub(r'^.*nor.*$',f'nor = {nor}', line)
            line = re.sub(r'^.*llr:N_nr.*$' , 'llr:N_nr=0', line)
            print(line,end='',file=io)