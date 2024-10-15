#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os.path
import re
from shutil import copyfile

def predat_from_csv(folder,infofile):
    
    os.makedirs(os.path.join(folder,"base"), exist_ok=True)
    newinfofile = os.path.join(folder,"base","info.csv")
    copyfile(infofile,newinfofile)
    info_df = pd.read_csv(newinfofile)
    predat = os.path.join(folder,"base","pre.dat")

    print('Creating inital a and E')
    V = info_df['Lt'][0]*info_df['Ls'][0]**3
    umin, umax = info_df['umin'][0], info_df['umax'][0]
    nreplicas  = info_df['n_replicas'][0]
    IS_betas   = eval(info_df['IS_b'][0])
    std_folder = info_df['std_folder'][0]
    pre_dat(std_folder,V,umin,umax,nreplicas,IS_betas,predat)
    return

def pre_dat(folder,V,up_min,up_max,N_intervals, betas, location):
    is_df = pd.read_csv(f'{folder}std.csv')
    x = np.array(betas)
    y = np.array([is_df[is_df['Beta']==b]['Plaq'] * 6 * V for b in betas]).flatten()
    fit = np.poly1d(np.polyfit(y,x,3))
    Eks = np.linspace(up_min,up_max, N_intervals)* 6 * V
    aks = fit(Eks)
    dE = (Eks[1]-Eks[0])*2
    output = ''
    for ek,ak in zip(Eks,aks):
        output+=f'{ek:.5f} {ak:.5f} {dE:.5f}\n'
    with open(location, 'w') as f:f.write(output)
    
def input_files_from_csv(infile,outfile,infofile):
    info_df = pd.read_csv(infofile)
    
    nreplicas = info_df['n_replicas'][0]
    N_meas = info_df['N_meas'][0]
    N_NR = info_df['N_NR'][0]
    N_RM = info_df['N_RM'][0]
    N_th = info_df['N_th'][0]
    Lt = info_df['Lt'][0] # temporal length 
    Ls = info_df['Ls'][0] # spatial length
    PX = info_df['PX'][0] # domain decomposition

    io = open(outfile, "w")
    with open(infile, "r") as f:
        for line in f:
            if "GLB_T" in line:
                print("GLB_T =",Lt,file=io)
                continue
            if "GLB_X" in line:
                print("GLB_X =",Ls,file=io)  
                continue
            if "GLB_Y" in line:
                print("GLB_Y =",Ls,file=io)
                continue
            if "GLB_Z" in line:
                print("GLB_Z =",Ls,file=io)
                continue
            if "NP_X" in line:
                print("NP_X =",PX,file=io)
                continue
            if "N_REP" in line:
                print("N_REP =",nreplicas,file=io)
                continue
            if "llr:nmc" in line:
                print("llr:nmc =",N_meas,file=io)
                continue
            if "llr:nth" in line:
                print("llr:nth =",N_th,file=io)
                continue
            if "run_name=" in line:
                print("run_name=sp4_",Lt,"x",Ls,"_",nreplicas,sep='',file=io)
                continue
            if "n_RM=" in line:
                print("n_RM=",N_RM,sep='',file=io)
                continue
            if "n_NR=" in line:
                print("n_NR=",N_NR,sep='',file=io)
                continue
            else:
                print(line, end='',file=io)

def copy_identical_files(folder,basedir):
        # Note, fxa is copied assuming that the parameters for the fxa will not change
        for f in ["list_configs.sh","setup_replicas.sh","setup_replicas_start_cont.sh","setup_replicas_cont.sh","setup_replicas_fxa.sh"]:
            src = os.path.join(basedir,f)
            dst = os.path.join(folder ,f)
            copyfile(src,dst)
        return 

# define ceil division in analogy to floor division
def ceildiv(a, b):
    return -(a // -b)

def csd3_batch_files(infile,outfile,infofile,cores_per_node):
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