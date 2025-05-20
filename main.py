#!/usr/bin/env python
# coding: utf-8
import llrinput.inputfiles as ifiles
import pandas as pd 
import os.path as op
import os
from shutil import copyfile
from argparse import ArgumentParser, FileType
import tqdm

## create a suitable name for the run:
def get_run_name(input_data,ind):
    Lt  = input_data["Lt"].values[0]
    Ls  = input_data["Ls"].values[0]
    Rep = input_data["n_replicas"].values[0]
    return f"Run{ind:03}_LLR_{Lt:02}x{Ls:03}_{Rep:03}"

def main(infofile,args):
    outdir     = "./output/"
    input_dir  = "./input/templates/"
    bash_files  = ["sp4_llr_therm.sh","sp4_llr_newton_raphson.sh","sp4_llr_robbins_monro.sh","sp4_llr_fxa.sh"]
    input_files = ["input_file_therm", "input_file_newton_raphson", "input_file_robbins_monro", "input_file_fxa"]
    setup_files = ["list_configs.sh","update_replicas.sh"]

    input_data     = pd.read_csv(infofile)
    template_dir   = input_data["machine"].values[0]
    cores_per_node = input_data["cores_per_node"].values[0]
    run_name       = get_run_name(input_data,args.run_index)
    folder         = op.join(outdir,run_name)

    os.makedirs(os.path.join(folder,"base"), exist_ok=True)
    newinfofile = os.path.join(folder,"base","info.csv")
    copyfile(infofile,newinfofile)

    Eks, aks, dE, nreplicas = ifiles.initial_an(newinfofile)
    ifiles.setup_bash_files(op.join(input_dir,template_dir,"setup_llr_repeat.sh"),op.join(folder,"setup_llr_repeat.sh"),newinfofile)

    for f in setup_files:
        src = os.path.join(input_dir,f)
        dst = os.path.join(folder ,f)
        copyfile(src,dst)

    for infile in input_files:
        ifiles.setup_input_files(op.join(input_dir,"base","input_file"),op.join(folder,"base",infile),newinfofile)
        for i in range(nreplicas):
            replica_dir = os.path.join(folder,"base",f"Rep_{i}")
            in_replica = op.join(input_dir,"base","input_file_rep")
            out_replica  = op.join(folder,"base",f"Rep_{i}",infile)
            os.makedirs(replica_dir,exist_ok=True)
            ifiles.setup_input_files(in_replica, out_replica, newinfofile)
            ifiles.setup_initial_an_inplace(out_replica, min(Eks), max(Eks), Eks[i], dE, aks[i])

    for i in tqdm.tqdm(range(nreplicas), ncols=100, desc='Creating replicas:'):
        ifiles.setup_fxa_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_fxa"))
        ifiles.setup_nr_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_newton_raphson"),infofile)
        ifiles.setup_rm_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_robbins_monro"),infofile)

    for name in bash_files:
        ifiles.setup_batch_files(op.join(input_dir,template_dir,name),op.join(folder,name),newinfofile,cores_per_node,args)

def get_args():
    parser = ArgumentParser(description="Set up structure for LLR heatbath runs with HiRep")
    parser.add_argument("--infofile",  default=None, help="The csv file with input parameters to read")
    parser.add_argument("--machine",   default=None, help="Name of the HPC cluster")
    parser.add_argument("--partition", default=None, help="Partition to be used on the cluster")
    parser.add_argument("--account",   default=None, help="Account to be used on the cluster")
    parser.add_argument("--modules",   default=None, help="Modules to be loaded on the cluster")
    parser.add_argument("--email",     default=None, help="Email address for cluster notifications")
    parser.add_argument("--run_index" ,default=1,    help="Start index for numbering runs")
    parser.add_argument("--mpi_runner",default="srun", help="Specify mpi runner: (srun|mpirun|mpiexec)")
    parser.add_argument("--path_llr_exec",default="${HOME}", help="Specify path to LLR executable")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    infofile = args.infofile
    main(infofile,args)
