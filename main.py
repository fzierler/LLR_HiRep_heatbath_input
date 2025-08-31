#!/usr/bin/env python
# coding: utf-8
import llrinput.inputfiles as ifiles
import llrinput.provenance as pv
import pandas as pd 
import os.path as op
import os
from shutil import copyfile
from argparse import ArgumentParser
import tqdm

## create a suitable name for the run:
def get_run_name(input_data_row,ind):
    Lt  = input_data_row["Lt"]
    Ls  = input_data_row["Ls"]
    Rep = input_data_row["n_replicas"]
    return f"Run_{ind}_LLR_{Lt}x{Ls}_{Rep}"

def main(infofile,args):
    outdir      = args.output_run_dir
    file_dir    = os.path.dirname(os.path.realpath(__file__))
    input_dir   = op.join(file_dir,"input/templates/")
    bash_files  = ["sp4_llr_therm.sh","sp4_llr_newton_raphson.sh","sp4_llr_robbins_monro.sh","sp4_llr_fxa.sh"]
    input_files = ["input_file_therm", "input_file_newton_raphson", "input_file_robbins_monro", "input_file_fxa"]
    setup_files = ["list_configs.sh","update_replicas.sh"]
    input_data  = pd.read_csv(infofile)
    index       = args.run_index

    for row_ind, input_data_row in input_data.iterrows():
        template_dir   = "local" if args.machine == "local" else "generic"
        cores_per_node = input_data_row["cores_per_node"]
        run_name       = get_run_name(input_data_row,index)
        folder         = op.join(outdir,run_name)
        index         += 1

        os.makedirs(os.path.join(folder,"base"), exist_ok=True)
        info_base = os.path.join(folder,"base","info.csv")

        io = open(info_base, "w")
        print(pv.provenance_string("#"), end='',file=io)
        io.close()
        input_data[row_ind:row_ind+1].to_csv(info_base,index=False,mode="a")
        
        Eks, aks, dE, nreplicas = ifiles.initial_an(input_data_row,file_dir)
        ifiles.setup_bash_files(op.join(input_dir,template_dir,"setup_llr_repeat.sh"),op.join(folder,"setup_llr_repeat.sh"),input_data_row)

        for f in setup_files:
            src = os.path.join(input_dir,f)
            dst = os.path.join(folder ,f)
            copyfile(src,dst)

        for infile in input_files:
            ifiles.setup_input_files(op.join(input_dir,"base","input_file"),op.join(folder,"base",infile),input_data_row)
            for i in range(nreplicas):
                replica_dir = os.path.join(folder,"base",f"Rep_{i}")
                in_replica = op.join(input_dir,"base","input_file_rep")
                out_replica  = op.join(folder,"base",f"Rep_{i}",infile)
                os.makedirs(replica_dir,exist_ok=True)
                ifiles.setup_input_files(in_replica, out_replica,input_data_row)
                ifiles.setup_initial_an_inplace(out_replica, min(Eks), max(Eks), Eks[i], dE, aks[i])
        # [end-for-loop] input_files

        print("Generating replicas in Run -->: " + run_name)

        for i in tqdm.tqdm(range(nreplicas), ncols=100, desc='Creating replicas:'):
            ifiles.setup_fxa_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_fxa"))
            ifiles.setup_nr_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_newton_raphson"),input_data_row)
            ifiles.setup_rm_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_robbins_monro"),input_data_row)
        # [end-for-looo] nreplicas

        for name in bash_files:
            ifiles.setup_batch_files(op.join(input_dir,template_dir,name),op.join(folder,name),input_data_row,cores_per_node,args)

def get_args():
    parser = ArgumentParser(description="Set up structure for LLR heatbath runs with HiRep")
    parser.add_argument("--input_params_csv",  default=None, help="The csv file with input parameters to read")
    parser.add_argument("--machine",   default=None, help="Name of the HPC cluster")
    parser.add_argument("--partition", default=None, help="Partition to be used on the cluster")
    parser.add_argument("--qos",       default=None, help="QoS of the partition to be used on the cluster")
    parser.add_argument("--account",   default=None, help="Account to be used on the cluster")
    parser.add_argument("--modules",   default=None, help="Modules to be loaded on the cluster")
    parser.add_argument("--email",     default=None, help="Email address for cluster notifications")
    parser.add_argument("--run_index" ,default=1, type=int, help="Start index for numbering runs (default=1)")
    parser.add_argument("--mpi_runner",default="srun", help="Specify mpi runner: (srun|mpirun|mpiexec), default=srun")
    parser.add_argument("--path_llr_exec" ,default="${HOME}" , help="Specify path to LLR executable (default=$HOME)")
    parser.add_argument("--output_run_dir",default="./output", help="Specify path to the output directory (default=./output)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    infofile = args.input_params_csv
    main(infofile,args)