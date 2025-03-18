#!/usr/bin/env python
# coding: utf-8
import llrinput.inputfiles as ifiles
import pandas as pd 
import os.path as op
import os
from shutil import copyfile

outdir     = "./output/"
input_dir  = "./input/templates/"
infofile   = "./input/local_tests.csv"

bash_files  = ["sp4_llr_start.sh","sp4_llr_start_cont.sh","sp4_llr_cont.sh","sp4_llr_fxa.sh"]
input_files = ["input_file_start", "input_file_start_cont", "input_file_cont", "input_file_fxa"]
setup_files = ["list_configs.sh","setup_replicas_start.sh","setup_replicas_start_cont.sh","update_replicas.sh","setup_replicas_fxa.sh"]
## create a suitable name for the run:
def get_run_name(input_data):
    Lt  = input_data["Lt"].values[0]
    Ls  = input_data["Ls"].values[0]
    Rep = input_data["n_replicas"].values[0]
    return f"LLR_{Lt}x{Ls}_{Rep}"

input_data     = pd.read_csv(infofile)
template_dir   = input_data["machine"].values[0]
cores_per_node = input_data["cores_per_node"].values[0]
run_name       = get_run_name(input_data)
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
    ifiles.setup_input_files(op.join(input_dir,"base","input_file"),     op.join(folder,"base",infile)         ,newinfofile)
    for i in range(nreplicas):
        replica_dir = os.path.join(folder,"base",f"Rep_{i}")
        in_replica = op.join(input_dir,"base","input_file_rep")
        out_replica  = op.join(folder,"base",f"Rep_{i}",infile)
        os.makedirs(replica_dir,exist_ok=True)
        ifiles.setup_input_files(in_replica, out_replica, newinfofile)
        ifiles.setup_initial_an_inplace(out_replica, min(Eks), max(Eks), Eks[i], dE, aks[i])

for i in range(nreplicas):
    ifiles.setup_fxa_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_fxa"))
    ifiles.setup_nr_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_start_cont"))
    ifiles.setup_rm_input_inplace(op.join(folder,"base",f"Rep_{i}","input_file_cont"))

for name in bash_files:
    ifiles.setup_batch_files(op.join(input_dir,template_dir,name),op.join(folder,name),newinfofile,cores_per_node)