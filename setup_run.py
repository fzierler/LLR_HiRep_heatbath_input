#!/usr/bin/env python
# coding: utf-8

import llranalysis.inputfiles as ifiles
import pandas as pd 
import os

outdir    = "./output/"
input_dir = "./input/templates/"
infofile  = "./input/local_tests.csv"
dial = False

if dial:
    cores_per_node = 128
    bash_files = ["sp4_llr_start_dial.sh","sp4_llr_start_cont_dial.sh","sp4_llr_cont_dial.sh","sp4_llr_fxa_dial.sh"]
else:
    cores_per_node = 76
    bash_files = ["sp4_llr_start.sh","sp4_llr_start_cont.sh","sp4_llr_cont.sh","sp4_llr_fxa.sh"]

# create a suitable name for the run:
input_data = pd.read_csv(infofile)
Lt  = input_data["Lt"].values[0]
Ls  = input_data["Ls"].values[0]
Rep = input_data["n_replicas"].values[0]
run_name = "LLR_"+str(Lt)+"x"+str(Ls)+"_"+str(Rep)
folder = os.path.join(outdir,run_name)

ifiles.predat_from_csv(folder,infofile)
ifiles.copy_identical_files(folder,input_dir)

infile  = os.path.join(input_dir,"base","input_file")
outfile = os.path.join(folder,"base","input_file")
ifiles.input_files_from_csv(infile,outfile,infofile)

infile  = os.path.join(input_dir,"base","input_file_rep")
outfile = os.path.join(folder,"base","input_file_rep")
ifiles.input_files_from_csv(infile,outfile,infofile)

infile  = os.path.join(input_dir,"setup_llr_repeat.sh")
outfile = os.path.join(folder,"setup_llr_repeat.sh")
ifiles.input_files_from_csv(infile,outfile,infofile)

for name in bash_files:
    infile  = os.path.join(input_dir,name)
    outfile = os.path.join(folder,name.replace('_dial',''))
    ifiles.csd3_batch_files(infile,outfile,infofile,cores_per_node)
