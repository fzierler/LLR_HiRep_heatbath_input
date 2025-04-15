from shutil import copyfile
import os.path as op

dir_in  = "/home/fabian/Downloads/LLR_cont/sp4_5x48_48"
dir_out = "/home/fabian/Downloads/LLR_cont/LLR_5x48_48"
N_repeats  = 25
N_replicas = 48

for i in range(N_repeats):
    for j in range(N_replicas):
        file_in  = op.join(dir_in ,f"{i}",f"Rep_{j}","out_0")
        file_out = op.join(dir_out,f"{i}",f"Rep_{j}","out_0")
        copyfile(file_in,file_out)