import os
from shutil import copyfile

dir_old = "~/Downloads/missing_poly/sp4_5x64_95"
dir_new = "~/Downloads/missing_poly/LLR_5x64_95"

for repeat in range(20):
    for replica in range(95):
        old_file = os.path.join(os.path.expanduser(dir_old),str(repeat),f"Rep_{replica}","out_0")
        new_file = os.path.join(os.path.expanduser(dir_new),str(repeat),f"Rep_{replica}","out_0")
        copyfile(old_file,new_file)