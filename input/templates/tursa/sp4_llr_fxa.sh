#!/bin/bash 

#SBATCH --account=dp208
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --qos=standard
#SBATCH --partition=cpu
#SBATCH -o %x.out
#SBATCH -e %x.err

# Use modules to setup the runtime environment
module load gcc/9.3.0 openmpi/4.1.5

###
bash ../update_replicas.sh -r 19 -i input_file_fxa
srun -n 76 ../../llr_hb -i input_file_fxa
