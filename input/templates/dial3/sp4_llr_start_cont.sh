#!/bin/bash 

#SBATCH --account=dp208
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128

#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fabian.zierler@swansea.ac.uk

# Use modules to setup the runtime environment
module purge                               # Removes all modules still loaded
module load gcc/10.3.0
module load openmpi/4.0.5

###
bash ../setup_replicas_start_cont.sh -r 19
srun -n 76 ../../llr_hb
