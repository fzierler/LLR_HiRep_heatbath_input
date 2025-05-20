#!/bin/bash 

#SBATCH --account=dp208
#SBATCH --partition=high
#SBATCH --time=4-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128

#SBATCH -o %x_%J_%t.out
#SBATCH -e %x_%J_%t.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fabian.zierler@swansea.ac.uk

# Use modules to setup the runtime environment
module purge
module load gcc/10.3.0 openmpi/4.0.5

###
bash ../update_replicas.sh -r 19 -i input_file_robbins_monro 
srun -n 76 ../../llr_hb -i input_file_robbins_monro
