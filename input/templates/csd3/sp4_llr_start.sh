#!/bin/bash 

#SBATCH -A DIRAC-DP208-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=76
#SBATCH --ntasks-per-node=76

#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fabian.zierler@swansea.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment
module load gcc/11

### 
srun -n 76 ../../llr_hb -i input_file_start
