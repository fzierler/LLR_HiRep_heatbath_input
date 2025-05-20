#!/bin/bash 

#SBATCH --account=
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --time=

#SBATCH --nodes=
#SBATCH --ntasks=
#SBATCH --ntasks-per-node=

#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

# Use modules to setup the runtime environment
module purge
module load

###
bash ../update_replicas.sh -r 19 -i input_file_newton_raphson
srun -n 76 ../../llr_hb -i input_file_newton_raphson
