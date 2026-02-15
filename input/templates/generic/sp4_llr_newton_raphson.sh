#!/bin/bash 

#SBATCH --account=
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --time=

#SBATCH --nodes=
#SBATCH --ntasks=
#SBATCH --ntasks-per-node=
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

# Use modules to setup the runtime environment
module load

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

###
bash ../update_replicas.sh -r 19 -i input_file_newton_raphson
srun -n 76 ../../llr_hb -i input_file_newton_raphson
