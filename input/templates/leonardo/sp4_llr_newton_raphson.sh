#!/bin/bash 

#SBATCH --account=EUHPC_B22_046_0
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --partition=dcgp_usr_prod

#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fabian.zierler@swansea.ac.uk

# Use modules to setup the runtime environment
module purge
module load cuda/12.2 nvhpc/23.11 fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0 hdf5

###
bash ../update_replicas.sh -r 19 -i input_file_newton_raphson
srun -n 76 ../../llr_hb -i input_file_newton_raphson
