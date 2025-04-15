#!/bin/bash 

#SBATCH --exclusive
#SBATCH -p compute
#SBATCH -t 3-00:00:00          
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40

#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fabian.zierler@swansea.ac.uk

### 
bash ../update_replicas.sh -r 19 -i input_file_therm 
mpirun -n 76 ../../llr_hb -i input_file_therm
