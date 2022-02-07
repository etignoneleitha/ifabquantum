#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8300MB 
#SBATCH --account=IscrC_BAT-QuO
#SBATCH --partition=m100_usr_prod
#SBATCH --time=00:10:00
#SBATCH --error=job.%j.err
#SBATCH --output=job.%j.out
#SBATCH --job-name=nome_job

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module list
module load autoload profile/quantum
module load autoload qiskit
module load pulser

python main_qiskit.py  --Nbayes 40 --p 4