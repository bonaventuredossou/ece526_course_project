#!/bin/bash
#SBATCH --job-name=ecse_526_baseline_normal
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-gpu=18
#SBATCH --mem=96G
#SBATCH --time=168:00:00         
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/ece526_course_project/slurmerror.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/ece526_course_project/slurmoutput.txt


###########cluster information above this line
module load python/3.9 cuda/10.2/cudnn/7.6
source /home/mila/b/bonaventure.dossou/env/bin/activate
cd /home/mila/b/bonaventure.dossou/ece526_course_project/src
python main.py