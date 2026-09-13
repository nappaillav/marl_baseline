sbatch launcher.sh sc2_v2_protoss_10v10
sbatch launcher.sh sc2_v2_terran_10v10
sbatch launcher.sh sc2_v2_zerg_10v10

# 5 seeds
sbatch launcher_deepsetmix.sh 
sbatch launcher_deepsetmix.sh 
sbatch launcher_deepsetmix.sh 
sbatch launcher_deepsetmix.sh 
sbatch launcher_deepsetmix.sh 