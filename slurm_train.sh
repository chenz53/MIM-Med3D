# For example, using AWS g5.48xlarge instance for slurm training 2 days
# brats data pretraining using SimMIM on t1ce modality
sbatch --ntasks-per-node=192 \
       --partition=g5-on-demand \
       --time=2-00:00:00  \
       --gres=gpu:8 \
       --constraint="[g5.48xlarge]" \
       --wrap="sh train.sh code/experiments/ssl/simmim_pretrain_main.py code/configs/ssl/brats/vitsimmim_base_m0.75_t1ce.yaml"