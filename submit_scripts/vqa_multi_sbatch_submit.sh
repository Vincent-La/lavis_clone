python vqa_multi_sbatch.py --env vit_small_batch \
                           --nhrs 8 \
                           --qos scav \
                           --partition nexus \
                           --gpu 8 --gpu-type a6000 \
                           --cores 1 \
                           --mem 128 \
                           --base-dir /fs/nexus-scratch/vla/LAVIS \
                           --output-dirname vqa_slurm_files \

# TODO: set base-dir to project-dir