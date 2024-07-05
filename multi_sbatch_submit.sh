python multi_sbatch.py --env vit_attn \
                       --nhrs 8 \
                       --output-dirname slurm_files \
                       --qos scav \
                       --partition nexus \
                       --gpu 8 --gpu-type a5000 \
                       --cores 1 \
                       --mem 128 \
                       