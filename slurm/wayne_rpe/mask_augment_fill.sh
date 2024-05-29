source ~/.bashrc
pyenv activate PHASER

python main.py dataset.fill.enabled=true +wandb.tags=[all_channels,predict_phase,masking,augments,fill] wandb.run_name=fill_mask_augment_dsktp