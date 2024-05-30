source ~/.bashrc
pyenv activate PHASER

python main.py dataset.fill.enabled=true dataset.augment=false +wandb.tags=[all_channels,predict_phase,masking,no_augments,fill] wandb.run_name=fill_mask_noaugment_dsktp