source ~/.bashrc
pyenv activate PHASER

python main.py dataset.augment=false +wandb.tags=[all_channels,predict_phase,masking,no_augments,no_fill] wandb.run_name=noaugments_dsktp