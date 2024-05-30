source ~/.bashrc
pyenv activate PHASER

python main.py dataset.fill.enabled=false dataset.augment=false dataset.use_masks=null +wandb.tags=[all_channels,predict_phase,no_masking,no_augments,no_fill] wandb.run_name=nofill_nomask_noaugment_dsktp