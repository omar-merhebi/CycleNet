source ~/.bashrc
pyenv activate PHASER


python main.py +dataset.use_channels=[DNA1] +wandb.tags=[dna_channel,predict_phase,masking,augments,no_fill] wandb.run_name=dna_only_dsktp 