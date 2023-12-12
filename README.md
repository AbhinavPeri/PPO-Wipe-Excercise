In order to train PPO on the wipe environment, run  `python ppo_continuous_action.py --track --env-id=Wipe --anneal-lr=False --clip-vloss=False --exp_name "Insert Experiment Name"`

In order to visualize, run `python visualize_model.py`. Make sure to modify the checkpoint path in the script. You can find it in the runs folder. The extension of the checkpoitn file should be .pt
