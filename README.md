
Fingerspelling ASL Project

Current state:
- EmbeddedRNN baseline
- 60 character vocabulary (59 + <blank>)
- CTC Loss
- Greedy decoding
- TensorBoard logging
- Webcam demo (basic structure)

Run training:
python -m src.train

Run webcam:
python -m src.realtime_webcam

Run training with W&B:
python -m src.train --use_wandb --wandb_project fingerspelling_asl

Optional W&B flags:
--wandb_entity <your_team_or_user>
--wandb_run_name <custom_name>
--wandb_mode offline
--wandb_tags tag1,tag2
