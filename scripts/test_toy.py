from PIL import Image
from models import load_vla
import torch
import numpy as np
import time

# model = torch.nn.DataParallel(load_vla(
#         '/root/autodl-tmp/Hybrid-VLA/our_pretrain/checkpoints/latest-checkpoint.pt',
#         load_for_training=False,
#         future_action_window_size=0,
#         use_diff=True, # choose weither to use diff
#         action_dim=7,
#         )).to('cuda').eval()

model = load_vla(
        '/root/autodl-fs/Hybrid-VLA/our_pretrain/checkpoints/latest-checkpoint.pt',
        load_for_training=False,
        future_action_window_size=0,
        use_diff=True, # choose weither to use diff
        action_dim=7,
        )

model.to('cuda:0').eval()
# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16


example_image: Image.Image = Image.open('/root/autodl-fs/Hybrid-VLA/assets/000.png') 
example_prompt = "close the laptop"
example_cur_robot_state = np.array([ 0.27849028, -0.00815899,  1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ])
actions_diff, actions_ar, _ = model.predict_action(
            front_image=example_image,
            instruction=example_prompt,
            unnorm_key = 'rtx_dataset',
            cfg_scale = 0.0, 
            use_ddim = True,
            num_ddim_steps = 4,
            action_dim = 7,
            cur_robot_state = example_cur_robot_state,
            predict_mode = 'diff+ar'
            )
    
print(actions_diff)
