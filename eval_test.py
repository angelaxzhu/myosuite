import gym
from myosuite.utils import gym
import numpy as np
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import skvideo
import skvideo.io
import os
import random
from tqdm.auto import tqdm
import warnings

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")

nb_seed = 1

torso = False
movie = True
path = './'

class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.syn_action_shape = 24
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.syn_action_shape,),dtype=np.float32)
        #self.observation_space = env.observation_space
        
        # Define the mapping from reduced to original action space
        self.action_mapping = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #psoas major right
            1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  #psoas major left
            2: [22], # RA, right
            3: [23], #RA left
            4: [24, 25, 26, 27], #ILpL right
            5: [28, 29, 30, 31], #ILpL left
            6: [32, 33, 34, 35, 36, 37, 38, 39],  #ILpT right
            7: [40, 41, 42, 43, 44, 45, 46, 47], #ILpT left
            8: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], #LTpT right
            9: [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89], #LTpT left
            10: [90, 91, 92, 93, 94], #LTpL right
            11: [95, 96, 97, 98, 99], #LTpL left
            12: [100, 101, 102, 103, 104, 105, 106], #QL_post right
            13: [107, 108, 109, 110, 111, 112, 113],  #QL_post left
            14: [114, 115, 116, 117, 118],  #QL_mid right
            15: [119, 120, 121, 122, 123],  #QL_mid left
            16: [124, 125, 126, 127, 128, 129 ], #QL_ant right
            17: [130, 131, 132, 133, 134, 135], #QL_ant left
            18: [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160], #MF right
            19: [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185], #MF left
            20: [186, 187, 188, 189, 190, 191], #EO right
            21: [192, 193, 194, 195, 196, 197], #IO right
            22: [198, 199, 200, 201, 202, 203], #EO left
            23: [204, 205, 206, 207, 208, 209] #IO left
        }

    def action(self, action):
        assert len(action) == len(self.action_mapping)

        full_action = np.zeros(self.env.action_space.shape)
        for i, indices in self.action_mapping.items():
            full_action[indices] = action[i]
        return full_action

env_name = 'myoTorsoFixed-v0'

model_num = '2025_03_20_16_19_277'
#model = PPO.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name + '/' + model_num + r'/best_model')
model = SAC.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name + '/' + model_num + r'/best_model')


#model = PPO.load('ep_train_results')
env = gym.make(env_name)
env = ActionSpaceWrapper(env)
s, m, t = [], [], []

env.reset()

random.seed() 

frames = []
view = 'side'
m_act = []
all_rewards = []
for _ in tqdm(range(1)):
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    while (not done) and (step < 200):
          obs = env.unwrapped.obsdict2obsvec(env.unwrapped.obs_dict, env.unwrapped.obs_keys)[1]
          action , _ = model.predict(obs, deterministic= True)
          obs, reward, done, info, _ = env.step(action)
          ep_rewards.append(reward)
          m.append(action)
          if movie:
                  geom_1_indices = np.where(env.unwrapped.sim.model.geom_group == 1)
                  env.unwrapped.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  frame = env.unwrapped.sim.renderer.render_offscreen(width= 640, height=480,camera_id="front_view")
                  
                  #frame = (frame).astype(np.uint8)
                  frame = np.flipud(frame)
            # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
                  frames.append(frame[::-1,:,:])
                  #env.sim.mj_render(mode='window') # GUI
          step += 1
    all_rewards.append(np.sum(ep_rewards))
    m_act.append(muscle_act)
print(f"Average reward: {np.mean(all_rewards)}")


if movie:
    os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	