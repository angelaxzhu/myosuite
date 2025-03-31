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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")
nb_seed = 1

torso = False
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

def group(l):
    grouped = []
    g=0
    mapping = {
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
    while g < 24:
         #the values for each actuator within the group is the same -- take only first (or else when average, bias ... is this bad?)
        index = mapping[g][0]
        grouped.append(l[index])
        g= g+1
    return grouped

env_name_exo = 'myoTorsoExoFixed-v0'
env_name_torso ='myoTorsoFixed-v0'
model_num_exo = '2025_03_29_00_11_297'
model_num_torso = '2025_03_20_16_19_277'


angle = 00
model_exo = SAC.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name_exo + '/' + model_num_exo + r'/best_model')
model_torso = SAC.load(path+'/standingBalance/policy_best_model'+'/'+env_name_torso+'/'+model_num_torso+r'/best_model')

def testModel(all_activation, all_ctrl, all_activation_force,episode,env_name,model):
    env = gym.make(env_name)
    env = ActionSpaceWrapper(env)
    s, m, t = [], [], []   
    env.reset()
    random.seed() 
    m_act = []
    all_rewards = []
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    max_ep = 200
    act_all = []
    ctrl_all = []
    act_f_all = []
    while (not done) and (step < 200):
        obs = env.unwrapped.obsdict2obsvec(env.unwrapped.obs_dict, env.unwrapped.obs_keys)[1]
        action , _ = model.predict(obs, deterministic= True)
        obs, reward, done, info, obs_dict = env.step(action)
        ep_rewards.append(reward)
        m.append(action)
        #Add time step  
        act_all.append(group(obs_dict["obs_dict"]["act"]))
        ctrl_all.append(group(obs_dict["obs_dict"]["ctrl"]))
        act_f_all.append(group(obs_dict["obs_dict"]["act_f"]))
        step += 1

    all_rewards.append(np.sum(ep_rewards))
    m_act.append(muscle_act)

    #in case it ends before 200 steps
    act_all.extend([0]*(max_ep - len(act_all)))
    ctrl_all.extend([0]*(max_ep - len(ctrl_all)))
    act_f_all.extend([0]*(max_ep - len(act_f_all)))
    
    #Add episodes to the rest 
    all_activation.append(act_all)
    all_ctrl.append(ctrl_all)
    all_activation_force.append(act_f_all)


    print(f"Average reward: {np.mean(all_rewards)}")
    print(f"Average angle: {np.mean(obs_dict['obs_dict']['qpos'])}")
    return all_activation, all_ctrl, all_activation_force
def calculate(all_activation,all_ctrl):
    #calculate
    ####Fig 1 --> Average activation force over actuators####
    ave_act_overact = np.mean(all_activation, axis = 2)
    #sum over all time steps
    ave_act_tot = np.sum(ave_act_overact,axis = 1)
    #average over all ep
    ave_act = np.mean(ave_act_tot)
    std_act = np.std(ave_act_tot)
    print(f"fig1 size 1 {np.shape(ave_act_overact)}")
    print(f"fig1 size {np.shape(ave_act_tot)}")
   


    #Fig 2 --> Average activation force over episodes
    ave_act_overep = np.mean(all_activation,axis=0)
    std_act_overep = np.std(all_activation,axis=0)
    #std
    print(f"fig2 size {np.shape(ave_act_overep)}")


    #Fig 3 --> Average ctrl over episodes
    ave_ctrl_overep = np.mean(all_ctrl, axis=0)
    std_ctrl_overep = np.std(all_ctrl,axis=0)

    print(f"fig3 size {np.shape(ave_ctrl_overep)}")
    #Group according to wrapper 

    #Fig 4 --> Extract activation force of tendons 
    return ave_act, std_act, ave_act_overep,std_act_overep,ave_ctrl_overep,std_ctrl_overep
	
#number of episodes x muscles
all_activation_exo = []
all_activation_torso=[]
all_ctrl_exo = []
all_ctrl_torso = []
all_activation_force_exo = []
all_activation_force_torso = [] 
num_actuators = 0
episode = 0 

while(episode < 100):
    print(f"episode: {episode}")
    testModel(all_activation_exo,all_ctrl_exo,all_activation_force_exo,episode,env_name_exo,model_exo)
    testModel(all_activation_torso,all_ctrl_torso,all_activation_force_torso,episode,env_name_torso,model_torso)
    episode = episode +1

ave_act_exo, std_act_exo,ave_act_overep_exo,std_act_overep_exo,ave_ctrl_overep_exo,std_ctrl_overep_exo = calculate(all_activation_exo,all_ctrl_exo)
ave_act_torso, std_act_torso,ave_act_overep_torso,std_act_overep_torso,ave_ctrl_overep_torso,std_ctrl_overep_torso = calculate(all_activation_torso,all_ctrl_torso)

print(f"Average Activation with Exosuit {ave_act_exo} std {std_act_exo}")
print(f"Average Activation without Exosuit {ave_act_torso} std {std_act_torso}")



#plot
g = 0
muscle_group_name = ['psoas major right', 'psoas major left','RA, right', 'RA left','ILpL right','ILpL left','ILpT right','ILpT left','LTpT right','LTpT left','LTpL right','LTpL left','QL_post right','QL_post left'
            ,'QL_mid right'
            ,'QL_mid left'
            ,'QL_ant left'
            ,'MF right'
            ,'MF left'
            ,'EO right'
            ,'IO right'
            ,'EO left', 'IO left']

while g < 23:
    actuator_group = muscle_group_name[g]
    plt.figure()
    x = np.arange(200)

    plt.plot(x,ave_act_overep_exo[:,g],linestyle = '--',label="With exosuit")
    plt.fill_between(x,ave_act_overep_exo[:,g]-std_act_overep_exo[:,g],ave_act_overep_exo[:,g]+std_act_overep_exo[:,g],color='blue',alpha=0.3)

    plt.plot(x,ave_act_overep_torso[:,g],linestyle = '-',label="Without exosuit")
    plt.fill_between(x,ave_act_overep_torso[:,g]-std_act_overep_torso[:,g],ave_act_overep_torso[:,g]+std_act_overep_torso[:,g],color='pink',alpha=0.3)

    plt.title(f"Muscle Activation Over Time at {angle} degree angle for group {actuator_group}")
    plt.xlabel("Timestep")
    plt.ylabel("Activation")
    plt.legend()
    plt.savefig(path+'/data' +'/' + str(angle) + '_' + str(g)+'activation.png')
    plt.close()

    ave_ctrl_overep_exo[:,g].tofile(path+'/data' +'/' + 'exo' + str(angle) + '_' + str(g) + 'ctrl.csv', sep=",")
    g=g+1

#save
#os.makedirs(path+'/data' +'/' + env_name + model_num, exist_ok=True)
#ave_ctrl_overep.tofile(path+'/data' +'/' + env_name + model_num +'ctrl.csv', sep=",")



