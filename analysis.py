import gym
from myosuite.utils import gym
import numpy as np
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import os
import random
from tqdm.auto import tqdm
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import scipy

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
#Group muscles
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
         #the values for each actuator within the group is the same -- take only first 
        index = mapping[g][0]
        grouped.append(l[index])
        g= g+1
    return grouped
#USER INPUTS
env_name_exo = 'myoTorsoExoFixed-v0'
env_name_torso ='myoTorsoFixed-v0'
model_num_exo = '2025_03_29_00_11_297'
model_num_torso = '2025_03_19_15_39_247'


angle = 0
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
    time_all = []
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
        time_all.append((obs_dict["obs_dict"]["time"]))
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
     

    #print(f"Average reward: {np.mean(all_rewards)}")
    #print(f"Average angle: {np.mean(obs_dict['obs_dict']['qpos'])}")
    return all_activation, all_ctrl, all_activation_force
def ungroup(all_activation):
    #all_activation= episodes x timesteps x actuators 
    ep=0
    while ep < len(all_activation):
        t=0
        while t < len(all_activation[0]):
            all_activation[ep][t][0] =  all_activation[ep][t][0]*11
            all_activation[ep][t][1] =  all_activation[ep][t][1]*11
            all_activation[ep][t][2] =  all_activation[ep][t][2]*1
            all_activation[ep][t][3] =  all_activation[ep][t][3]*1
            all_activation[ep][t][4] =  all_activation[ep][t][4]*4
            all_activation[ep][t][5] =  all_activation[ep][t][5]*4
            all_activation[ep][t][6] =  all_activation[ep][t][6]*8
            all_activation[ep][t][7] =  all_activation[ep][t][7]*8
            all_activation[ep][t][8] =  all_activation[ep][t][8]*21
            all_activation[ep][t][9] =  all_activation[ep][t][9]*21
            all_activation[ep][t][10] =  all_activation[ep][t][10]*5
            all_activation[ep][t][11] =  all_activation[ep][t][11]*5
            all_activation[ep][t][12] =  all_activation[ep][t][12]*7
            all_activation[ep][t][13] =  all_activation[ep][t][13]*7
            all_activation[ep][t][14] =  all_activation[ep][t][14]*5
            all_activation[ep][t][15] =  all_activation[ep][t][15]*5
            all_activation[ep][t][16] =  all_activation[ep][t][16]*6
            all_activation[ep][t][17] =  all_activation[ep][t][17]*6
            all_activation[ep][t][18] =  all_activation[ep][t][18]*25
            all_activation[ep][t][19] =  all_activation[ep][t][19]*25
            all_activation[ep][t][20] =  all_activation[ep][t][20]*6
            all_activation[ep][t][21] =  all_activation[ep][t][21]*6
            all_activation[ep][t][22] =  all_activation[ep][t][22]*6
            all_activation[ep][t][23] =  all_activation[ep][t][23]*6
            t=t+1
        ep = ep+1
    return all_activation
    
    return 0 
def calculate(all_activation,all_ctrl):
    #calculate
    ###FIGURE 1: METABOLIC COST###
    #Average activation force over actuators
    all_activation_ungrouped = ungroup(np.square(all_activation))
    ave_act_overact = np.sum(all_activation_ungrouped,axis=2)/210
    #sum over all time steps
    ave_act_tot = np.sum(ave_act_overact,axis = 1)
    #average over all ep
    ave_act = np.mean(ave_act_tot)
    std_act = np.std(ave_act_tot)
    print(f"fig1 size {np.shape(ave_act_tot)}")
   

    ###FIGURE 2: ACTIVATION FORCE OVER TIME
    #Average activation force over episodes
    ave_act_overep = np.mean(all_activation,axis=0)
    std_act_overep = np.std(all_activation,axis=0)
    print(f"fig2 size {np.shape(ave_act_overep)}")


    ###FIGURE 3: EXCITATION SIGNAL (CTRL)
    #Average ctrl over episodes
    ave_ctrl_overep = np.mean(all_ctrl, axis=0)
    std_ctrl_overep = np.std(all_ctrl,axis=0)
    #(timesteps, muscles)
    ctrl_RL = {"LT-R":[ave_ctrl_overep[:,8],std_ctrl_overep[:,8]],"LT-L":[ave_ctrl_overep[:,9],std_ctrl_overep[:,9]],"LL-R":[ave_ctrl_overep[:,10],std_ctrl_overep[:,10]],"LL-L":[ave_ctrl_overep[:,11],std_ctrl_overep[:,11]],"IL-R":[ave_ctrl_overep[:,4],std_ctrl_overep[:,4]],"IL-R2":[ave_ctrl_overep[:,6],std_ctrl_overep[:,6]]}
    print(f"fig3 size {np.shape(ave_ctrl_overep)}")
    #extract only relevant group

    #Fig 4 --> Extract activation force of tendons (INCOMPLETE)
    return ave_act, std_act, ave_act_overep,std_act_overep,ctrl_RL
def emg_data():
    '''
        This was to extract the data processed by Anto from .csv files.
        Probably not relevant anymore.
    '''
    fourty_emg = {"LT-R":[0,0,0,0,0,0,0,0,0],"LT-L":[0,0,0,0,0,0,0,0,0],"LL-R":[0,0,0,0,0,0,0,0,0],"LL-L":[0,0,0,0,0,0,0,0,0],"IL-R":[0,0,0,0,0,0,0,0,0]}
    sixty_emg =  {"LT-R":[0,0,0,0,0,0,0,0,0],"LT-L":[0,0,0,0,0,0,0,0,0],"LL-R":[0,0,0,0,0,0,0,0,0],"LL-L":[0,0,0,0,0,0,0,0,0],"IL-R":[0,0,0,0,0,0,0,0,0]}
    eighty_emg= {"LT-R":[0,0,0,0,0,0,0,0,0],"LT-L":[0,0,0,0,0,0,0,0,0],"LL-R":[0,0,0,0,0,0,0,0,0],"LL-L":[0,0,0,0,0,0,0,0,0],"IL-R":[0,0,0,0,0,0,0,0,0]}
    #iterate through folders in data folder 
    subj_no=0
    max_length = 76939
    for subj in os.listdir("./data/processed_emg"):
        subjname = os.fsdecode(subj)
        print(subjname)
        #Extract Data
        counter=  0
        for angle in os.listdir("./data/processed_emg/"+subjname):
            file = os.fsdecode(angle)
            df = pd.read_csv("./data/processed_emg/"+subjname + "/" + file)
            if counter ==0:
                fourty_emg["LT-R"][subj_no]=df.iloc[:,0].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,0])))
                fourty_emg["LT-L"][subj_no]=df.iloc[:,1].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,1])))
                fourty_emg["LL-R"][subj_no]=df.iloc[:,2].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,2])))
                fourty_emg["LL-L"][subj_no]=df.iloc[:,3].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,3])))
                fourty_emg["IL-R"][subj_no]=df.iloc[:,4].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,4])))
            elif counter ==1:
                sixty_emg["LT-R"][subj_no]=df.iloc[:,0].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,0])))
                sixty_emg["LT-L"][subj_no]=df.iloc[:,1].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,1])))
                sixty_emg["LL-R"][subj_no]=df.iloc[:,2].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,2])))
                sixty_emg["LL-L"][subj_no]=df.iloc[:,3].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,3])))
                sixty_emg["IL-R"][subj_no]=df.iloc[:,4].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,4])))
            else:
                eighty_emg["LT-R"][subj_no]=df.iloc[:,0].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,0])))
                eighty_emg["LT-L"][subj_no]=df.iloc[:,1].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,1])))
                eighty_emg["LL-R"][subj_no]=df.iloc[:,2].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,2])))
                eighty_emg["LL-L"][subj_no]=df.iloc[:,3].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,3])))
                eighty_emg["IL-R"][subj_no]=df.iloc[:,4].to_list()+ ([np.nan] * (max_length - len(df.iloc[:,4])))
            counter=counter+1
        subj_no = subj_no +1
    
    #Average over eubjcts
    for key in fourty_emg:
        #instead of padding with 0, will only average those who have a value -- measurement stopped, not actual 0 emg
        #turn into array with each row is a subject, and each column a data point
        all_val = np.array(fourty_emg[key])
        fourty_emg[key]=[np.nanmean(all_val[:,:],axis=0),np.nanstd(all_val[:,:],axis=0)]
    for key in sixty_emg:
        all_val = np.array(sixty_emg[key])
        sixty_emg[key]=[np.nanmean(all_val[:,:],axis=0),np.nanstd(all_val[:,:],axis=0)]
    for key in eighty_emg:
        all_val = np.array(eighty_emg[key])
        eighty_emg[key]=[np.nanmean(all_val[:,:],axis=0),np.nanstd(all_val[:,:],axis=0)]
    #dictionary muscle: arr(ave),arr(std)
    return fourty_emg,sixty_emg,eighty_emg 
def plot_emg(fourty_emg,sixty_emg,eighty_emg,ctrl):
    if angle == 40:
        emg = fourty_emg 
    elif angle == 60:
        emg = sixty_emg
    elif angle == 80:
        emg = eighty_emg 
    else:
        print("angle 0")
        return 0
    for key in emg:
        plt.figure()
        x = np.arange(76939)
        x_in_seconds = x*(1/20480) #2048Hz
        #print(x_in_seconds)
        plt.plot(x_in_seconds,emg[key][0],linestyle = '--',label="EMG")
        plt.fill_between(x_in_seconds,emg[key][0]-emg[key][1],emg[key][0]+emg[key][1],color='blue',alpha=0.3)

        #from examinin data.time, seems like each step is 0.025 seconds
        x2 = np.arange(200)
        x2_in_seconds = x2*0.025
        plt.plot(x2_in_seconds,ctrl[key][0],linestyle = '-',label="Model")
        plt.fill_between(x2_in_seconds,ctrl[key][0]-ctrl[key][1],ctrl[key][0]+ctrl[key][1],color='pink',alpha=0.3)

        plt.title(f"Muscle Activation at {angle} Degree for Group {key}")
        plt.xlabel("Time (s)")
        plt.ylabel("Muscle Activation")
        plt.legend()
        plt.savefig(path+'/data' +'/' + str(angle) + '_' + str(key)+'emg.png')
        plt.close()

    return 0
def plot_activation(ave_act_overep_exo,std_act_overep_exo,ave_act_overep_torso,std_act_overep_torso):
    #plot
    g = 0
    muscle_group_name = ['psoas major right', 'psoas major left','RA right', 'RA left','ILpL right','ILpL left','ILpT right','ILpT left','LTpT right','LTpT left','LTpL right','LTpL left','QL_post right','QL_post left'
                ,'QL_mid right'
                ,'QL_mid left'
                ,'QL_ant right'
                ,'QL_ant left'
                ,'MF right'
                ,'MF left'
                ,'EO right'
                ,'IO right'
                ,'EO left', 'IO left']

    while g < 24:
        actuator_group = muscle_group_name[g]
        plt.figure()
        
        x = np.arange(200)
        x_in_seconds = x*0.025
        plt.plot(x_in_seconds,ave_act_overep_exo[:,g],linestyle = '--',label="With exosuit",color='darkblue',linewidth=2)
        plt.fill_between(x_in_seconds,ave_act_overep_exo[:,g]-std_act_overep_exo[:,g],ave_act_overep_exo[:,g]+std_act_overep_exo[:,g],color='blue',alpha=0.1,linestyle='--')
        plt.ylim(top=1)
        plt.ylim(bottom=0)
        plt.plot(x_in_seconds,ave_act_overep_torso[:,g],linestyle = '-',label="Without exosuit",color='crimson', linewidth=2)
        plt.fill_between(x_in_seconds,ave_act_overep_torso[:,g]-std_act_overep_torso[:,g],ave_act_overep_torso[:,g]+std_act_overep_torso[:,g],color='pink',alpha=0.1)

        plt.title(f'{actuator_group}',fontsize=24)
        plt.xlabel("Timestep (s)")
        plt.ylabel("Activation")
        plt.legend()
        plt.savefig(path+'/data' +'/' + str(angle) + '_' + str(g)+'activation.png')
        plt.close()

        #ave_ctrl_overep_exo[:,g].tofile(path+'/data' +'/' + 'exo' + str(angle) + '_' + str(g) + 'ctrl.csv', sep=",")
        g=g+1
    '''
    #ICORR FIGURES
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(5,4))
    fig,axs = plt.subplots(2)
    x = np.arange(200)
    x_in_seconds = x*0.025
    
    axs[0].set_title('ILpT Left',rotation='vertical',x=-0.1,y=0.3)
    axs[0].set_ylim(top=1)
    axs[1].set_ylim(bottom=0)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Activation")
    axs[0].xaxis.set_visible(False)
    axs[0].plot(x_in_seconds,ave_act_overep_exo[:,7],linestyle = '--',label="With exosuit",color='coral',linewidth=5)
    axs[0].fill_between(x_in_seconds,ave_act_overep_exo[:,7]-std_act_overep_exo[:,7],ave_act_overep_exo[:,7]+std_act_overep_exo[:,7],color='coral',alpha=0.3)
    axs[0].plot(x_in_seconds,ave_act_overep_torso[:,7],linestyle = '-',label="Without exosuit", color ='lightskyblue',linewidth=5)
    axs[0].fill_between(x_in_seconds,ave_act_overep_torso[:,7]-std_act_overep_torso[:,7],ave_act_overep_torso[:,7]+std_act_overep_torso[:,7],color='lightskyblue',alpha=0.3)


    axs[1].set_title('LTpL Left',rotation='vertical',x=-0.1,y=0.3)
    axs[1].set_ylim(top=1)
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlabel("Time(s)")
    axs[1].set_ylabel("Activation")
    line1,=axs[1].plot(x_in_seconds,ave_act_overep_exo[:,11],linestyle = '--',label="With exosuit",color='coral',linewidth=5)
    axs[1].fill_between(x_in_seconds,ave_act_overep_exo[:,11]-std_act_overep_exo[:,11],ave_act_overep_exo[:,11]+std_act_overep_exo[:,11],color='coral',alpha=0.3)
    line2,=axs[1].plot(x_in_seconds,ave_act_overep_torso[:,11],linestyle = '-',label="Without exosuit",color='lightskyblue',linewidth=5)
    axs[1].fill_between(x_in_seconds,ave_act_overep_torso[:,11]-std_act_overep_torso[:,11],ave_act_overep_torso[:,11]+std_act_overep_torso[:,11],color='lightskyblue',alpha=0.3)

    fig.legend(handles=[line1, line2], labels=["With exosuit", "Without exosuit"], loc='lower center', ncol=2)
    plt.tight_layout()
    plt.show()
    '''
    #save
    #os.makedirs(path+'/data' +'/' + env_name + model_num, exist_ok=True)
    #ave_ctrl_overep.tofile(path+'/data' +'/' + env_name + model_num +'ctrl.csv', sep=",")
def corr(ave_act_overep_torso,ave_act_overep_exo):
    m =0
    r_w_wo_exo=[0]*24
    p_w_wo_exo =[0]*24

    while (m<24):
        r_w_wo_exo[m],p_w_wo_exo[m] = scipy.stats.pearsonr(ave_act_overep_torso[:][m],ave_act_overep_exo[:][m])
        m=m+1

    r_ave = np.mean(r_w_wo_exo)
    r_std = np.std(r_w_wo_exo)
    p_ave = np.mean(p_w_wo_exo)
    p_std = np.std(p_w_wo_exo)


    r_homologous_exo = [0]*12
    r_homologous_torso = [0]*12
    p_homologous_exo =[0]*12
    p_homologous_torso = [0]*12
    m = 0
    counter = 0
    while (m<20):
        r_homologous_torso[counter],p_homologous_torso[counter] = scipy.stats.pearsonr(ave_act_overep_torso[:][m],ave_act_overep_torso[:][m+1])
        r_homologous_exo[counter],p_homologous_exo[counter] = scipy.stats.pearsonr(ave_act_overep_exo[:][m],ave_act_overep_exo[:][m+1])
        m = m+2
        counter = counter +1

    #EO
    r_homologous_torso[10],p_homologous_torso[10] = scipy.stats.pearsonr(ave_act_overep_torso[:][20],ave_act_overep_torso[:][22])
    r_homologous_exo[10],p_homologous_exo[10] = scipy.stats.pearsonr(ave_act_overep_exo[:][20],ave_act_overep_exo[:][22])

    #IO
    r_homologous_torso[11],p_homologous_torso[11] = scipy.stats.pearsonr(ave_act_overep_torso[:][21],ave_act_overep_torso[:][23])
    r_homologous_exo[11],p_homologous_exo[11] = scipy.stats.pearsonr(ave_act_overep_exo[:][21],ave_act_overep_exo[:][23])

    data={
        'w/wo exo r':r_w_wo_exo,
        'w/wo exo p':p_w_wo_exo,
        'homo exo r':r_homologous_exo,
        'homo exo p':p_homologous_exo,
        'homo torso r': r_homologous_torso,
        'homo torso p':p_homologous_torso
    }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

    filename=f"{angle}corr.csv"
    df.to_csv(filename,index=False)
  
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

ave_act_exo, std_act_exo,ave_act_overep_exo,std_act_overep_exo,ctrl_RL_exo = calculate(all_activation_exo,all_ctrl_exo)
ave_act_torso, std_act_torso,ave_act_overep_torso,std_act_overep_torso,ctrl_RL_torso = calculate(all_activation_torso,all_ctrl_torso)

print(f"Average Activation with Exosuit {ave_act_exo} std {std_act_exo}")
print(f"Average Activation without Exosuit {ave_act_torso} std {std_act_torso}")
plot_activation(ave_act_overep_exo,std_act_overep_exo, ave_act_overep_torso,std_act_overep_torso)
corr(ave_act_overep_torso,ave_act_overep_exo)
#fourty_emg,sixty_emg,eighty_emg = emg_data()
#plot_emg(fourty_emg,sixty_emg,eighty_emg, ctrl_RL_exo )
