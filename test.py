import os
import numpy as np
import pandas as pd 
for subj in os.listdir("./data/processed_emg"):
        subjname = os.fsdecode(subj)
        print(subjname)
        #Extract Data
        longest = 0 
        subj_no = 0
        long_name = ""
        for angle in os.listdir("./data/processed_emg/"+subjname):
            file = os.fsdecode(angle)
            df = pd.read_csv("./data/processed_emg/"+subjname + "/" + file)
            local_max = max(len(df.iloc[:,0]),len(df.iloc[:,1]),len(df.iloc[:,2]),len(df.iloc[:,3]),len(df.iloc[:,4]))
            print(local_max)
            if local_max>longest:
                longest = local_max
                long_name = f"{subjname}{file}"
            
        subj_no = subj_no +1
print(longest)
print(long_name)