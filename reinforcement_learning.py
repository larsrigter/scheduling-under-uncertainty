import pandas as pd
from collections import Counter
from datetime import timedelta,time,datetime
import numpy as np
import copy
from sklearn.utils import shuffle
import itertools
import copy
import datetime as DateTime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import timedelta,time,datetime
from sklearn.utils import shuffle
import numpy as np
import pickle
from environment import *
from utilities import *
class ReinforcementLearning:
    def __init__(self,gamma=0.95,decay=30000,learnRate=0.0001,
                 capacity=30000,tau=4000,setup='setup1',policyBatchSize=5,
                 saveNorm=2016,qUpdateFreq=2,updatePol=2000,saveFiles=None):
        self.setup=setup
        self.memory=self.ReplayMemory(capacity)
        self.gamma=gamma
        self.decay=decay
        self.tau=tau
        self.actions=[0,1]
        self.policyBatchSize=5
        self.saveFiles=saveFiles
        
        self.policyBatchSize=policyBatchSize
        self.saveNorm=saveNorm
        self.qUpdateFreq=qUpdateFreq
        self.updatePol=updatePol
        
        if self.setup=='setup1':
            self.inputDim=65
        elif self.setup=='setup2':
            self.inputDim=60
        elif self.setup=='setup3':
            self.inputDim=55
            
        self.qModel=self.QNetwork(self.inputDim)
        self.policy=self.PolicyNetwork(self.inputDim)
        self.target_qModel=copy.deepcopy(self.qModel)
        self.optimizer=torch.optim.Adam(self.qModel.parameters(), learnRate)
        self.polOptimizer=torch.optim.Adam(self.policy.parameters(), learnRate)
        
        self.states=None
        self.oldStates=None
        self.reward=None
        self.logProbs=[]
        self.rewards=[]
        self.qVals=[]
        self.losses=[]
        
        #batch 3 update
        if self.setup=='setup1':
            self.normalizer=Normalizer(65)
        elif setup=='setup2':
            self.normalizer=Normalizer(60) 
        elif setup=='setup3':
            self.normalizer=Normalizer(55)

    class QNetwork(nn.Module):
        def __init__(self, inputDim):
            nn.Module.__init__(self)
            self.layers=nn.Sequential(
                nn.Linear(inputDim, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1))
        def forward(self, x):
            out=self.layers(x)
            return out
        
    class PolicyNetwork(nn.Module):
        def __init__(self, inputDim):
                nn.Module.__init__(self)
                self.layers=nn.Sequential(
                    nn.Linear(inputDim, 100),
                    nn.ReLU(),
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 1),
                    )
        def forward(self, x):
            out=self.layers(x)
            return out.flatten(0)

    class ReplayMemory:
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
        def push(self, transition):
            if len(self.memory) == self.capacity:
                del self.memory[0]
            self.memory.append(transition)
        def sample(self, batchSize):
            return random.sample(self.memory, k=batchSize)
        
        def __len__(self):
            return len(self.memory)


    def compute_target(self,reward, nextState):
        self.target_qModel.eval()
        maxQs=[]
        with torch.no_grad():
            maxQ=self.target_qModel(nextState[0]).flatten(0).max()
        target = reward[0] + torch.tensor(self.gamma) * maxQ
        return target
    
    
    def train_dqn(self):        
        if len(self.memory)<1:
            return None
        
        #sample from experience buffer
        transitions=self.memory.sample(1)
        state, reward, nextState = zip(*transitions)
        state=torch.stack(state[0])
        target = self.compute_target(reward, nextState)
        target = target.repeat(state.shape[0])
    
        #train model
        self.qModel.train()
        qVal = self.qModel.forward(state).flatten(0)
        loss = torch.nn.MSELoss()(qVal, target)

        #gradient step with parameter cap
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qModel.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def load_target_qModel(self):
        self.target_qModel.load_state_dict(torch.load(self.saveFiles['critic']))
        
    def save_actor(self,e,reward=None):
        if type(reward)!=type(None):
            torch.save(self.policy.state_dict(), self.saveFiles['actor']+"_"+str(e)+"_"+reward.replace(':','_')+'.mod')
        else:
            torch.save(self.policy.state_dict(), self.saveFiles['actor'])
        
    def save_critic(self,e=None,reward=None):
        if type(reward)!=type(None):
            torch.save(self.qModel.state_dict(), self.saveFiles['critic']+"_"+str(e)+"_"+reward.replace(':','_')+'.mod')
        else:
            torch.save(self.qModel.state_dict(), self.saveFiles['critic'])

    def load_actor(self):
        path=self.saveFiles['actor']
        self.policy.load_state_dict(torch.load(path))
        
    def load_critic(self):
        path=self.saveFiles['critic']
        self.qModel.load_state_dict(torch.load(path))
        
    def get_epsilon(self,it):
        if it == 0:
            return 1
        elif it <= self.decay:
            return 1 - (it * (0.95/self.decay))
        else:
            return 0.05

    def plan_jobs_ac(self,env,pendingJobs,time,intervalLength,ite,oldStates,status,reward):
        totCharged=0
        storeJobs=[]
        newStates=None
        logProbs=None
        pickedQVal=None
            
        if len(pendingJobs)>0:
                                  
            #convert state,actions to features
            self.states=get_features_rl(env,pendingJobs,time,self.normalizer,ite,status,setup=self.setup)
            
            #predict q values
            self.qModel.eval()
            with torch.no_grad():
                qValues=self.qModel(copy.deepcopy(self.states)).flatten(0)

            #add state, action reward tuple to memory
            if type(self.oldStates)!=type(None):
                self.memory.push((self.oldStates, self.reward,self.states))

            charged=0
            newStates=[]
            logProbs=[]
            pickedQVal=[]
            newJobs=[]

            #approximate state, action representation
            output=self.policy(self.states).flatten()
                                  
            # sample jobs while there are jobs in the queue and capacity is available
            while True and len(pendingJobs)>0:
                probs=nn.Softmax(dim=0)(output)
                dist = torch.distributions.Categorical(probs=probs)
                if status=='train':
                    action = dist.sample()
                else:
                    action=torch.argmax(probs)
                if charged<=env.demand[time]:
                    #handle administration
                    job=pendingJobs[action]
                    newJobs.append(job)
                    del pendingJobs[action]
                    newStates.append(self.states[action])
                    pickedQVal.append(qValues[action])
                    output=torch.cat([output[:action], output[action+1:]])
                    states=torch.cat([self.states[:action], self.states[action+1:]])
                    qValues=torch.cat([qValues[:action], qValues[action+1:]])
                    logProb=dist.log_prob(action)
                    logProbs.append(logProb)
                    charged+=job['MaxPower(KW)']
                    
                else:
                    break
            #add non allocated jobs to the queue
            for job in pendingJobs:
                newJobs.append(job)
            pendingJobs=newJobs
                                  
            #calculate average q value of selected jobs
            pickedQVal=torch.mean(torch.stack(pickedQVal))
                                  
            #caclulate reward
            self.reward=self.reward_function(pendingJobs,time,intervalLength,env.demand[time])
            if len(newStates)==0:
                newStates=None
        else:
            reward=None

        #save normalizer at 7th episode
        if ite==(self.saveNorm):
            self.memory.memory = []
            pickle.dump(self.normalizer, open(self.saveFiles['normalizer']+'_AC', "wb" ) )
        self.oldStates=newStates
        
        return pendingJobs,logProbs,pickedQVal
                                  
    def plan_jobs_dqn(self,env,pendingJobs,time,intervalLength,ite,status):
        newStates=None
        if len(pendingJobs)>1:
            #predict Q-values of jobs
            newJobs=[]
            self.qModel.eval()
            self.states=get_features_rl(env,pendingJobs,time,self.normalizer,ite,status,setup=self.setup)
            with torch.no_grad():
                qValues=self.qModel(self.states).flatten(0)
            for i,(job,qVal) in enumerate(zip(pendingJobs,qValues)):
                job['target']=qVal.item()
                job['allocatedPos']=i
                newJobs.append(job)
                
            #sort jobs on Q-value
            pendingJobs=sorted(newJobs, key=lambda tup: tup['target'],reverse=True)

            #push states to memory
            if type(self.oldStates)!=type(None):
                self.memory.push((self.oldStates, self.reward,self.states))
#                 print(self.oldStates,self.reward,self.states)
            
            #with epsilon probability randomly shuffle jobs
            epsilon=self.get_epsilon(ite)
            rand=np.random.uniform(low=0.0, high=1.0, size=None)
            if rand<=epsilon and status!='test':
                random.shuffle(pendingJobs)
            
            #assign first n jobs to be executed and collect states
            assignedJobs=[]
            charged=0
            newStates=[]
            for i in range(len(pendingJobs)):
                job=pendingJobs.pop(0)
                assignedJobs.append(job)
                charged+=job['MaxPower(KW)']
                if charged>env.demand[time]:
                    break
                newStates.append(self.states[job['allocatedPos'],:])
            
            #keep administration right
            for job in pendingJobs:
                assignedJobs.append(job)
            pendingJobs=assignedJobs
            
            #calculate rewards
            self.reward=self.reward_function(pendingJobs,time,intervalLength,env.demand[time])
            
            #if no jobs can be executed set to none
            if len(newStates)==0:
                newStates=None
        
        # save normalizer after 7 episodes
        if ite==(self.saveNorm):
            self.memory.memory = []
            pickle.dump(self.normalizer, open(self.saveFiles['normalizer']+'_DQN', "wb" ) )
        self.oldStates=newStates
        return pendingJobs
                                  
    def loss_function(self,logProbs,rewards,qVals):                  
        #calculate gains
        gains=[]
        for i in range(len(logProbs)-1):
            gains.append(rewards[i]+qVals[i+1])
            
        logProbs=logProbs[0:len(logProbs)-1]                      
        losses=torch.tensor(0,dtype=torch.float)
        for (dayProb,gain) in zip(logProbs,gains):
            dayLos=-sum(dayProb)
            dayLos*=gain
            losses+=dayLos
        return losses
                                  
    def train_policy(self,logProbs,rewards,qVals):
        loss=self.loss_function(logProbs,rewards,qVals)
        self.polOptimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.polOptimizer.step()
        loss=loss.item()
        return loss
    
    def dqn_policy(self,env,pendingJobs,time,intervalLength,ite,status):
        if status=='train': 
            #fix target network every tau steops
            if (ite+1)%self.tau==0:
                self.load_target_qModel()
            if (ite)%int((self.tau))==0:
                self.save_critic()
            #Update dqn two times per iteration
            for i in range(self.qUpdateFreq):
                loss=self.train_dqn()
                if type(loss)!=type(None):
                    self.losses.append(loss)
        #load network for testing
        if status=='test':
            self.load_critic()

        #plan jobs
        pendingJobs=self.plan_jobs_dqn(env,pendingJobs,time,intervalLength,ite,status)

        return pendingJobs
    
    def actor_critic_policy(self,pendingJobs,time,intervalLength,ite,e,episodeStep,
                        status):
        if status=='train':
            #load target model
            if (ite+1)%self.tau==0:
                print("update target model")
                self.load_target_qModel()

            if (ite)%int((self.tau))==0:
                self.save_critic(e)

            #train critic
            for i in range(self.qUpdateFreq):
                loss=self.train_dqn()
                if type(loss)!=type(None):
                    self.losses.append(loss)

            #train policy
            #########change to 2000
            if ite>self.updatePol and len([i for i in self.logProbs if type(i)!=type(None)])==self.policyBatchSize:
                self.rewards[-1]=0
                loss=self.train_policy(self.logProbs,self.rewards,self.qVals)
                self.logProbs=[]
                self.rewards=[]
                self.qVals=[]
            self.save_actor(e)

        #plan jobs
        if status=='test'and ite==0:
            self.load_actor()
        pendingJobs,logProb,qVal=self.plan_jobs_ac(env,pendingJobs,time,intervalLength,ite,self.states,status,self.reward)

        #do administration
        if len(self.qVals)==self.policyBatchSize:
            self.qVals=self.qVals[1:]
            self.logProbs=self.logProbs[1:]
            self.rewards=self.rewards[1:]
            
        #make sure rewards are asynchronic
        if episodeStep==1 and torch.rand(1)[0]>0.5:
            self.logProbs=[]
            self.rewards=[]
            self.qVals=[]
        self.qVals.append(qVal)
        self.logProbs.append(logProb)
        self.rewards.append(self.reward)

        return pendingJobs
    def reward_function(self,pendingJobs,time,intervalLength,capacity):
        chargedPow=0
        chargedVol=0
        for job in pendingJobs:
            chargedPow+=job['MaxPower(KW)']
            if chargedPow>capacity:
                surplus=(job['duetime_no']-time)*np.round((intervalLength/60)*job['MaxPower(KW)']+0.0005,3)-job['TotalEnergy(KWh)']
                if surplus<0:
                    chargedVol-=(np.round((intervalLength/60)*job['MaxPower(KW)']+0.0005,3))
        return chargedVol
def get_features_rl(env,pendingJobs,time,normalizer,ite,status,setup='setup1'):
    arrivedJobs=copy.deepcopy(pendingJobs)    
    
    #calculate fields
    if setup!='setup3':
        minDue=min([i['duetime_no'] for i in arrivedJobs])
        maxDue=max([i['duetime_no'] for i in arrivedJobs])
        avgDue=np.mean([i['duetime_no'] for i in arrivedJobs])

    minArrive=min([i['arrivaltime_no'] for i in arrivedJobs])
    maxArrive=max([i['arrivaltime_no'] for i in arrivedJobs])
    avgArrive=np.mean([i['arrivaltime_no'] for i in arrivedJobs])

    minChargeVol=min([i['chargedVolume'] for i in arrivedJobs])
    maxChargeVol=max([i['chargedVolume'] for i in arrivedJobs])
    avgChargeVol=np.mean([i['chargedVolume'] for i in arrivedJobs])
    sumCharge=sum([i['chargedVolume'] for i in arrivedJobs])

    minPower=min([i['MaxPower(KW)'] for i in arrivedJobs])
    maxPower=max([i['MaxPower(KW)'] for i in arrivedJobs])
    avgPower=np.mean([i['MaxPower(KW)'] for i in arrivedJobs])
    sumPower=sum([i['MaxPower(KW)'] for i in arrivedJobs])
     
    if setup=='setup1':
        minVol=min([i['TotalEnergy(KWh)'] for i in arrivedJobs])
        maxVol=max([i['TotalEnergy(KWh)'] for i in arrivedJobs])
        meanVol=np.mean([i['TotalEnergy(KWh)'] for i in arrivedJobs])
        sumVol=sum([i['TotalEnergy(KWh)'] for i in arrivedJobs])
        
    #select features
    arrivedJobs=pd.DataFrame(arrivedJobs)
    if setup=='setup1':
        arrivedJobs=arrivedJobs[['MaxPower(KW)', 'arrivaltime_no', 'duetime_no','chargedVolume','TotalEnergy(KWh)']]
    elif setup=='setup2':
        arrivedJobs=arrivedJobs[['MaxPower(KW)', 'arrivaltime_no', 'duetime_no','chargedVolume']]
    elif setup=='setup3':
        arrivedJobs=arrivedJobs[['MaxPower(KW)', 'arrivaltime_no','chargedVolume']]
    
    #add demand
    for i in range(20):
        if time+i<=287:
            arrivedJobs['demand '+str(i)]=env.demand[time+i]#/env.orDemand[time+i]
        else:
            arrivedJobs['demand '+str(i)]=0
        if time-i-1>0:
            arrivedJobs['demand '+str(-i)]=env.demand[time-i-1]#/env.orDemand[time-i-1]
        else:
            arrivedJobs['demand '+str(-i)]=0
            
    #add global features  
    arrivedJobs['njobs']=len(arrivedJobs)
    arrivedJobs['time']=time  
    
    if setup!='setup3':
        arrivedJobs['maxToCharge']=(arrivedJobs['duetime_no']-time)*arrivedJobs['MaxPower(KW)']
        arrivedJobs['minDue']=minDue
        arrivedJobs['maxDue']=maxDue
        arrivedJobs['avgDue']=avgDue
        
    arrivedJobs['minArrive']=minArrive
    arrivedJobs['maxArrive']=maxArrive
    arrivedJobs['avgArrive']=avgArrive
    
    arrivedJobs['minChargeVol']=minChargeVol
    arrivedJobs['maxChargeVol']=maxChargeVol
    arrivedJobs['avgChargeVol']=avgChargeVol
    arrivedJobs['sumChargeVol']=sumCharge

    arrivedJobs['minPower']=minPower
    arrivedJobs['maxPower']=maxPower
    arrivedJobs['avgPower']=avgPower
    arrivedJobs['sumPower']=sumPower

    if setup=='setup1':
        arrivedJobs['minVol']=minVol
        arrivedJobs['maxVol']=maxVol
        arrivedJobs['meanVol']=meanVol
        arrivedJobs['sumVol']=sumVol
        
    #normalize features  
    arrivedJobs=torch.Tensor(arrivedJobs.values)
    if ite<=7*287 and status!='test':
        for job in range(arrivedJobs.shape[0]):
            normalizer.observe(arrivedJobs[job,:])
    arrivedJobs = normalizer.normalize(arrivedJobs)
    
    return arrivedJobs
