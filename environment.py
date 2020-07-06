import random
import copy
import numpy as np
import pandas as pd
from utilities import *

class Environment:
    def __init__(self,df,demand=None,prices=None,intervalLength=5,intraPrices=None,trainDf=None):
        self.trainDf=trainDf
        self.demandDf=demand
        self.priceDf=prices
        self.intraDf=intraPrices
        self.intervalLength=intervalLength
        self.currentEpisode=min(df['days_for_aggregation'])
        self.dataframe=df
        self.episode=copy.deepcopy(self.dataframe[self.dataframe['days_for_aggregation']==self.currentEpisode])
        self.episode=self.episode.sort_values(by='start_time')
        self.weekday=self.episode['weekday'].unique()[0]
        
        self.demand,self.timeIntervals=copy.deepcopy(self.demandDf[self.weekday])
        
        self.demand=np.round(self.demand+0.0005,3)
        self.timeIntervals.append(self.timeIntervals[-1].replace(hour=23,minute=59,second=59))
        self.demandLeft=copy.deepcopy(self.demand)
        self.lowerInterval=0
        self.upperInterval=1
        self.jobNumber=0
        self.transactionCosts=0.09
        
    def step(self):
        if self.upperInterval==len(self.timeIntervals):
            return [], True, self.timeIntervals[self.lowerInterval]
        leftBound=self.lowerInterval
        rightBound=self.upperInterval
        self.lowerInterval+=1
        self.upperInterval+=1
        jobs=copy.deepcopy(self.episode[(self.episode['arrivaltime_no']>=leftBound)
                 &(self.episode['arrivaltime_no']<rightBound)])
        if len(jobs)>0:
            jobs=jobs[['arrival_time','due_time','time_till_due','MaxPower(KW)','TotalEnergy(KWh)',
                       'StartCard', 'ChargePoint','weekday', 'start_hour','arrivaltime_no',
                       'duetime_no','days_for_aggregation']]
            jobs['chargedVolume']=0
            jobs['estimatedVolume']=0
            jobs['allocatedPos']=0
            jobs['target']=0
            jobs['orig_energ(KWh)']=jobs['TotalEnergy(KWh)']
            jobList=list(jobs.to_dict(orient='index').items())
            jobList=[i[1] for i in jobList]
            return jobList, False, leftBound
        else:
            return [] , False, leftBound

    def reset(self):
        if self.currentEpisode==max(self.dataframe['days_for_aggregation']):
            self.currentEpisode=min(self.dataframe['days_for_aggregation'])
        else:
            self.currentEpisode+=1
        self.episode=copy.deepcopy(self.dataframe[self.dataframe['days_for_aggregation']==self.currentEpisode])
        self.episode=self.episode.sort_values(by='start_time')
        self.weekday=self.episode['weekday'].unique()[0]
        self.demand,self.timeIntervals=copy.deepcopy(self.demandDf[self.weekday])
        self.timeIntervals.append(self.timeIntervals[-1].replace(hour=23,minute=59,second=59))
        self.demandLeft=copy.deepcopy(self.demand)
        self.lowerInterval=0
        self.upperInterval=1
        self.jobNumber=0
        
    def calculate_costs(self):
        totPrice=0
        for i in range(24):
            totDemand=totalDemand(self.demand[(i*12):(i+1)*12],self.intervalLength)
            price=self.priceDf[self.priceDf['days_for_aggregation']==self.currentEpisode-1]['price(EUR/KWh)'].values[i]
            totPrice+=(totDemand*price)
        totDeliverBackPrice=0
        noIntraCosts=0
        for i in range(24):
            totDemand=totalDemand(self.demandLeft[(i*12):(i+1)*12],self.intervalLength)
            price=self.priceDf[self.priceDf['days_for_aggregation']==self.currentEpisode-1]['price(EUR/KWh)'].values[i]
            priceIntra=self.intraDf[self.intraDf['days_for_aggregation']==self.currentEpisode]['price(EUR/KWh)'].values[i]
            totDeliverBackPrice+=(totDemand*priceIntra) -(self.transactionCosts if totDemand > 1 else 0)
            noIntraCosts+=(totDemand*price)
        return np.round(totPrice-totDeliverBackPrice,3), np.round(totPrice-noIntraCosts,3)
    
    def get_feasible_jobs(self,jobList,time):
        feasibleList=[]
        unfeasibleNo=0
        leftoverEnergy=0
        currentTime=time
        for job in jobList:
            jobTime=job['duetime_no']
            if jobTime>currentTime and job['TotalEnergy(KWh)']>=np.round((self.intervalLength/60)*job['MaxPower(KW)']+0.0005,3):
                feasibleList.append(job)
            elif jobTime<=currentTime and job['TotalEnergy(KWh)']>0:
                unfeasibleNo+=1
                leftoverEnergy+=job['TotalEnergy(KWh)']
        return feasibleList,unfeasibleNo,leftoverEnergy
    
    def check_available_capacity(self,time,job):
        if self.demandLeft[time]>=job['MaxPower(KW)']:
            return True
        else:
            return False
        print("no matching time found",time)

    def plan_job(self,job,time):
        chargedCap=0
        prevVol=job['TotalEnergy(KWh)']
        if prevVol<0:
            print(job)
        job['TotalEnergy(KWh)']-=np.round((self.intervalLength/60)*job['MaxPower(KW)']+0.0005,3)

        if job['TotalEnergy(KWh)']<0:
            chargedCap=0
            self.demandLeft[time]-=job['MaxPower(KW)']
        else:
            chargedCap=np.round((self.intervalLength/60)*job['MaxPower(KW)']+0.0005,3)
            job['chargedVolume']+=np.round((self.intervalLength/60)*job['MaxPower(KW)']+0.0005,3)
            self.demandLeft[time]-=job['MaxPower(KW)']
        return job,chargedCap
    def process_queue(self,time,pendingJobs):
        itCharged=0
        storeJobs=[]
        availableCapacity=True
        if len(pendingJobs)>0:
            availableCapacity=self.check_available_capacity(time,pendingJobs[0])
            while len(pendingJobs)>0 and availableCapacity:
                job,charged=self.plan_job(pendingJobs[0],time)
                itCharged+=charged
                pendingJobs=pendingJobs[1:]
                storeJobs.append(job)
                if len(pendingJobs)>0:
                    availableCapacity=self.check_available_capacity(time,pendingJobs[0])
            pendingJobs=storeJobs+pendingJobs
        return pendingJobs,itCharged