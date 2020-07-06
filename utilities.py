import numpy as np
from datetime import timedelta,time,datetime
from pulp import *
import copy
import torch
#convert power to electricity volume
def totalDemand(y,intervalLength):
    c=0
    for i in list(y):
        c+=(i*(intervalLength/60))
    return c
#get time interval of  intervallength for onef day
def get_time_intervals(intervalLength):
    intervals=np.array(range(0,(24*12)))
    intervals*=intervalLength
    timeIntervals=[timedelta(minutes=int(i)) for i in intervals]
    timeIntervals=[(datetime.min + i).time() for i in timeIntervals]
    timeIntervals.append(timeIntervals[-1].replace(hour=23,minute=59,second=59))
    timeIntervals=sorted(timeIntervals)
    return timeIntervals

#convert time point to interval
def convertTimeToInterval(time,timeIntervals):
    i=0
    for left,right in zip(timeIntervals[0:-1],timeIntervals[1:]):
        try:
            if time.time()<right and time.time()>=left:
                return int(i)
        except:
            if time<right and time>=left:
                return int(i)
        i+=1
    return None
#calculate demand of dataframe
def calculate_demand(subDf,intervalLength,timeIntervals,chargeType='asap'):
    assert 'days_for_aggregation' in subDf.keys(), "columm days_for_aggregation is not present"
    assert 'Started' in subDf.keys(), "columm Started is not present"
    assert 'EndCharge' in subDf.keys(), "columm EndCharge is not present"
    assert 'MaxPower' in subDf.keys(), "MaxPower"
    assert 60%intervalLength==0, "hour not divisible with intervalLength"
    if chargeType=='asap':
        columnForStart='Started'
        columnForEnd='EndCharge'
        columnForPower='MaxPower'
    if chargeType=='latest':
        columnForStart='latest_start_time'
        columnForEnd='Ended'
        columnForPower='MaxPower'
    if chargeType=='average':
        columnForStart='Started'
        columnForEnd='Ended'
        columnForPower='AveragePowerConnTime'
        
        
    
    y=np.zeros(len(timeIntervals)-1)
    
    i=0
    for lowerbound,upperbound in zip(timeIntervals[0:-1],timeIntervals[1:]):
        for row in subDf[[columnForStart,columnForEnd,columnForPower]].values:
            startTime=row[0]
            endTime=row[1]
            power=row[2]
            if endTime.time()>=lowerbound and startTime.time()<upperbound:
                y[i]+=(power/1000)

        i+=1
    return y

#calculate price optimal demand
def price_opt_sched(df,intervalLength,timeIntervals,priceDf=None,day=0,fracGap=0.01):
    schedule=iLPOptSchedule(df,intervalLength,priceDf=priceDf,day=day,fracGap=0.01)
    power=np.array(df["MaxPower(KW)"].values)
    power=power.reshape(1,power.shape[0])
    volumes=np.matmul(power,schedule)
    return volumes.flatten()

#estimate demand
def estimate_demand(df,intervalLength,chargeType='asap',priceDf=None,surplus=1,fracGap=0.01):
    average_demand={}
    intervals=get_time_intervals(intervalLength)
    for i in range(7):
        aggDays=list(df[df['weekday']==i]['days_for_aggregation'].unique())
        sumUsage=None
        for day in aggDays:
            subDf=df[df['days_for_aggregation']==day]
            if chargeType=='price_opt':
                usage=price_opt_sched(subDf,intervalLength,intervals,priceDf=priceDf,day=day,fracGap=0.01)
            else:
                usage=calculate_demand(subDf,intervalLength,intervals,chargeType=chargeType)
            if type(sumUsage)==type(None):
                sumUsage=usage
            else:
                sumUsage=np.sum((usage,sumUsage),axis=0)
        try:
            sumUsage*=surplus
            average_demand[i]=(sumUsage/len(aggDays),intervals[0:-1])
        except:
            continue
    return average_demand
	
#optimize schedule for price or maximum allocation
#optimize schedule for price or maximum allocation
def iLPOptSchedule(df,intervalLength,demand=None,
    day=0,priceOpt=True,allocOpt=False,priceDf=None,maxTime=50,fracGap=0.05):
    testDf=copy.deepcopy(df)
    jobs  = range(len(testDf))
    time = range(24*int(60/intervalLength))
    timeNo=0

    #set variables
    var = pulp.LpVariable.dicts("if_i_charged_at_j", ((i, j) for i in jobs for j in time),cat='Binary')
    
    #objective
    powers=testDf['MaxPower(KW)'].values
    if priceOpt:
        model = pulp.LpProblem("scheduling_model", pulp.LpMinimize)
        costList=[]
        for price in list(priceDf[priceDf['days_for_aggregation']==day-1]['price(EUR/KWh)'].values):
            for i in range(int(60/intervalLength)):
                costList.append(price) 
        model+=lpSum(var[item]*powers[item[0]]*(1/(60/intervalLength))*costList[item[1]] for item in var.keys())
    
    
    if allocOpt:
        model = pulp.LpProblem("scheduling_model", pulp.LpMaximize)
        model+=lpSum(var[item]*powers[item[0]] for item in var.keys())
        for t in time:
            model+=lpSum(var[item]*powers[item[0]] for item in [i for i in var.keys() if i[1] == t]) <= demand[t]

    #make sure everything is allocated between arrival time and due time
    jobNo=0
    for arrivTime,dueTime  in zip(testDf['arrivaltime_no'],testDf['duetime_no']):
        times=np.setdiff1d(time,range(arrivTime,dueTime))
        for t in times:
            if t-timeNo>=0:
                model += var[(jobNo,t)]==0
        jobNo+=1
        
    #restrictions total volume
    jobNo=0
    for energy,maxPower in zip(testDf['TotalEnergy(KWh)'].values,testDf['MaxPower(KW)'].values):
        #include correction of one time interval
        correction=np.round((intervalLength/60)*maxPower,3)
        if energy-correction<0:
            energy=0
            correction=0
        
        #for price opt make sure everything is charged
        if priceOpt:
            model += lpSum([var[item]*np.round((intervalLength/60)*maxPower+0.0005,3) for item in [i for i in var.keys() if i[0]==jobNo]]) >= energy-correction
        
        #for maximum allocation make sure not more is allocated than can be charged
        if allocOpt:
            model += lpSum([var[item]*np.round((intervalLength/60)*maxPower+0.0005,3) for item in [i for i in var.keys() if i[0]==jobNo]]) <= energy
        jobNo+=1
    
    model.solve(PULP_CBC_CMD(fracGap = fracGap, maxSeconds = maxTime, threads = None))
    print("Status:", LpStatus[model.status])
    
    #construct schedule
    schedule=np.zeros((len(testDf),24*int(60/intervalLength)))
    rows=[]
    listIsCharged=[b for b in model.variables() if b.varValue>0]
    for i in listIsCharged:
        split=str(i).split('(')[1]
        row=split.split(',')[0]
        rows.append(int(row))
        column=split.split(',')[1][1:-1]
        schedule[int(row)][int(column)]=1

    return schedule
	
class Normalizer():
	def __init__(self, num_inputs):
		self.n = torch.zeros(num_inputs)
		self.mean = torch.zeros(num_inputs)
		self.mean_diff = torch.zeros(num_inputs)
		self.var = torch.zeros(num_inputs)

	def observe(self, x):
		self.n += 1.
		last_mean = self.mean.clone()
		self.mean += (x-self.mean)/self.n
		self.mean_diff += (x-last_mean)*(x-self.mean)
		self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

	def normalize(self, inputs):
		obs_std = torch.sqrt(self.var)
		return (inputs - self.mean)/obs_std

