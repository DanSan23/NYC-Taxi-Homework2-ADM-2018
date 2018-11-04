
# coding: utf-8

# # NYC Taxi homework2


#Data Cleaning phase:

# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
%matplotlib inline

# In[2]:
error = pd.DataFrame()
for i in range(1,7):
    d=pd.read_csv("C:/Users/Atefeh/Desktop/ADM-2/yellow_tripdata_2018-0"+str(i)+".csv")
    d['yearS'] = pd.to_datetime(d["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.year
    d['yearE'] = pd.to_datetime(d["tpep_dropoff_datetime"],format='%Y-%m-%d %H:%M:%S').dt.year
    d['hourS'] = pd.to_datetime(d["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.hour
    d['weekdayS'] = pd.to_datetime(d["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.weekday
    d['monthS'] = pd.to_datetime(d["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.month
    d['hourE'] = pd.to_datetime(d["tpep_dropoff_datetime"],format='%Y-%m-%d %H:%M:%S').dt.hour
    d['weekdayE'] = pd.to_datetime(d["tpep_dropoff_datetime"],format='%Y-%m-%d %H:%M:%S').dt.weekday
    d['monthE'] = pd.to_datetime(d["tpep_dropoff_datetime"],format='%Y-%m-%d %H:%M:%S').dt.month    
    d=d.loc[(d.yearS==2018)&(d.yearE==2018)&(d.monthS<7)&(d.monthE<7)&(d.trip_distance>0.2)]
    error = error.append(d[d['monthS']!=i])
    d.reset_index(drop=True)
    d.to_csv("Filtered"+str(i)+".csv")
    del d
    
# In[3]:   
   for i in range(1,7):
    string_new = "Filtered"+str(i)+".csv"
    df = pd.read_csv(string_new)
    new = error.loc[error.monthS==i]
    df = pd.concat([df,new])
    df.reset_index(drop=True)
    df.to_csv("Filtered"+str(i)+".csv")
    del df
    del new
    
# ## RQ1 In what period of the year Taxis are used more?
# #### Create a plot that, for each month, shows the average number of trips recorded each day. Due to the differences among New York zones, we want to visualize the same information for each boroughs. Do you notice any difference among them? Provide comments and plausible explanations about what you observe (e.g.: what is the month with the highest daily average?)

# #### First, we imported all the libraries we will use to answer the question


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from scipy import stats
import folium
import json
get_ipython().magic('matplotlib inline')


# ##### we started the work by importing all the datasets for each month with Pandas, summed the number of daily trips for each month dividing by the number of days of the month , obtaining the desired average. After this operation we plot the result with seaborn. The first Graph show the average trips per Day.

# In[2]:

months = 'January February March April May June'.split()
df_new = pd.Series(index = months)
for i in range(num_plots):
    string = "Filtered"+str(i+1)+".csv"
    df = pd.read_csv(string,usecols=["tpep_pickup_datetime","monthS","weekdayS"])
    df=df.loc[df.monthS==(i+1)]
    df['day'] = pd.to_datetime(df["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.day
    df.dropna(inplace=True)
    df_new[months[i]] = len(df)/len((df["day"].unique().tolist()))
    del df
ax = sns.barplot(x=df_new.index, y=df_new.values)
ax.set(xlabel='Month', ylabel='Average Trips per Day')


# In[ ]:



# #### after this we plot the same graph for every Borough. 

# In[9]:


dataframe=[]
zones = pd.read_csv("taxi_zone_lookup.csv")
for i in range(1,7):
    df = pd.read_csv("Filtered"+str(i)+".csv",usecols=["tpep_pickup_datetime","monthS","DOLocationID"])
    df=df.loc[df.monthS==i]
    df = pd.merge(df,zones[['LocationID','Borough']], left_on=['DOLocationID'], right_on=['LocationID'])[['Borough',"tpep_pickup_datetime","monthS"]]
    df['day'] = pd.to_datetime(df["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.day
    df2 =(df.groupby(['monthS','Borough']).count()['day']//len((df["day"].unique().tolist())))
    dataframe.append(df2)
    d=pd.concat(dataframe)
    d.rename(index={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June'},inplace=True)
    d.unstack(level=0).plot.bar(width=1.0, figsize=(8, 8))
# # RQ2 What are the time slots with more passengers?
# #### Set your own time slots and discover which are those when Taxis drive the highest number of passengers overall New York and repeat the analysis for each borough. Provide the results through a visualization and comment them

# In[14]:


time_slots = ['1.[1-6 AM]','2.[7-12 AM]','3.[13-18 PM]','4.[19-00 PM]']
df_total = pd.DataFrame([])
for month in range(6):
    string = "Filtered"+str(i+1)+".csv""
    df = pd.read_csv(string,usecols=['hour','DOLocationID','passenger_count'])
    df.dropna(inplace=True)
    df_total = df_total.append(df,ignore_index=True)
    del df
df_total = pd.merge(df_total,zones[['LocationID','Borough']], left_on=['DOLocationID'], right_on=['LocationID'])[['Borough','hour','passenger_count']]
df_total['time_slots'] = df_total.hour.apply(lambda x:time_slots[abs((x-1)//6)])


# In[15]:


num_plots = 7
fig, axes = plt.subplots(num_plots,1,figsize=(30,30))
Boroughs = df_total.Borough.unique()
Boroughs.sort()
for Borough in Boroughs:
    df = df_total[df_total['Borough']==Borough]
    df=df.groupby('time_slots',as_index=False).sum()
    df.sort_values(by='time_slots',inplace=True)
    ax_curr = axes[np.where(Boroughs == Borough)[0][0]]
    sns.set_style("whitegrid")
    fg = sns.barplot(x=df.time_slots,y=df.passenger_count,ax=ax_curr)
    ax_curr.set_title(Borough)
    del df
fig.tight_layout()


# In[16]:


df_total=df_total.groupby('time_slots',as_index=False).sum()
df_total.sort_values(by='time_slots',inplace=True)
sns.set_style("whitegrid")
fig = sns.barplot(x=df_total.time_slots,y=df_total.passenger_count)
fig.set_title('Overall New York')


# # RQ3 Do the all trips last the same? 
# #### Let's put our attention on the distribution of trip's duration. Provide a plot for it and comment what you see. Run this analysis for NYC and for each borough (and obviously comment the results!).

# In[1]:
df=[]
for i in range(1,7):
    d = pd.read_csv("Filtered"+str(i)+".csv",usecols=["'tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID'])
    d['start-time'] = pd.to_datetime(d['tpep_pickup_datetime'])
    d['End-time'] = pd.to_datetime(d['tpep_dropoff_datetime'])
    d["duration"]=d["End-time"]-d["start-time"]
    d["durationm"]=d["duration"]/pd.Timedelta('1 minute')
    d["durations"]=d["duration"]/pd.Timedelta('1 second')
    d['years'] = pd.to_datetime(d["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.year
    d['yeare'] = pd.to_datetime(d["tpep_dropoff_datetime"],format='%Y-%m-%d %H:%M:%S').dt.year
    df.append(d)
    del d     

# In[2]:                                                      
dft=[]
for d in df:
    dft.append(d.loc[(d.durations>0)&(d.durations<5000)&(d.years==2018)&(d.yeare==2018)&(d.PULocationID!=264)&(d.PULocationID!=265)])

 # In[3]:
DF=pd.concat(dft)
DF=DF.reset_index(drop=True)
sns.distplot(DF["durations"])  
                                                      
# In[4]:                                                   
zone=pd.read_csv("taxi_zone_lookup.csv")
dfz=zone[['LocationID','Borough']]    
 
for d in dft:
    d.rename(columns={'PULocationID':'LocationID'},inplace=True)
dj=[]
for d in dft:
    dj.append(dfz.join(d.set_index('LocationID'), on='LocationID'))    
dj1=[]
for d in dj:
    d=d.reset_index(drop=True)
    dj1.append(d[["Borough","LocationID","durations"]])
dj1=pd.concat(dj1)
dj1=dj1.reset_index(drop=True)
subset=dj1["Borough"].unique().tolist()
subset=subset[0:6]  
                                                      
# In[5]:                                                       
for s in subset:
    l = dj1[dj1['Borough'] == s]
    l=l.dropna()
    # Draw the density plot
    sns.distplot(l['durations'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = s)
    
# Plot formatting
plt.legend(prop={'size': 16}, title = 'Duration')
plt.title('Density Plot with Multiple Borough')
plt.xlabel('Duration (second)')
plt.ylabel('Density')
                                                     
# # RQ4 What is the most common way of payments? 
# #### Discover the way payments are executed in each borough and visualize the number of payments for any possible means. Then run the Chi-squared test to see whether the method of payment is correlated to the borough. Then, comment the results.

# In[17]:


months = 'January February March April May June'.split()
df_total = pd.DataFrame([])
fig, axes = plt.subplots(1,1,figsize=(15,15))
for i in range(6):
    string = "yellow_tripdata_2018-0"+str(i+1)+"_new.csv"
    df = pd.read_csv(string)
    df = df[["DOLocationID","payment_type","month"]]
    df = pd.merge(df,zones[['LocationID','Borough']], left_on=['DOLocationID'], right_on=['LocationID'])[['Borough','payment_type','month']]
    df_total = df_total.append(df,ignore_index=True)
    del df
df_total['month'] = df_total['month'].apply(lambda x : months[x-1])
sns.set_style("whitegrid")
fg = sns.countplot(x='Borough', hue='payment_type', data=df_total,palette='hls')
fg.set_title('Distribution of payment type for every Borough')


# In[18]:


manhattan_value = df_total.loc[df_total['Borough'] == 'Manhattan']['payment_type']
Bronx_value = df_total.loc[df_total['Borough'] == 'Bronx']['payment_type']
Brooklyn_value = df_total.loc[df_total['Borough'] == 'Brooklyn']['payment_type']
EWR_value = df_total.loc[df_total['Borough'] == 'EWR']['payment_type']
Queens_value = df_total.loc[df_total['Borough'] == 'Queens']['payment_type']
State_Island_value = df_total.loc[df_total['Borough'] == 'Staten Island']['payment_type']


# In[21]:


chi_test=np.array([manhattan_value.value_counts().sort_index(axis=0),Bronx_value.value_counts().sort_index(axis=0),Brooklyn_value.value_counts().sort_index(axis=0),EWR_value.value_counts().sort_index(axis=0),Queens_value.value_counts().sort_index(axis=0),State_Island_value.value_counts().sort_index(axis=0)]).T
chi_test_2=pd.DataFrame(chi_test)
chi2_stat, p_val, dof, ex = stats.chi2_contingency(chi_test_2)
print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)


# # RQ5 Does a long distance correlate with the duration of the trip on average?
# #### Make a plot that show the dependence between distance and duration of the trip. Then compute the Pearson Coefficient, is it significant? Comment the results you obtain.

# In[2]:


df_total = pd.DataFrame([])
for month in range(6):
    string = "yellow_tripdata_2018-0"+str(month+1)+"_new.csv"
    df = pd.read_csv(string,usecols=['tpep_pickup_datetime','tpep_dropoff_datetime','trip_distance'])
    df = df[df['trip_distance']>2]  ##take long distances
    df.dropna(inplace=True)
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['trip_duration'] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
    df['trip_duration'] = pd.to_timedelta(df['trip_duration']).apply(lambda x:x.seconds/60)
    df_total = df_total.append(df[['trip_distance','trip_duration']],ignore_index=True)
    del df


# In[5]:


df1 = df_total[(df_total['trip_duration'] <= 100) & (df_total['trip_distance'] <= 40)]
ax = sns.regplot(x="trip_distance", y="trip_duration", data=df1.sample(5000))


# In[6]:


correlation = df1.corr(method='pearson')
print(correlation)


# # CRQ1: Does the fare for mile change across NY's borough? 
# ## #We want to discover whether the expenses of a user that enjoys Taxis in one zone is different from those that uses it in another one.
# #### 1)Considering the fare amount:
# #### -Compute the price per mile equation for each trip.
# #### -Run the mean and the standard deviation of the new variable for each borough. Then plot the distribution. What do you see?
# #### -Run the t-test among all the possible pairs of distribution of different boroughs.
# #### -Can you say that statistically significant differences, on the averages, hold among zones? In other words, are Taxis trip in some boroughs, on average, more expensive than others?
# #### 2)The price per mile might depend on traffic the Taxi finds on its way. So we try to mitigate this effect:
# #### -Likely, the duration of the trip says something about the city's congestion, especially if combined with the distances. It might be a good idea to weight the price for mile using the T time equation needed to complete the trip. Thus, instead of P equation, you can useP'=P/T equation, where T equation is the time needed to complete the trip.
# #### -Run the mean and the standard deviation of the new variable for each borough. Then plot the distribution. What do you see?
# #### -Run the t-test among all the possible pairs of new distribution of different boroughs.
# #### -Can you say that statistically significant differences, on the averages, hold among zones? In other words, are Taxis trip in some boroughs, on average, more expensive than others?
# #### -3)Compare the results obtained for the price per mile and the weighted price for mile. What do you think about that?
                                                      
# In[1]:
dft=[]
for i in range(1,7):
    df=pd.read_csv("C:/Users/Atefeh/Desktop/ADM2/Filtered"+str(i)+".csv",usecols=['fare_amount','trip_distance','tpep_dropoff_datetime','tpep_pickup_datetime','PULocationID','total_amount'])
    df["Pricem"]=df["fare_amount"]/df["trip_distance"]
    df['start-time'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['End-time'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df["durationm"]=(df["End-time"]-df["start-time"])/pd.Timedelta('1 minute')
    df["Pricem/m"]=df["Pricem"]/df["durationm"]
    df=df.loc[(df.durationm>20) & (df.fare_amount>0) & (df.trip_distance>0.1) & (df.total_amount>0.8) & (df.PULocationID!=264) & (df.PULocationID!=265)]
    df=df.reset_index(drop=True)
    dft.append(df[["Pricem","PULocationID","Pricem/m"]])
    del df
dft=pd.concat(dft)
dft=dft.reset_index(drop=True)

d=dft.loc[dft.Pricem<10]
sns.distplot(d['Pricem'],kde=True,bins=15,hist_kws=dict(edgecolor="k", linewidth=2),axlabel="NY")         
                                                      
# In[2]:
zone=pd.read_csv("taxi_zone_lookup.csv")
dfz=zone[['LocationID','Borough']]
d.rename(columns={'PULocationID':'LocationID'},inplace=True)
dfz=dfz.join(d.set_index('LocationID'), on='LocationID')
dfz=dfz.reset_index(drop=True)   
                                                      
# In[3]:
Boroughs = (dfz["Borough"].unique().tolist())[0:6]
dm=pd.DataFrame(columns=['mean','std'],index=Boroughs) 
                                                      
# In[4]:                                                      
import matplotlib.pyplot as plt
i=0
Boroughs = (dfz["Borough"].unique().tolist())[0:6]
fig, axes = plt.subplots(6,1,figsize=(30,30))
for l in Boroughs:
    a=dfz.loc[dfz.Borough==l]
    a=a.dropna()
    m=a["Pricem"].mean()
    s=a["Pricem"].std()
    dm.loc[l,"mean"]=m
    dm.loc[l,"std"]=s
    #ax_curr = axes[np.where(Boroughs == l)[0][0]]
    sns.distplot(a['Pricem'],kde=True,bins=20,color='red',hist_kws=dict(edgecolor="k", linewidth=2),ax=axes[i],axlabel=l)
    #sns.distplot(a['Pricem'],kde=True,bins=15,hist_kws=dict(edgecolor="k", linewidth=2,alpha=0.5),ax=ax_curr,label='Overall New York')
    #axes.legend()
    sns.despine()
    i+=1
 dm

# In[5]:                                                       
from scipy import stats
boroughs=dfz["Borough"].unique().tolist()[0:6]                                                     
Boroughs1=boroughs     
idx = pd.MultiIndex.from_product([boroughs,
                                  ['t-value', 'p-value', 'H0 hypothesis']])
col = boroughs
dt = pd.DataFrame('-', idx, col)           
for i in boroughs:
    a=dfz.loc[dfz.Borough==i]["Pricem"]
    a=a.dropna()
    for j in Boroughs1:
            b=dfz.loc[dfz.Borough==j]["Pricem"]
            b=b.dropna()
            t2, p2 = stats.ttest_ind(a,b)
            dt.loc[(i,"t-value"),j]=t2
            dt.loc[(i,"p-value"),j]=p2
            if(p2>0.05):
                dt.loc[(i,"H0 hypothesis"),j]='Fail to Reject H0'
            else:
                dt.loc[(i,"H0 hypothesis"),j]='Reject H0'  
 dt                                                     
#print(dm)                                                      
# # CRQ2: Visualize Taxis movements! 
# #### NYC is divided in many Taxis zones. For each yellow cab trip we know the zone the Taxi pick up and drop off the users. Let's visualize, on a chropleth map, the number of trips that starts in each zone. Than, do another map to count the races that end up in the single zone. Comment your discoveries

# In[7]:


zones = pd.read_csv("taxi_zone_lookup.csv")
value_to_choropleth = pd.DataFrame([])
for i in range(6):
    string = "yellow_tripdata_2018-0"+str(i+1)+"_new.csv"
    df = pd.read_csv(string,usecols=["PULocationID","DOLocationID"])
    df.dropna(inplace=True)
    value_to_choropleth=value_to_choropleth.append(df,ignore_index=True)
    del df


# #### Number of trips that starts in each zone

# In[8]:


value_to_choropleth_start = pd.merge(value_to_choropleth,zones, left_on=['PULocationID'], right_on=['LocationID'])[['Zone','PULocationID']]
test=value_to_choropleth_start.groupby(['Zone'],as_index=False).count()
test.columns=['Zone','Counter']
value_to_choropleth_start2 = pd.merge(test,zones, left_on=['Zone'], right_on=['Zone'])


# In[9]:


map_1 = folium.Map(location=[40.767937,-73.982155 ],tiles='OpenStreetMap',
 zoom_start=12)


# In[12]:


with open('taxi_zones.json') as f:
    data = json.load(f)


# In[14]:


map_1.choropleth(
    geo_data='taxi_zones.json',
    data=value_to_choropleth_start2,
    columns=['LocationID','Counter'],
    key_on='feature.properties.LocationID',
    fill_color='YlOrRd',
    legend_name='Taxi Counter',
    fill_opacity=0.7,
    line_opacity=0.2,
    threshold_scale = [20,200,2000,20000,200000,2000000]
    
)
map_1.save('GeoJSON_New_york_Start.html')


# #### Number of trips that end in each zone

# In[15]:


value_to_choropleth_end = pd.merge(value_to_choropleth,zones, left_on=['DOLocationID'], right_on=['LocationID'])[['Zone','DOLocationID']]
test1=value_to_choropleth_end.groupby(['Zone'],as_index=False).count()
test1.columns=['Zone','Counter']
value_to_choropleth_end2 = pd.merge(test1,zones, left_on=['Zone'], right_on=['Zone'])


# In[16]:


map_1.choropleth(
    geo_data='taxi_zones.json',
    data=value_to_choropleth_end2,
    columns=['LocationID','Counter'],
    key_on='feature.properties.LocationID',
    fill_color='YlOrRd',
    legend_name='Taxi Counter',
    fill_opacity=0.7,
    line_opacity=0.2,
    threshold_scale = [20,200,2000,20000,200000,2000000]
    
)
map_1.save('GeoJSON_New_york_end.html')

