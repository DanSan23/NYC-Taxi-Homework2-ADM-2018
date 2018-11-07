
# coding: utf-8

# In[ ]:


df=[]
for i in range(1,7):
    #import the dataset
    d = pd.read_csv("Filtered"+str(i)+".csv",usecols=['tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID'])
    #create column called start_time where to_datetime is used to convert argument in tpep_pickup_datetime to datetime format
    d['start-time'] = pd.to_datetime(d['tpep_pickup_datetime'])
    #create column called End_time where to_datetime is used to convert argument in tpep_dropoff_datetime to datetime format
    d['End-time'] = pd.to_datetime(d['tpep_dropoff_datetime'])
    #compute difference between tpep_dropoff_datetime and tpep_pickup_datetime
    d["duration"]=d["End-time"]-d["start-time"]
    #convert the duration to minute and we assign into new coloum
    d["durationm"]=d["duration"]/pd.Timedelta('1 minute')
    d["durations"]=d["duration"]/pd.Timedelta('1 second')
    d['years'] = pd.to_datetime(d["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.year
    d['yeare'] = pd.to_datetime(d["tpep_dropoff_datetime"],format='%Y-%m-%d %H:%M:%S').dt.year
    #append to list df in order to concatenate in next sesssion
    df.append(d)
    del d


# In[ ]:


dft=[]
for d in df:
    #PULocationID which are corresponded to unknown zones are filtered
    dft.append(d.loc[(d.durations>0)&(d.durations<5000)&(d.PULocationID!=264)&(d.PULocationID!=265)])
del df    


# In[ ]:


#concatenate the list of datasets in dft list into DF dataframe
DF=pd.concat(dft)
DF=DF.reset_index(drop=True)
sns.distplot(DF["durations"])  


# In[ ]:


zone=pd.read_csv("taxi_zone_lookup.csv")
dfz=zone[['LocationID','Borough']]    

for d in dft:
    d.rename(columns={'PULocationID':'LocationID'},inplace=True)
dj=[]
for d in dft:
    #each dataframe in dft list is joined with the zone dataset 
    dj.append(dfz.join(d.set_index('LocationID'), on='LocationID'))    
dj1=[]
for d in dj:
    #for each dataframe in dj the index is reset and the previous index is dropped
    d=d.reset_index(drop=True)
    dj1.append(d[["Borough","LocationID","durations"]])
dj1=pd.concat(dj1)
dj1=dj1.reset_index(drop=True)
#take the unique values of borough and convert it to list
subset=dj1["Borough"].unique().tolist()
#the range is fom 0 to 6 because last one (7) is unknown
subset=subset[0:6]  


# In[ ]:


#in subset we have all boroughs 
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

