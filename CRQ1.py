
# coding: utf-8

# In[ ]:


dft=[]
for i in range(1,7):
    df=pd.read_csv("Filtered"+str(i)+".csv",usecols=['fare_amount','trip_distance','tpep_dropoff_datetime','tpep_pickup_datetime','PULocationID','total_amount'])
    df["Pricem"]=df["fare_amount"]/df["trip_distance"]
    df['start-time'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['End-time'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df["durationm"]=(df["End-time"]-df["start-time"])/pd.Timedelta('1 minute')
    df["Pricem/m"]=df["Pricem"]/df["durationm"]
    df=df.loc[(df.durationm>20) & (df.fare_amount>0) & (df.trip_distance>0.1) & (df.total_amount>0.8) & (df.PULocationID!=264) & (df.PULocationID!=265)]
    df=df.reset_index(drop=True)
    dft.append(df[["Pricem","PULocationID","Pricem/m"]])
    del df


# In[ ]:


dft=pd.concat(dft)
dft=dft.reset_index(drop=True)
d=dft.loc[dft.Pricem<10]


# In[ ]:


sns.distplot(d['Pricem'],kde=True,bins=15,hist_kws=dict(edgecolor="k", linewidth=2),axlabel="NY")


# In[ ]:


zone=pd.read_csv("taxi_zone_lookup.csv")
dfz=zone[['LocationID','Borough']]
d.rename(columns={'PULocationID':'LocationID'},inplace=True)
dfz=dfz.join(d.set_index('LocationID'), on='LocationID')
dfz=dfz.reset_index(drop=True)


# In[ ]:


Boroughs = (dfz["Borough"].unique().tolist())[0:6]
dm=pd.DataFrame(columns=['mean','std'],index=Boroughs)


# In[ ]:


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
#print(dm)


# In[ ]:


dm


# In[ ]:


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


# In[ ]:


dt


# In[ ]:


sns.distplot(d['Pricem/m'],kde=True,bins=15,hist_kws=dict(edgecolor="k", linewidth=2),axlabel="NY")


# In[ ]:


Boroughs = (dfz["Borough"].unique().tolist())[0:6]
dn=pd.DataFrame(columns=['mean','std'],index=Boroughs)


# In[ ]:


import matplotlib.pyplot as plt
i=0
Boroughs = (dfz["Borough"].unique().tolist())[0:6]
fig, axes = plt.subplots(6,1,figsize=(30,30))
for l in Boroughs:
    a=dfz.loc[dfz.Borough==l]
    a=a.dropna()
    m=a["Pricem/m"].mean()
    s=a["Pricem/m"].std()
    dn.loc[l,"mean"]=m
    dn.loc[l,"std"]=s
    #ax_curr = axes[np.where(Boroughs == l)[0][0]]
    sns.distplot(a['Pricem/m'],kde=True,bins=20,color='red',hist_kws=dict(edgecolor="k", linewidth=2),ax=axes[i],axlabel=l)
    #sns.distplot(a['Pricem'],kde=True,bins=15,hist_kws=dict(edgecolor="k", linewidth=2,alpha=0.5),ax=ax_curr,label='Overall New York')
    #axes.legend()
    sns.despine()
    i+=1


# In[ ]:


idx = pd.MultiIndex.from_product([boroughs,
                                  ['t-value', 'p-value', 'H0 hypothesis']])
col = boroughs
dtt = pd.DataFrame('-', idx, col)
for i in boroughs:
    print(i)
    a=dfz.loc[dfz.Borough==i]["Pricem/m"]
    a=a.dropna()
    for j in Boroughs1:
            b=dfz.loc[dfz.Borough==j]["Pricem/m"]
            b=b.dropna()
            t2, p2 = stats.ttest_ind(a,b)
            dtt.loc[(i,"t-value"),j]=t2
            dtt.loc[(i,"p-value"),j]=p2
            if(p2>0.05):
                dtt.loc[(i,"H0 hypothesis"),j]='Fail to Reject H0'
            else:
                dtt.loc[(i,"H0 hypothesis"),j]='Reject H0'


# In[ ]:


dtt

