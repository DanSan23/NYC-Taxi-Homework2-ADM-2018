
# coding: utf-8

# In[ ]:


df_total = pd.DataFrame([])
for i in range(1,7):
    string = "Filtered"+str(i)+".csv"
    df = pd.read_csv(string,usecols=['tpep_pickup_datetime','tpep_dropoff_datetime','trip_distance'])
    df = df[df['trip_distance']>2]  
    df.dropna(inplace=True)
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['trip_duration'] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
    df['trip_duration'] = pd.to_timedelta(df['trip_duration']).apply(lambda x:x.seconds/60)
    df_total = df_total.append(df[['trip_distance','trip_duration']],ignore_index=True)
    del df


# In[ ]:


df1 = df_total[(df_total['trip_duration'] <= 100) & (df_total['trip_distance'] <= 40)]
ax = sns.regplot(x="trip_distance", y="trip_duration", data=df1.sample(2000))


# In[ ]:


correlation = df1.corr(method='pearson')
print(correlation)

