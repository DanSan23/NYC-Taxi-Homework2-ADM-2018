
# coding: utf-8

# In[ ]:


zones = pd.read_csv("taxi_zone_lookup.csv")
#set the time slots
time_slots = ['1.[1-6 AM]','2.[7-12 AM]','3.[13-18 PM]','4.[19-00 PM]']
#creation of new Dataframe
df_total = pd.DataFrame([])
for i in range(1,7):
    string = "Filtered"+str(i)+".csv"
    #import the dataset with required columns
    df = pd.read_csv(string,usecols=['hourS','DOLocationID','passenger_count'])
    #delete the missing value
    df.dropna(inplace=True)
    #add to the new Dataframe
    df_total = df_total.append(df,ignore_index=True)
    #free memory
    del df
#merge dt_total with zone.csv and take only required colums
df_total = pd.merge(df_total,zones[['LocationID','Borough']], left_on=['DOLocationID'], right_on=['LocationID'])[['Borough','hourS','passenger_count']]
df_total['time_slots'] = df_total.hourS.apply(lambda x:time_slots[abs((x-1)//6)])


# In[ ]:


#gropby and sum are used in order to have the total number of passengers for every time slot
df_total=df_total.groupby('time_slots',as_index=False).sum()
#sort the value
df_total.sort_values(by='time_slots',inplace=True)
#plot
sns.set_style("whitegrid")
fig = sns.barplot(x=df_total.time_slots,y=df_total.passenger_count)
fig.set_title('Overall New York')


# In[ ]:


num_plots = 7
fig, axes = plt.subplots(num_plots,1,figsize=(30,30))
#a list of boroughs is created
Boroughs = df_total.Borough.unique()
#sorting boroughs
Boroughs.sort()
for Borough in Boroughs:
    #chose the data frame with corresponding borough
    df = df_total[df_total['Borough']==Borough]
    #use groupby and .sum() in order to take the of passengers in each time slot
    df=df.groupby('time_slots',as_index=False).sum()
    #sort values
    df.sort_values(by='time_slots',inplace=True)
    #set the axes for the current subplot
    ax_curr = axes[np.where(Boroughs == Borough)[0][0]]
    sns.set_style("whitegrid")
    fg = sns.barplot(x=df.time_slots,y=df.passenger_count,ax=ax_curr)
    ax_curr.set_title(Borough)
    del df
fig.tight_layout()

