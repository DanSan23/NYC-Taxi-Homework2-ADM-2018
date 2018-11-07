
# coding: utf-8

# In[ ]:


#list of months
months = 'January February March April May June'.split()
#create a new DataFrame
df_new = pd.Series(index = months)
for i in range(6):
    string = "Filtered"+str(i+1)+".csv"
    #read csv and take only tpep_pickup_datetime with usecols
    df = pd.read_csv(string,usecols=["tpep_pickup_datetime"])
    #using datetime in order to convert argument to datetime.
    df['day'] = pd.to_datetime(df["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.day
    #delete missing value
    df.dropna(inplace=True)
    #computation of average daily trip for every month
    df_new[months[i]] = len(df['day'])/(len(df['day'].unique()))
    #free memory
    del df
ax = sns.barplot(x=df_new.index, y=df_new.values)
ax.set(xlabel='Month', ylabel='Average Trips per Day')


# In[ ]:


#create a list
dataframe=[]
#read csv file of zone
zones = pd.read_csv("taxi_zone_lookup.csv")
for i in range(1,7):
    #read csv file of taxi and take only tpep_pickup_datetime,monthS and DOLocationID with usecols
    df = pd.read_csv("Filtered"+str(i)+".csv",usecols=["tpep_pickup_datetime","monthS","DOLocationID"])
    #select the specific month
    df=df.loc[df.monthS==i]
    #merge Data Frame of taxi with DataFrame of zone and take only Borough,tpep_pickup_datetime and monthS column
    df = pd.merge(df,zones[['LocationID','Borough']], left_on=['DOLocationID'], right_on=['LocationID'])[['Borough',"tpep_pickup_datetime","monthS"]]
    #using datetime in order to convert argument to datetime
    df['day'] = pd.to_datetime(df["tpep_pickup_datetime"],format='%Y-%m-%d %H:%M:%S').dt.day
    #computation of daily trip average for every month in every Borough, 
    df2 =(df.groupby(['monthS','Borough']).count()['day']//len((df["day"].unique().tolist())))
    dataframe.append(df2)
d=pd.concat(dataframe)
#rename of index, with inplace=True the data is renamed in place
d.rename(index={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June'},inplace=True)
#We have a multi index series(the output of groupby), ustack function on level 0 convert the 0 level of index as a columns 
         #and the level -1 as an index"""
d.unstack(level=0).plot.bar(width=1.0, figsize=(8, 8))

