
# coding: utf-8

# In[ ]:


zones = pd.read_csv("taxi_zone_lookup.csv")
#list of month
months = 'January February March April May June'.split()
#creation of a DataFrame
df_total = pd.DataFrame([])
#plot specifications
fig, axes = plt.subplots(1,1,figsize=(15,15))
for i in range(1,7):
    string = "Filtered"+str(i)+".csv"
    #import dataset
    df = pd.read_csv(string)
    #select the columns useful for carrying out the analysis 
    df = df[["DOLocationID","payment_type","monthS"]]
    #merge df with the zone.csv and take only Borough,payment_type and month
    df = pd.merge(df,zones[['LocationID','Borough']], left_on=['DOLocationID'], right_on=['LocationID'])[['Borough','payment_type','monthS']]
    #add to the  df_total dataframe
    df_total = df_total.append(df,ignore_index=True)
    #free memory
    del df
#created a new column called df_total['month'] in order to assign the name to each month; to do this the lambda function is applied
df_total['month'] = df_total['monthS'].apply(lambda x : months[x-1])
sns.set_style("whitegrid")
fg = sns.countplot(x='Borough', hue='payment_type', data=df_total,palette='hls')
fg.set_title('payment type for every Borough')


# In[ ]:


manhattan_value = df_total.loc[df_total['Borough'] == 'Manhattan']['payment_type']
Bronx_value = df_total.loc[df_total['Borough'] == 'Bronx']['payment_type']
Brooklyn_value = df_total.loc[df_total['Borough'] == 'Brooklyn']['payment_type']
EWR_value = df_total.loc[df_total['Borough'] == 'EWR']['payment_type']
Queens_value = df_total.loc[df_total['Borough'] == 'Queens']['payment_type']
State_Island_value = df_total.loc[df_total['Borough'] == 'Staten Island']['payment_type']


# In[ ]:


list=[manhattan_value.value_counts().sort_index(axis=0),Brooklyn_value.value_counts().sort_index(axis=0),EWR_value.value_counts().sort_index(axis=0),Queens_value.value_counts().sort_index(axis=0),State_Island_value.value_counts().sort_index(axis=0)]
for i in list:
    i[5]=0
list.append(Bronx_value.value_counts().sort_index(axis=0))    


# In[ ]:


chi_test=np.array(list).T
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

