
# coding: utf-8

# In[ ]:


zones = pd.read_csv("taxi_zone_lookup.csv")
value_to_choropleth = pd.DataFrame([])
for i in range(6):
    string = "Filtered"+str(i+1)+".csv"
    df = pd.read_csv(string,usecols=["PULocationID","DOLocationID"])
    df.dropna(inplace=True)
    value_to_choropleth=value_to_choropleth.append(df,ignore_index=True)
    del df


# In[ ]:


#merge the DataFrame value_to_choropleth with zone.csv
value_to_choropleth_start = pd.merge(value_to_choropleth,zones, left_on=['PULocationID'], right_on=['LocationID'])[['Zone','PULocationID']]
#used groupby and count in order to have a dataframe with zone and the count of trips started in every zone
test=value_to_choropleth_start.groupby(['Zone'],as_index=False).count()
test.columns=['Zone','Counter']
value_to_choropleth_start2 = pd.merge(test,zones, left_on=['Zone'], right_on=['Zone'])


# In[ ]:


map_1 = folium.Map(location=[40.767937,-73.982155 ],tiles='OpenStreetMap',
 zoom_start=12)


# In[ ]:


with open('taxi_zones.json') as f:
    data = json.load(f)


# In[ ]:


# .choropleth function of folium is used to see the choropleth map
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


# In[ ]:


value_to_choropleth_end = pd.merge(value_to_choropleth,zones, left_on=['DOLocationID'], right_on=['LocationID'])[['Zone','DOLocationID']]
test1=value_to_choropleth_end.groupby(['Zone'],as_index=False).count()
test1.columns=['Zone','Counter']
value_to_choropleth_end2 = pd.merge(test1,zones, left_on=['Zone'], right_on=['Zone'])


# In[ ]:


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

