#!/usr/bin/env python
# coding: utf-8

# CoG Colculation
# =========================================
#              Import Libraries           #
# =========================================

import os
import pandas as pd
import csv
import numpy as np
from geopy.geocoders import Nominatim

#%matplotlib inline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from itertools import combinations 
from sklearn.cluster import KMeans
#import folium

# For UI
#import tkinter as tk
#from tkinter import filedialog
#from tkinter import simpledialog
import streamlit as st
from streamlit_folium import folium_static
import folium

#Evaluation
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist


# =========================================
#          Define the Objects Color       #
# =========================================
color_options = {'demand':    'green',
                 'supply':    'Orange',
                 'flow':      'gray',
                 'cog':       'Blue',
                 'candidate': 'yellow',
                 'other':     'black'}


def no_dc_centers(valueset):
    if (valueset == '2'):
        no = 2
    elif (valueset == '3'):
        no = 3
    elif (valueset == '4'):
        no = 4
    elif (valueset == '5'):
        no = 5
    elif (valueset == '6'):
        no = 6
    elif (valueset == '7'):
        no = 7
    else:
        no = 4
    return no

# =========================================
#  Create map frame with location points  #
# =========================================
def location_map_frame(data):
    
    datamap = folium.Map(location = data[['Latitude', 'Longitude']].mean(),
                       fit_bounds = [[data['Latitude'].min(),data['Longitude'].min()],                             
                                    [data['Latitude'].max(),data['Longitude'].max()]])
                            
                            
    # Add unit weight points
    for _, row in data.iterrows():
        folium.CircleMarker(location=[row['Latitude'],row['Longitude']],                                   
                            radius=(row['Del GrossWeight_KG']**0.11),
                            color=color_options.get(str(row['Location Type']).lower(), 'gray'),
                            tooltip=str(row['city'])+' '+str(row['Del GrossWeight_KG'])).add_to(datamap)
                                    #row['Longitude']]).add_to(datamap)
        
    # Zoom
    datamap.fit_bounds(data[['Latitude', 'Longitude']].values.tolist())
    # Show
    #folium_static(datamap)
    return datamap


# =========================================
#              COG Calculation            #
# =========================================
def cog_calculation(n_centers, data):

    numberofDCcenter = n_centers
    kmeans = KMeans(n_clusters = numberofDCcenter,
                    #init = 'random', #default = k-means++
                    #n_init = 10,     #default = 10
                    max_iter = 300,  #default = 300
                    #tol = 0.0001,  #convergence
                    random_state=0).fit(data.loc[data['Calc_unitweight'] > 0, ['Latitude','Longitude']],                                                                   
                                        sample_weight = data.loc[data['Calc_unitweight'] > 0, 
                                                               'Calc_unitweight'])
    # Get centers of gravity from K-means
    cogs = kmeans.cluster_centers_
    cogs = pd.DataFrame(cogs, columns=['Latitude','Longitude'])

    # Get unit weight assigned to each cluster
    data['Cluster'] = kmeans.predict(data[['Latitude', 'Longitude']])
    cogs = cogs.join(data.groupby('Cluster')['Del GrossWeight_KG'].sum())

    # assigned COG coordinates in data by point 
    data = data.join(cogs, on='Cluster', rsuffix='_COG')
    return cogs,data,kmeans.n_iter_



# =========================================
#   Get CoG city name by Lat&Long         #
# =========================================
def cog_city_name(dfcog):
    
    geolocator = Nominatim(user_agent="csclog_cog")
    list_city = []
    list_country = []

    for index, row in dfcog.iterrows():

        try:
            #print(index)
            Latitude = dfcog.loc[index,'Latitude']
            Longitude = dfcog.loc[index,'Longitude']
            #print(Latitude,Longitude)
            location = geolocator.reverse("{}, {}".format(Latitude, Longitude))
            address = location.raw['address']

            city = address.get('city', '')
            state = address.get('state', '')
            country = address.get('country', '')
            code = address.get('country_code')
            zipcode = address.get('postcode')
            list_country.append(country)
            list_city.append(state + "/" + city)
            #print('City : ',city)
            #print('State : ',state)
            #print('Country : ',country)
            #print('Zip Code : ', zipcode)
        except:
            #print('Out of range')
            country = "Others"
            state ="Others"
            list_country.append(country)
            list_city.append(state)

    dfcog['country']=list_country
    dfcog['city']=list_city    
    return dfcog


# In[8]:


# =========================================
#   Add flow line between CoG and site    #
# =========================================
def add_flow_line(dframe,cogs,dmap):
    
    for _, row in dframe.iterrows():
        # Flow lines
        if str(row['Location Type']).lower() in (['demand', 'supply']):
            folium.PolyLine([(row['Latitude'],row['Longitude']),                         
                             (row['Latitude_COG'],row['Longitude_COG'])],                          
                            color=color_options['flow'],
                            weight=(row['Del GrossWeight_KG']**0.1),
                            opacity=0.8).add_to(dmap)

    # Add COG
    for _, row in cogs.iterrows():

        folium.CircleMarker(location=[row['Latitude'],row['Longitude']],

                            fill = True,
                            radius = (row['Del GrossWeight_KG']**0.15),
                            color = color_options['cog'],
                            fill_color = color_options['cog'],
                            tooltip = str(row['country']) + " " + str(row['city']) + " " + str(row['Del GrossWeight_KG'])).add_to(dmap)    

    return dmap
    #dmap



# =========================================
#           Load shipment data            #
# =========================================

def load_data(file_path):    
    #CSV file
    data = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin-1', dtype = {'City': str,'Location Type': str})
    #data
    return data

# =========================================
#          Transportation rate            #
# =========================================

def loc_type_mult(x):

    IB_ratio_OB = ib_ratio_ob
    
    if x.lower() == 'supply':       
        return 1
    elif x.lower() == 'demand':       
        return IB_ratio_OB
    else:        
        return 0


def transportation_ratio():         
    # Adjust unit weight as "Calc_unitweight"
    data['Calc_unitweight'] = data['Location Type'].apply(str).apply(loc_type_mult)*data['Del GrossWeight_KG']


# =========================================
#          Add COG Legend on map          #
# =========================================

def add_legend_map(dmap):
    
    from branca.element import Template, MacroElement
    template = """
    {% macro html(this, kwargs) %}
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>jQuery UI Draggable - Default functionality</title>
      <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
      
      <script>
      $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

      </script>
    </head>
    <body>

     
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
         border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
         
    <div class='legend-title'>CoG Legend (draggable)</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        <li><span style='background:blue;opacity:0.7;'></span>CoG Location</li>
        <li><span style='background:orange;opacity:0.7;'></span>Ship from Location</li>
        <li><span style='background:green;opacity:0.7;'></span>Ship to Location</li>

      </ul>
    </div>
    </div>
     
    </body>
    </html>

    <style type='text/css'>
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    macro = MacroElement()
    macro._template = Template(template)
    dmap.get_root().add_child(macro)   
    return dmap


# =========================================
#           K value Evaluating            #
# =========================================
def data_evaluating(X):
    seed_random = 1
    fitted_kmeans = {}
    labels_kmeans = {}
    df_scores = []
    k_values_to_try = np.arange(2, 15)  # pre-set DCs arange
    for n_clusters in k_values_to_try:

        #Perform clustering.
        kmeans = KMeans(n_clusters = n_clusters,
                        random_state = seed_random,
                        )
        labels_clusters = kmeans.fit_predict(X)

        #Insert fitted model and calculated cluster labels in dictionaries,
        #for further reference.
        fitted_kmeans[n_clusters] = kmeans
        labels_kmeans[n_clusters] = labels_clusters

        #Calculate various scores, and save them for further reference.
        silhouette = silhouette_score(X, labels_clusters)
        ch = calinski_harabasz_score(X, labels_clusters)
        db = davies_bouldin_score(X, labels_clusters)
        tmp_scores = {"no_clusters/DC centers": n_clusters,
                      "silhouette_score": silhouette,
                      "calinski_harabasz_score": ch,
                      "davies_bouldin_score": db,
                      }
        df_scores.append(tmp_scores)

    #Create a DataFrame of clustering scores, using `n_clusters` as index, for easier plotting.
    df_scores = pd.DataFrame(df_scores)
    df_scores.set_index("no_clusters/DC centers", inplace = True)
    return df_scores


def silh_score_plot(df_scores):
    #st.write("")
    st.write("Silhouette Score: ")
    fig = plt.figure(figsize=(6, 4))
    plt.xlabel(r'no_clusters/DC centers *k*')
    plt.ylabel('silhouette_score');
    plt.plot(df_scores["silhouette_score"])
    plt.title("Silhouette Score - the optimal K")
    st.pyplot(fig)
    

def elbow_plot(X):
    #st.write("")
    st.write("Elbow point: ")
    distortions = []
    K = range(2, 12)
    for k in K:
        kmeanModel = KMeans(n_clusters = k).fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
    # Plot the elbow
    fig = plt.figure(figsize=(6, 4))
    plt.plot(K, distortions, 'bx-') #bx-
    plt.xlabel('K no of DC centres')
    plt.ylabel('Distortion')
    plt.title('Elbow Method - the optimal K')
    st.pyplot(fig)



# =========================================
#             Save map to file            #
# =========================================
def save_map():
    
    fmap ="CoG_Map_of_" + str(numberofDCcenter)+"_DC.html"
    fpmap = os.path.join(os.getcwd(), "COG_data/CoG data", fmap)
    datamap.save(fpmap)


# =========================================
#     Save CoG location data to file      #
# =========================================
def save_cog_data():
    
    f_Cog ="CoG_Data_of_" + str(numberofDCcenter)+"_DC.csv"
    fpCog = os.path.join(os.getcwd(), "COG_data/CoG data", f_Cog)
    cogs.to_csv(fpCog, index = False, encoding='utf_8_sig')


# =========================================
#       Save Clustered data to file       #
# =========================================
def save_cluster_data():
    
    f_Name ="Cluster Data_of_" + str(numberofDCcenter)+"_DC.csv"
    file_path = os.path.join(os.getcwd(), "COG_data/CoG data", f_Name)
    data.to_csv(file_path, index = False)


# =========================================
#                   GUI                   #
# =========================================

st.set_page_config(
    page_title ="COG",
    layout ="wide"  #centered/wide
    ) 

with st.sidebar:
    col_left, col_middle, col_right = st.columns([1,1,1])
    col_middle.image("warehouse.png")

    #Header
    sideHeader = "LOG Center-of-Gravity Analysis"
    st.header(sideHeader)

    #file loader
    uploaded_file = st.file_uploader("Data file:", type =['csv'])
    #st.write("__________")

    #LocationTypeSeleced = st.multiselect('Location Type: ', ['Demand','Supply'])
  
    #cog centers
    numberof_centers = st.selectbox(
        'No of Centers (DC):',('2','3','4','5','6','7'))
    st.write("Centers selected : ", numberof_centers)
    #st.write("__________")
    
    #Transportation Ratio 
    transportation_rate = st.slider(
        label ='Inbound-ratio-Outbound: ',
        min_value = 1.0,
        max_value = 10.0,
        step = 0.1)
    st.write("Ratio set : ", transportation_rate)

    
    Cal_button = st.button(label = 'Calculate')

    st.write("__________")
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    method_option =st.selectbox('Evaluation Methods: ',
                             ('Elbow method','Silhouette analysis',))

    Eval_button = st.button(label = 'Evaluate')
    #st.write("")

      
#st.header('Logistics Centre-of-Gravity')    


if(Cal_button):
    if uploaded_file is not None:
        #fpath = os.path.join(os.getcwd(),"shipmentdata",selected_filename)
        number_centers = no_dc_centers(numberof_centers)
        ib_ratio_ob = transportation_rate
        #st.write("Selected file is :",fpath)
        
        st.write("- Location data map:  **Green** = shipto location,  **Orange** = ship from location")
        
        df = load_data(uploaded_file)        
        df_supply = df.loc[df['Location Type'] == "Supply"]

        #if (LocationTypeSeleced == "Supply"):
            #do somethin

        df_map = location_map_frame(df)
        folium_static(df_map)
           
        #price
        df['Calc_unitweight'] = df['Location Type'].apply(str).apply(loc_type_mult)*df['Del GrossWeight_KG']
        #df
    
        #COG calculating
        df_cogs, df, no_ofIterationRun = cog_calculation(number_centers,df)
        #st.write("Number of iteration run",no_ofIterationRun)
        #df
    
        #get cog city name
        df_cogs = cog_city_name(df_cogs)
        #df_cogs
        
        #add flow line between CoG and site
        st.write("- COG (**Blue** circle) map:")
        df_map = add_flow_line(df,df_cogs,df_map)
        #folium_static(df_map)

        #add legend
        cog_map = add_legend_map(df_map)
        folium_static(cog_map)
        
        #
        st.write("- COG location: ")
        st.write(df_cogs)
    else:
        st.warning("No data file loaded !")



if (Eval_button):
    if uploaded_file is not None:
        Emethod = method_option
        df_E0 = load_data(uploaded_file)
        df_E1 = df_E0.loc[df_E0['Del GrossWeight_KG'] > 0, ['Latitude','Longitude']]
        sc = data_evaluating(df_E1)

        if (Emethod == "Silhouette analysis"):
            silh_score_plot(sc)
        elif (Emethod == "Elbow method"):
            elbow_plot(sc)
        else:
            st.write('something wrong...')
    else:
        st.warning("No data file loaded !")




        
