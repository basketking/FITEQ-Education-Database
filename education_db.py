import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


header1, image1 = st.columns([2,1])
header1.header('Education Database - Numbers')
image1.image("https://www.sportaccord.sport/wsbs-2022/wp-content/uploads/sites/2/2020/11/Fiteq1800x1200-e1605615045111-1024x683.png", width = None )
st.subheader('Filters')

## Load Dataframe

df_ref = pd.read_excel("https://github.com/basketking/FITEQ-Education-Database/blob/main/Education%20database.xlsx?raw=true", sheet_name = 'Referee database')
df_coach = pd.read_excel("https://github.com/basketking/FITEQ-Education-Database/blob/main/Education%20database.xlsx?raw=true", sheet_name = 'Coach database')

## Create only relevant information in each table --> Continent, Nationality, Total Ref, Female Ref, Male Ref      - 
#                                                     Continent, Nationality, Total Coach, Female Coach, Male Coach
df_ref['Male'] = df_ref['Gender'].apply(lambda x: 1 if x =="Male" else 0)

df_ref['Female'] = df_ref['Gender'].apply(lambda x: 1 if x =="Female" else 0)


df_coach['Male'] = df_coach['Gender'].apply(lambda x: 1 if x =="Male" else 0)

df_coach['Female'] = df_coach['Gender'].apply(lambda x: 1 if x =="Female" else 0)


df_ref['Referee'] = 1

df_coach['Coach'] = 1


## Group ref and coach and merge

df_ref_grouped = df_ref.groupby(['Continent', 'Nationality']).agg({'Name': 'count',
                                                  'Male': 'sum',
                                                  'Female':  'sum'}).rename(columns ={'Name': 'Total Referees',
                                                                                      'Male': 'Male Referees',
                                                                                      'Female': 'Female Referees'})

df_coach_grouped = df_coach.groupby(['Continent', 'Nationality']).agg({'Name': 'count',
                                                  'Male': 'sum',
                                                  'Female':  'sum'}).rename(columns ={'Name': 'Total Coaches',
                                                                                      'Male': 'Male Coaches',
                                                                                      'Female': 'Female Coaches'})

df_education = pd.merge(df_ref_grouped, df_coach_grouped, left_index = True, right_index = True, how = 'outer').fillna(0)
df_education = df_education.astype(int)
df_education.reset_index(inplace = True)
df_education.rename(columns={'0': 'Continent', '1': 'Nationality'}, inplace=True)
nf_list = ['Algeria', 'Benin', 'Burundi', 'Cameroon', 'Cape Verde', 'Chad', 'Djibouti', 'Equatorial Guinea', 'Eswatini', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritius', 'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'Togo', 'Tunisia', 'Zambia', 'Zimbabwe', 'Afghanistan', 'Bahrain', 'Brunei', 'Cambodia', 'Hong Kong China', 'India', 'Indonesia', 'Iran', 'Iraq', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Lebanon', 'Malaysia', 'Mongolia', 'Nepal', 'Pakistan', 'Palestine', 'Philippines', 'Republic of Korea', 'Singapore', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkmenistan', 'Uzbekistan', 'Yemen', 'Albania', 'Armenia', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czech Republic', 'France', 'Georgia', 'Hungary', 'Italy', 'Kosovo', 'Luxembourg', 'Malta', 'Moldova', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Slovakia', 'Slovenia', 'Switzerland', 'Ukraine', 'Cook Islands', 'Guam', 'New Caledonia', 'Papua New Guinea', 'Samoa', 'Tuvalu', 'Vanuatu', 'Antigua and Barbuda', 'Bahamas', 'Belize', 'Bermuda', 'Canada', 'Chile', 'Guatemala', 'Guyana', 'Haiti', 'Jamaica', 'Panama', 'Peru', 'Saint Kitts and Nevis', 'Trinidad and Tobago', 'United States', 'Uruguay', 'Venezuela']
df_education['NF'] = df_education['Nationality'].apply(lambda x : x in nf_list)
df_education = df_education[['Continent', 'Nationality', 'NF', 'Total Referees', 'Total Coaches', 'Male Referees', 'Female Referees', 'Male Coaches', 'Female Coaches']]
## st.dataframe(df_education)


## --------- CONTINENT SELECTION

continent = df_education['Continent'].unique().tolist()

continent_selection = st.multiselect('Continent',
                                        continent,
                                        default = continent)



## ------- NATIONAL FEDERATION SELECTION

nf = df_education['NF'].unique().tolist()
nf_selection = st.multiselect('National Federation',
                               nf,
                               default = nf)



## -------- FILTER DATAFRAME BASED ON SELECTION
mask = df_education['Continent'].isin(continent_selection) & df_education['NF'].isin(nf_selection)

col1, col2, col3 = st.columns(3)
number_of_countries = df_education[mask].shape[0]
number_of_referees = df_education[mask]['Total Referees'].sum()
number_of_coaches = df_education[mask]['Total Coaches'].sum()
col1.metric("Total Countries", number_of_countries)
col2.metric("Total Number of Referees", number_of_referees)
col3.metric("Total Number of Coaches", number_of_coaches)

col4, col5, col6, col7 = st.columns(4)
number_of_male_referees = df_education[mask]['Male Referees'].sum()
number_of_male_coaches = df_education[mask]['Male Coaches'].sum()
number_of_female_referees = df_education[mask]['Female Referees'].sum()
number_of_female_coaches = df_education[mask]['Female Coaches'].sum()

col4.markdown(f'Male Referees: {number_of_male_referees}')
col5.markdown(f'Female Referees: {number_of_female_referees}')
col6.markdown(f'Male Coaches: {number_of_male_coaches}')
col7.markdown(f'Female Coaches: {number_of_female_coaches}')

### --------- ADD SOME CHARTS 

st.subheader('Some interesting charts')

# CONTINENTAL REPRESENTATION

labels_cont = ['Africa', 'Asia', 'Europe', 'Pan America', 'Oceania']
sizes_cont = [df_education[df_education['Continent'] == 'Africa'].shape[0], 
            df_education[df_education['Continent'] == 'Asia'].shape[0], 
            df_education[df_education['Continent'] == 'Europe'].shape[0], 
            df_education[df_education['Continent'] == 'Pan America'].shape[0], 
            df_education[df_education['Continent'] == 'Oceania'].shape[0]]
fig1, ax1 = plt.subplots()
ax1.barh(np.arange(len(sizes_cont)), sizes_cont)
ax1.set_yticklabels(['empty', 'Africa', 'Asia', 'Europe', 'Pan America', 'Oceania'])
for i, v in enumerate(sizes_cont):
    ax1.text(v, i, str(v) + '%', color='black', fontweight='bold')

with st.expander("Percentage of people educated (referees + coaches) per continent"):
    st.pyplot(fig1)


## TOP 10 PERFORMING COUNTRIES 
top10_ref = df_education.sort_values(by='Total Referees', ascending = False)[0:10]
top10_coach = df_education.sort_values(by='Total Coaches', ascending = False)[0:10]

fig2, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(top10_ref['Nationality'], top10_ref['Total Referees'])
ax2.bar(top10_coach['Nationality'], top10_coach['Total Coaches'])
fig2.autofmt_xdate(rotation= 90, ha= 'center')
pps1 = ax1.bar(np.arange(len(top10_ref)), top10_ref['Total Referees'])
for p in pps1:
   height = p.get_height()
   ax1.annotate('{}'.format(height),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 1), # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom', size = 7)

pps2 = ax2.bar(np.arange(len(top10_coach)), top10_coach['Total Coaches'])
for p in pps2:
   height = p.get_height()
   ax2.annotate('{}'.format(height),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 1), # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom', size = 7)
with st.expander("Top 10 performing countries in Referee and Coach education"):
    st.pyplot(fig2)


## display masked table

st.subheader('Filtered table')

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(df_education[mask])

st.download_button(
     label="Download table as CSV",
     data=csv,
     file_name='education_database.csv',
     mime='text/csv',
 )

st.table(df_education[mask])


## -------- Group Data by selection


