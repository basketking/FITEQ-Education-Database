import pandas as pd
import streamlit as st
from PIL import Image




st.header('Education Database - Numberssss')
st.subheader('SZLAMB')

## Load Dataframe

df_ref = pd.read_excel("https://github.com/basketking/FITEQ-Education-Database/blob/main/Education_database.xlsx?raw=true", sheet_name = 'Referee database')
df_coach = pd.read_excel("https://github.com/basketking/FITEQ-Education-Database/blob/main/Education_database.xlsx?raw=true", sheet_name = 'Coach database')

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
number_of_countries = df_education[mask].shape[0]
number_of_referees = df_education[mask]['Total Referees'].sum()
number_of_coaches = df_education[mask]['Total Coaches'].sum()
st.metric("Total Number of Referees", number_of_referees)
st.metric("Total Number of Coaches", number_of_coaches)


st.table(df_education[mask])



## -------- Group Data by selection


