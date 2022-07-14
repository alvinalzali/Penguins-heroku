import streamlit as st
import pandas as pd
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('NBA Player Stats Explorer')

st.markdown("""
This app is a tool to explore the **NBA** player stats using webscrapper.
* **Python libs :** base64, matplotlib, numpy, pandas, seaborn, streamlit
* **Data source :** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2023))))

# Web scrapper to get the data
@st.cache
def load_data(year):
    url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_game.html'
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) #remove the header row
    raw = raw.fillna(0) #fill the missing values with 0
    playerstats = raw.drop(['Rk'], axis=1) #remove the Rk column

    return playerstats

playerstats = load_data(selected_year)

#sidebar - Team Selection
sorted_unique_teams = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_teams, sorted_unique_teams)

#sidebar - Position Selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

#filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension :' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns')
st.dataframe(df_selected_team.astype(str))

#Download the data
# https://discuss.streamlit.io/t/how-to-download-a-file-from-a-streamlit-app/10

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download csv file</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

#Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Heatmap of Intercorrelation')
    df_selected_team.to_csv('playerstats.csv', index=False)
    df = pd.read_csv('playerstats.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(11, 9))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)
