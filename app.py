import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ma'lumotni yuklash
@st.cache_data
def load_data():
    urls = [
        "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2015-16/eng.1.csv",
        "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2016-17/eng.1.csv",
        "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2017-18/eng.1.csv",
        "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2018-19/eng.1.csv",
        "https://raw.githubusercontent.com/footballcsv/england/master/2010s/2019-20/eng.1.csv",
    ]
    frames = [pd.read_csv(u) for u in urls]
    df = pd.concat(frames, ignore_index=True)
    df[['home_goals','away_goals']] = df['FT'].str.split('-', expand=True).astype(int)
    def natija(row):
        if row['home_goals'] > row['away_goals']: return 'H'
        elif row['home_goals'] < row['away_goals']: return 'A'
        else: return 'D'
    df['result'] = df.apply(natija, axis=1)
    return df

df = load_data()

le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team'] = le_home.fit_transform(df['Team 1'])
df['away_team'] = le_away.fit_transform(df['Team 2'])
df['round_num'] = df['Round']

X = df[['home_team','away_team','round_num']]
y = df['result']

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X, y)

# Interfeys
st.title("⚽ Futbol O'yini Bashorati")
st.write("Jamoalarni tanlang va natijani biling!")

jamoalar = sorted(df['Team 1'].unique().tolist())

uy = st.selectbox("🏠 Uy jamoasi", jamoalar)
mehmon = st.selectbox("✈️ Mehmon jamoasi", jamoalar)
davr = st.slider("Davr", 1, 38, 15)

if st.button("Bashorat qil!"):
    uy_r = le_home.transform([uy])[0]
    meh_r = le_away.transform([mehmon])[0]
    pred = model.predict([[uy_r, meh_r, davr]])[0]
    if pred == 'H':
        st.success(f"🏠 {uy} g'alaba qiladi!")
    elif pred == 'A':
        st.error(f"✈️ {mehmon} g'alaba qiladi!")
    else:
        st.warning("🤝 Durang bo'ladi!")
