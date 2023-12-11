#-----------------------------------------------------------------Algiers---------------------------------------------------------------------------------------
import numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

df1 = pd.read_csv('Algiers.csv')

# Créer des visualisations pour explorer la distribution des caractéristiques
df1.hist(bins=20, figsize=(12, 8))
plt.show()

print(df1.isnull().sum())

df_Al=df1.loc[:,["DATE","PRCP","TAVG","TMAX","TMIN","STATION"]]
print(df_Al["PRCP"].value_counts())
print(df_Al["TAVG"].value_counts())
print(df_Al["TMAX"].value_counts())
print(df_Al["TMIN"].value_counts())


df_Al['DATE'] = pd.to_datetime(df_Al['DATE'])

df_Al['YEAR'], df_Al['MONTH'] = df_Al['DATE'].dt.year, df_Al['DATE'].dt.month

date = df_Al.DATE.dt.month*100 + df_Al.DATE.dt.day
df_Al['SEASON'] = (pd.cut(date,[0,321,620,922,1220,1300],labels=['winter','spring','summer','autumn','winter ']).str.strip())

#remplacer les val categorique du mois par des val nume
df_Al['SEASON'].replace(['winter','spring','summer','autumn'],[0,1,2,3], inplace=True)

df_Al=df_Al.drop("DATE",axis=1)

#val null
print(df_Al.isnull().sum())

print(df_Al.isnull().sum()/df_Al.shape[0])

#remplir chaque les nan par la moyenne de chaque annee du nan
df_Al["PRCP"] = df_Al["PRCP"].fillna(0)

df_Al=df_Al.fillna(df_Al.groupby('MONTH').transform('mean'))


df_Al['SEASON'] = df_Al['SEASON'].astype(np.int64)
df_Al['YEAR'] = df_Al['YEAR'].astype(np.int64)


#-----------------------------------------------------------------Constantine---------------------------------------------------------------------------------------


df2 = pd.read_csv('Constantine.csv')
# Créer des visualisations pour explorer la distribution des caractéristiques
df2.hist(bins=20, figsize=(12, 8))
plt.show()

print(df2.isnull().sum())

df_CO=df2.loc[:,["DATE","PRCP","TAVG","TMAX","TMIN","STATION"]]
print(df_CO["PRCP"].value_counts())
print(df_CO["TAVG"].value_counts())
print(df_CO["TMAX"].value_counts())
print(df_CO["TMIN"].value_counts())

df_CO['DATE'] = pd.to_datetime(df_CO['DATE'])

df_CO['YEAR'], df_CO['MONTH'] = df_CO['DATE'].dt.year, df_CO['DATE'].dt.month

date = df_CO.DATE.dt.month*100 + df_CO.DATE.dt.day
df_CO['SEASON'] = (pd.cut(date,[0,321,620,922,1220,1300],labels=['winter','spring','summer','autumn','winter ']).str.strip())

#remplacer les val categorique du mois par des val nume
df_CO['SEASON'].replace(['winter','spring','summer','autumn'], [0,1,2,3], inplace=True)

df_CO=df_CO.drop("DATE",axis=1)

#val nul
print(df_CO.isnull().sum())
print(df_CO.isnull().sum()/df_CO.shape[0])


#remplir chaque les nan par la moyenne de chaque annee du nan
df_CO["PRCP"] = df_CO["PRCP"].fillna(0)
df_CO=df_CO.combine_first(df_CO.groupby('MONTH').transform('mean'))
df_CO['SEASON'] = df_CO['SEASON'].astype(np.int64)
df_CO['YEAR'] = df_CO['YEAR'].astype(np.int64)


#--------------------------------------------ORAN----------------------------------------------------------------------------------------------------------

df3 = pd.read_csv('Oran.csv')
# Créer des visualisations pour explorer la distribution des caractéristiques
df3.hist(bins=20, figsize=(12, 8))
plt.show()
print(df3.isnull().sum())

df_OR=df3.loc[:,["DATE","PRCP","TAVG","TMAX","TMIN","STATION"]]
print(df_OR["PRCP"].value_counts())
print(df_OR["TAVG"].value_counts())
print(df_OR["TMAX"].value_counts())
print(df_OR["TMIN"].value_counts())

df_OR['DATE'] = pd.to_datetime(df_OR['DATE'])
df_OR['YEAR'], df_OR['MONTH'] = df_OR['DATE'].dt.year, df_OR['DATE'].dt.month

date = df_OR.DATE.dt.month*100 + df_OR.DATE.dt.day
df_OR['SEASON'] = (pd.cut(date,[0,321,620,922,1220,1300], labels=['winter','spring','summer','autumn','winter ']).str.strip())


#remplacer les val categorique du mois par des val nume
df_OR['SEASON'].replace(['winter','spring','summer','autumn'], [0,1,2,3], inplace=True)

df_OR=df_OR.drop("DATE",axis=1)

print(df_OR.isnull().sum())
print(df_OR.isnull().sum()/df_OR.shape[0])

#remplir chaque les nan par la moyenne de chaque annee du nan
df_OR["PRCP"] = df_OR["PRCP"].fillna(0)
df_OR=df_OR.combine_first(df_OR.groupby('MONTH').transform('mean'))
df_OR['SEASON'] = df_OR['SEASON'].astype(np.int64)
df_OR['YEAR'] = df_OR['YEAR'].astype(np.int64)


#----------------------------------------------------------------RABAT-----------------------------------------------------------------------------

df4 = pd.read_csv('Rabat.csv')
# Créer des visualisations pour explorer la distribution des caractéristiques
df4.hist(bins=20, figsize=(12, 8))
plt.show()

print(df4.isnull().sum())

df_RA=df4.loc[:,["DATE","PRCP","TAVG","TMAX","TMIN","STATION"]]
print(df_RA["PRCP"].value_counts())
print(df_RA["TAVG"].value_counts())
print(df_RA["TMAX"].value_counts())
print(df_RA["TMIN"].value_counts())

df_RA['DATE'] = pd.to_datetime(df_RA['DATE'])
df_RA['YEAR'], df_RA['MONTH'] = df_RA['DATE'].dt.year, df_RA['DATE'].dt.month

date = df_RA.DATE.dt.month*100 + df_RA.DATE.dt.day
df_RA['SEASON'] = (pd.cut(date,[0,321,620,922,1220,1300], labels=['winter','spring','summer','autumn','winter ']).str.strip())


#remplacer les val categorique du mois par des val nume
df_RA['SEASON'].replace(['winter','spring','summer','autumn'], [0,1,2,3], inplace=True)


df_RA=df_RA.drop("DATE",axis=1)

print(df_RA.isnull().sum())
print(df_RA.isnull().sum()/df_RA.shape[0])

#remplir chaque les nan par la moyenne de chaque annee du nan
df_RA["PRCP"] = df_RA["PRCP"].fillna(0)
df_RA=df_RA.combine_first(df_RA.groupby('MONTH').transform('mean'))
df_RA['SEASON'] = df_RA['SEASON'].astype(np.int64)
df_RA['YEAR'] = df_RA['YEAR'].astype(np.int64)


#-----------------------------------------------------------Tunis--------------------------------------------------------------------------------

df5= pd.read_csv('Tunis.csv')
# Créer des visualisations pour explorer la distribution des caractéristiques
df5.hist(bins=20, figsize=(12, 8))
plt.show()
print(df5.isnull().sum())

df_TU=df5.loc[:,["DATE","PRCP","TAVG","TMAX","TMIN","STATION"]]
print(df_TU["PRCP"].value_counts())
print(df_TU["TAVG"].value_counts())
print(df_TU["TMAX"].value_counts())
print(df_TU["TMIN"].value_counts())

df_TU['DATE'] = pd.to_datetime(df_TU['DATE'])
df_TU['YEAR'], df_TU['MONTH'] = df_TU['DATE'].dt.year, df_TU['DATE'].dt.month

date = df_TU.DATE.dt.month*100 + df_TU.DATE.dt.day
df_TU['SEASON'] = (pd.cut(date,[0,321,620,922,1220,1300], labels=['winter','spring','summer','autumn','winter ']).str.strip())

#remplacer les val categorique du mois par des val nume
df_TU['SEASON'].replace(['winter','spring','summer','autumn'], [0,1,2,3], inplace=True)


df_TU=df_TU.drop("DATE",axis=1)


print(df_TU.isnull().sum()/df_TU.shape[0])

#remplir chaque les nan par la moyenne de chaque annee du nan
df_TU["PRCP"] = df_TU["PRCP"].fillna(0)
df_TU=df_TU.combine_first(df_TU.groupby('MONTH').transform('mean'))

df_TU['SEASON'] = df_TU['SEASON'].astype(np.int64)
df_TU['YEAR'] = df_TU['YEAR'].astype(np.int64)
#---------------------------------------------------combined data---------------------------------------------------------------------------




df_ALL = pd.concat(
    map(pd.DataFrame, [df_Al, df_CO, df_OR,df_RA,df_TU]), ignore_index=True)
print(df_ALL)
print(df_ALL["PRCP"].value_counts())


dict = {
        "AGM00060369": 1,
        "AG000060390": 1,
        "AGM00060419": 2,
        "AGM00060490" : 3,
        "AGM00060461" :3,
        "AGM00060452": 3,
        "MOW00013017" : 4,
        "MOM00060135": 4,
        "TSM00060715": 5,
    }

df_ALL['STATION'].replace(dict, inplace=True)

# Visualisation de TAVG
plt.title('Evolution TAVG en courbe par annee')
df_ALL.groupby(["YEAR","STATION"]).mean()["TAVG"].unstack().plot.bar()
plt.title('Evolution TAVG en barres par annee')
df_ALL.groupby(["SEASON","STATION"]).mean()["TAVG"].unstack().plot.bar()
plt.title('Evolution TAVG en barres par saison')
df_ALL.groupby(["MONTH","STATION"]).mean()["TAVG"].unstack().plot.bar()
plt.title('Evolution TAVG en barres par mois')
# Visualisation de PRCP
plt.title('Evolution PRCP en courbe par annee')
df_ALL.groupby(["YEAR","STATION"]).mean()["PRCP"].unstack().plot()
plt.title('Evolution PRCP en barres par annee')
df_ALL.groupby(["YEAR","STATION"]).mean()["PRCP"].unstack().plot.bar()
plt.title('Evolution PRCP en barres par saison')
df_ALL.groupby(["SEASON","STATION"]).mean()["PRCP"].unstack().plot.bar()
plt.title('Evolution PRCP en barres par mois')
df_ALL.groupby(["MONTH","STATION"]).mean()["PRCP"].unstack().plot.bar()



#--------------Model pour predir par annee------------



a=[[2023], [2024], [2025], [2026], [2027], [2028], [2029],[2030]]
a2=[2023, 2024, 2025, 2026, 2027, 2028, 2029,2030]

X_YEAR = df_ALL[["YEAR"]].values
Y= df_ALL[["TAVG","PRCP"]].values

def model_year (df,delta):
    x = df[["YEAR"]].values
    y = df[["TAVG","PRCP"]].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    m = LinearRegression().fit(X_train,y_train)
    pred= np.squeeze(m.predict(delta))
    model_accuracy =m.score(X_test, y_test)
    print('model accuracy =', model_accuracy)
    print("-------------------------------------------------------")
    print("Resultat de la regression", pred)
    return pred

prediction_by_year=df_ALL.groupby('STATION').apply(model_year,delta=a)



ry = []
rp = []

for i in prediction_by_year.values:
    print("----------------------------------------------------")
    for j in i:
        ry.append(j[0])
        rp.append(j[1])

ry = np.atleast_2d(ry).T
rp = np.atleast_2d(rp).T

    
city= ["ALGERIES", "CONSTANTINE","ORAN", "RABAT", "TUNIS"]

tabc = []
for e1 in city:
    for e2 in a2:
        tabc.append((e1,e2))
ryear = pd.DataFrame(tabc)

ryear.rename(columns={0:'CITY', 1: 'YEAR'}, inplace=True)

ryear['TAVG']= ry
ryear['PRCP']= rp

#---------------Visualisation des predictions--------------------------------

plt.title('Prediction TAVG en barres par annee')
ryear.groupby(["YEAR","CITY"]).mean()["TAVG"].unstack().plot.bar()
plt.ylabel('TAVG')
plt.title('Prediction TAVG en courbe par annee')
ryear.groupby(["YEAR","CITY"]).mean()["TAVG"].unstack().plot()
plt.ylabel('TAVG')
plt.title('Prediction PRCP en barres par annee')
ryear.groupby(["YEAR","CITY"]).mean()["PRCP"].unstack().plot.bar()
plt.ylabel('PRCP')
plt.title('Prediction PRCP en courbe par annee')
ryear.groupby(["YEAR","CITY"]).mean()["PRCP"].unstack().plot()
plt.ylabel('PRCP')





# ----------------------Model pour predir par mois----------------

m=[1,2,3,4,5,6,7,8,9,10,11,12]

l1 = []
for e1 in a2:
    for e2 in m:
        l1.append((e1,e2))
annee_mois = pd.DataFrame(l1)



X_MONTH = df_ALL[["YEAR","MONTH"]].values
Y= df_ALL[["TAVG","PRCP"]].values

def model_month (df,delta):
    x = df[["YEAR","MONTH"]].values
    y = df[["TAVG","PRCP"]].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    m = LinearRegression()
    m.fit(X_train, y_train)
    pred = np.squeeze(m.predict(delta))
    model_accuracy = m.score(X_test, y_test)
    print('model accuracy =', model_accuracy)
    return pred
prediction_by_month=df_ALL.groupby('STATION').apply(model_month,delta=annee_mois)

ry = []
rp = []

for i in prediction_by_month.values:
    print("----------------------------------------------------")
    for j in i:
        ry.append(j[0])
        rp.append(j[1])

ry = np.atleast_2d(ry).T
rp = np.atleast_2d(rp).T



tabc = []
for e1 in city:
    for e2 in a2:
        for e3 in m:
            tabc.append((e1,e2,e3))

rmonth = pd.DataFrame(tabc)

rmonth.rename(columns={0: 'CITY', 1: 'YEAR', 2: 'MONTH'}, inplace=True)
rmonth['TAVG']= ry
rmonth['PRCP']= rp

#---------------Visualisation des predictions-------------------------------------------
rmonth.groupby(["MONTH","CITY"]).mean()["TAVG"].unstack().plot.bar()
plt.title('Prediction TAVG en barres par mois')
plt.ylabel('TAVG')
plt.title('Prediction PRCP en barres par mois')
rmonth.groupby(["MONTH","CITY"]).mean()["PRCP"].unstack().plot.bar()
plt.ylabel('PRCP')



#--------------------Model pour predir par saison-----------------------

s=[0,1,2,3]

l2 = []
for e1 in a2:
    for e2 in s:
        l2.append((e1,e2))

annee_saison = pd.DataFrame(l2)
annee = pd.DataFrame(a2)

X_SESON = df_ALL[["YEAR","SEASON"]].values
Y= df_ALL[["TAVG","PRCP"]].values

def model_season (df,delta):
    x = df[["YEAR","SEASON"]].values
    y = df[["TAVG","PRCP"]].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    m = LinearRegression().fit(X_train, y_train)
    pred = np.squeeze(m.predict(delta))
    model_accuracy = m.score(X_test, y_test)
    print('model accuracy =', model_accuracy)
    return pred
prediction_by_season=df_ALL.groupby('STATION').apply(model_season,delta=annee_saison)

ry = []
rp = []

for i in prediction_by_season.values:
    print("----------------------------------------------------")
    for j in i:
        ry.append(j[0])
        rp.append(j[1])

ry = np.atleast_2d(ry).T
rp = np.atleast_2d(rp).T


tabc = []
for e1 in city:
    for e2 in a2:
        for e3 in s:
            tabc.append((e1,e2,e3))

rseason = pd.DataFrame(tabc)
rseason.rename(columns={0:'CITY', 1: 'YEAR', 2: 'SEASON'}, inplace=True)
rseason['TAVG']= ry
rseason['PRCP']= rp

#---------------Visualisation des predictions----------------------------------------------

plt.title('Prediction TAVG en barres par saison')
rseason.groupby(["SEASON","CITY"]).mean()["TAVG"].unstack().plot.bar()
plt.ylabel('TAVG')
plt.title('Prediction PRCP en barres par saison')
rseason.groupby(["SEASON","CITY"]).mean()["PRCP"].unstack().plot.bar()
plt.ylabel('PRCP')

#---------------Random forest----------------------------------------------

#------------TAVG

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Suppose df_ALL contient les données combinées et les colonnes 'YEAR' et 'TAVG'
X = df_ALL[['YEAR']]
y_tavg = df_ALL['TAVG']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_tavg, test_size=0.2, random_state=42)

# Création du modèle RandomForestRegressor
rf_tavg = RandomForestRegressor(n_estimators=100, random_state=42)

# Adapter le modèle
rf_tavg.fit(X_train, y_train)

# Prédictions
y_pred_tavg = rf_tavg.predict(X_test)

# Évaluation du modèle
mse_tavg = mean_squared_error(y_test, y_pred_tavg)
r2_tavg = r2_score(y_test, y_pred_tavg)
print(f"TAVG Mean Squared Error: {mse_tavg}")
print(f"TAVG R-squared: {r2_tavg}")

# Plot des prédictions par rapport aux valeurs réelles
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_tavg, alpha=0.5)
plt.xlabel('Valeurs Réelles (y_test)')
plt.ylabel('Valeurs Prédites (y_pred_tavg)')
plt.title('Prédictions TAVG avec Random Forest Regressor')
plt.show()


#------------PRCP

# Supposons que df_ALL contient les données combinées et les colonnes 'YEAR' et 'PRCP'
X = df_ALL[['YEAR']]
y_prcp = df_ALL['PRCP']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_prcp, test_size=0.2, random_state=42)

# Création du modèle RandomForestRegressor pour PRCP
rf_prcp = RandomForestRegressor(n_estimators=100, random_state=42)

# Adapter le modèle
rf_prcp.fit(X_train, y_train)

# Prédictions
y_pred_prcp = rf_prcp.predict(X_test)

# Évaluation du modèle
mse_prcp = mean_squared_error(y_test, y_pred_prcp)
r2_prcp = r2_score(y_test, y_pred_prcp)
print(f"PRCP Mean Squared Error: {mse_prcp}")
print(f"PRCP R-squared: {r2_prcp}")

# Plot des prédictions par rapport aux valeurs réelles
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_prcp, alpha=0.5)
plt.xlabel('Valeurs Réelles (y_test)')
plt.ylabel('Valeurs Prédites (y_pred_prcp)')
plt.title('Prédictions PRCP avec Random Forest Regressor')
plt.show()

#---------------RNN------------------------------------------------------------------
#-------------#--------TAVG-------------#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# Supposons que df_ALL contient les données combinées et les colonnes 'YEAR', 'MONTH', 'TMAX', 'TMIN', 'TAVG'
features = ['YEAR', 'MONTH', 'TMAX', 'TMIN']
target = 'TAVG'

# Sélection des caractéristiques et de la cible
X = df_ALL[features].values
y = df_ALL[target].values

# Normalisation des données
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Préparation des séquences pour le RNN
SEQUENCE_LENGTH = 12  # Longueur de la séquence pour chaque échantillon

X_seq, y_seq = [], []
for i in range(len(X) - SEQUENCE_LENGTH):
    X_seq.append(X[i:i + SEQUENCE_LENGTH])
    y_seq.append(y[i + SEQUENCE_LENGTH])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Création du modèle RNN
model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(SEQUENCE_LENGTH, len(features))))
model.add(Dense(1))  # Couche de sortie avec une seule sortie pour la prédiction de TAVG

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# Prédiction sur l'ensemble de test
predicted_values = model.predict(X_test)

# Inverser la normalisation pour obtenir les vraies valeurs
predicted_values = scaler.inverse_transform(predicted_values)
y_test = scaler.inverse_transform(y_test)

# Plot des valeurs prédites par rapport aux valeurs réelles
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Valeurs Réelles')
plt.plot(predicted_values, label='Valeurs Prédites')
plt.xlabel('jours')
plt.ylabel('TAVG')
plt.title('Prédictions RNN vs Réel pour TAVG')
plt.legend()
plt.show()

#-------------#--------RNN PRCP-------------#

# il n'est pas adapté

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# Supposons que df_ALL contient les données combinées et les colonnes 'YEAR', 'MONTH', 'TMAX', 'TMIN', 'PRCP'
features = ['YEAR', 'MONTH', 'TMAX', 'TMIN']
target = 'PRCP'

# Sélection des caractéristiques et de la cible
X = df_ALL[features].values
y = df_ALL[target].values

# Normalisation des données
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Préparation des séquences pour le RNN
SEQUENCE_LENGTH = 12  # Longueur de la séquence pour chaque échantillon

X_seq, y_seq = [], []
for i in range(len(X) - SEQUENCE_LENGTH):
    X_seq.append(X[i:i + SEQUENCE_LENGTH])
    y_seq.append(y[i + SEQUENCE_LENGTH])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Création du modèle RNN
model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(SEQUENCE_LENGTH, len(features))))
model.add(Dense(1))  # Couche de sortie avec une seule sortie pour la prédiction de PRCP

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# Prédiction sur l'ensemble de test
predicted_values = model.predict(X_test)

# Inverser la normalisation pour obtenir les vraies valeurs
predicted_values = scaler.inverse_transform(predicted_values)
y_test = scaler.inverse_transform(y_test)

# Plot des valeurs prédites par rapport aux valeurs réelles
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Valeurs Réelles')
plt.plot(predicted_values, label='Valeurs Prédites')
plt.xlabel('Index')
plt.ylabel('PRCP')
plt.title('Prédictions RNN vs Réel pour PRCP')
plt.legend()
plt.show()

#-------------------------------LSTM Algorithme TAVG----------------------

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Sélection des données de température TAVG pour une ville spécifique (par exemple, Algeries)
selected_city_data = df_ALL[df_ALL['STATION'] == 1]['TAVG'].values

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
selected_city_data = selected_city_data.reshape(-1, 1)
scaled_data = scaler.fit_transform(selected_city_data)

# Création des séquences temporelles
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

sequence_length = 10  # Longueur de la séquence temporelle
X, y = create_sequences(scaled_data, sequence_length)

# Division des données en ensembles d'entraînement et de test
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Construction du modèle LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Prédictions sur les données de test
predicted_values = model.predict(X_test)
predicted_values = scaler.inverse_transform(predicted_values)

# Calcul de l'erreur (MSE)
mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predicted_values)
print('Mean Squared Error:', mse)

# Plot des prédictions et des vraies valeurs
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='True Values')
plt.plot(predicted_values, label='Predicted Values')
plt.ylabel("TAVG")
plt.legend()
plt.show()


#-------------------------RNN TAVG-------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Charger les données
# Assurez-vous d'avoir vos données chargées dans des DataFrame df_Al, df_CO, df_OR, df_RA, df_TU

# Concaténer les DataFrames en un seul
df_ALL = pd.concat([df_Al, df_CO, df_OR, df_RA, df_TU])

# Choix de la colonne cible (TAVG) et des caractéristiques
features = ['YEAR', 'MONTH', 'TMAX', 'TMIN']
target = 'TAVG'

# Séparation des données en features (X) et target (y)
X = df_ALL[features].values
y = df_ALL[target].values

# Normalisation des données
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Préparation des séquences pour le RNN
SEQUENCE_LENGTH = 12  # Longueur de la séquence pour chaque échantillon

X_seq, y_seq = [], []
for i in range(len(X) - SEQUENCE_LENGTH):
    X_seq.append(X[i:i + SEQUENCE_LENGTH])
    y_seq.append(y[i + SEQUENCE_LENGTH])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Création du modèle RNN
model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(SEQUENCE_LENGTH, len(features))))
model.add(Dense(1))  # Couche de sortie avec une seule sortie pour la prédiction de TAVG

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# Prédiction sur l'ensemble de test
predicted_values = model.predict(X_test)

# Inverser la normalisation pour obtenir les vraies valeurs
predicted_values = scaler.inverse_transform(predicted_values)
y_test = scaler.inverse_transform(y_test)
years_test = df_ALL['YEAR'].values[SEQUENCE_LENGTH:]
# Plot des valeurs prédites par rapport aux valeurs réelles
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Valeurs Réelles')
plt.plot(predicted_values, label='Valeurs Prédites')
plt.xlabel('nombre de jour')
plt.ylabel('TAVG')
plt.title('Prédictions RNN vs Réel')
plt.legend()
plt.show()

#------------------SVM Algorithme----------------------
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Utilisation des données existantes
X = df_ALL[["YEAR"]].values
y = df_ALL["TAVG"].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle SVM
svm_model = SVR(kernel='linear')

# Entraînement du modèle
svm_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
predicted_TAVG_svm = svm_model.predict(X_test)

# Affichage du plot
plt.figure(figsize=(20, 6))
plt.scatter(X_test, y_test, color='black', label='Données réelles')
plt.scatter(X_test, predicted_TAVG_svm, color='red', label='Prédictions SVM')
plt.xlabel('Année')
plt.ylabel('Température moyenne (TAVG)')
plt.title('Prédictions de température moyenne avec SVM')
plt.legend()
plt.show()
# Afficher quelques-unes des valeurs prédites et réelles
print("Quelques valeurs prédites :", predicted_TAVG_svm[:10])
print("Quelques valeurs réelles :", y_test[:10])

