#manipulacion de datos
import pandas as pd
from sklearn.preprocessing import StandardScaler
#visializacion de datos
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns',None)
df = pd.read_csv('data.csv')
df.head()

#eliminacion de columnas que no necesitaremos
df=df.drop(['id' , 'Unnamed: 32'] , axis=1)

#transformar la clase en variable numerica

le = LabelEncoder()

df['diagnosis'] = le.fit_transform(df['diagnosis'])
df.head()


scaler = StandardScaler()

scaler.fit(df)
scaled_data = scaler.transform(df)



pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

#dividir datos en etiquetas
X = df.drop('diagnosis', axis = 1)
y = df['diagnosis']






#split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('Shape of train set ', X_train.shape)
print('Shape of test set ', X_test.shape)

LR = LogisticRegression()

#entrenar al modelo 
LR.fit(X_train,y_train)

#predicciones
Y_LR = LR.predict(X_test)

pacientenuevo = pd.DataFrame({"num1":[18.49,17.52,121.3,1068,0.1012,0.1317,0.1491,0.09183,0.1832,0.06697,0.7923,1.045,4.851,95.77,0.007974,0.03214,0.04435,0.01573,0.01617,0.005255,22.75,22.88,146.4,1600,0.1412,0.3089,0.3533,0.1663,0.251,0.09445]})
print(LR.predict(pacientenuevo.T))