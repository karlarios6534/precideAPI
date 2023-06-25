from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
from fastapi.encoders import jsonable_encoder
#manipulacion de datos
import pandas as pd
from sklearn.preprocessing import StandardScaler
#visializacion de datos
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
app = FastAPI()

@app.get("/{variables}")
async def hola(variables:str):
    try:
        catched = variables
        variables=variables.replace('{','').replace('}','')
        variables = list(map(float, variables.split("%")))
        new_variables = []
        indexes=[0,3,6,9,12,15,18,21,24,27,1,4,7,10,13,16,19,22,25,28,2,5,8,11,14,17,20,23,26,29]
        new_variables=[]
        for idx in indexes:
            new_variables.append(variables[idx])
        variables=new_variables
        pd.set_option('display.max_columns',None)
        df = pd.read_csv('C:/Users/Damian Wayne/Desktop/pm/data.csv')

        #eliminacion de columnas que no necesitaremos
        df=df.drop(['id' , 'Unnamed: 32'] , axis=1)

        #transformar la clase en variable numerica
        le = LabelEncoder()

        df['diagnosis'] = le.fit_transform(df['diagnosis'])

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

        LR = LogisticRegression()

        #entrenar al modelo 
        LR.fit(X_train,y_train)

        #predicciones
        Y_LR = LR.predict(X_test)
        #[18.49,17.52,121.3,1068,0.1012,0.1317,0.1491,0.09183,0.1832,0.06697,0.7923,1.045,4.851,95.77,0.007974,0.03214,0.04435,0.01573,0.01617,0.005255,22.75,22.88,146.4,1600,0.1412,0.3089,0.3533,0.1663,0.251,0.09445]
        pacientenuevo = pd.DataFrame({"num1":variables})

        resultado = LR.predict(pacientenuevo.T)

        y_pred_proba = LR.predict_proba(pacientenuevo.T)

        res = resultado.item()
        # Convertir el elemento en un entero
        res = int(res)
        # Obtener el porcentaje de probabilidad de la veracidad del resultado
        if res == 1:
            veracidad_probabilidad = y_pred_proba[0][1]  # Porcentaje de probabilidad de la clase positiva
            res='Maligno'
        elif res == 0:
            veracidad_probabilidad = y_pred_proba[0][0] # Porcentaje de probabilidad de la clase negativa
            res='Benigno'
        else:
            print('Error')

        resultado = str(LR.predict(pacientenuevo.T))

        data = {'resultado': res, 'porcentaje':veracidad_probabilidad}
        response = jsonable_encoder(data)
        return response
    except Exception as e:
        data = {'exeption': str(e), 'variables':catched}
        response = jsonable_encoder(data)
        return response
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)