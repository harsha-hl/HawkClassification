#!pip install gradio
import numpy as np
import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Getting dependent and independent variables from dataframe

hawks = pd.read_csv('Hawks.csv')
X = hawks.iloc[:,1:].values
y = hawks.iloc[:,0].values


# Encoding data in x and y

le = LabelEncoder()
X[:,0]= le.fit_transform(X[:,0])
y=le.fit_transform(y)


# Splitting data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 1)


# Replacing missing values

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
X_train[:,1:] = imputer.fit_transform(X_train[:,1:])
X_test[:,1:] = imputer.transform(X_test[:,1:])


# Feature Scaling

sc = StandardScaler()
X_train[:,1:] = sc.fit_transform(X_train[:,1:])
X_test[:,1:] = sc.transform(X_test[:,1:])


# Using Logistic Regression 

log_model=LogisticRegression(penalty='l2')
log_model.fit(X_train,y_train)


def predictSpecies(age,wing,weight,culmen,hallux,tail):
    
    a = sc.transform([[wing,weight,culmen,hallux,tail]])
    i = np.array([[age,a[0][0],a[0][1],a[0][2],a[0][3],a[0][4]]])
    o = log_model.predict(i)
    s=o[0]
    ch = r'/home/harsha/Downloads/HAWKS_ML/CH.jpg'
    rt = r'/home/harsha/Downloads/HAWKS_ML/RT.jpg'
    ss = r'/home/harsha/Downloads/HAWKS_ML/SS.jpg'
    chj = r'/home/harsha/Downloads/HAWKS_ML/CHJ.jpg'
    rtj = r'/home/harsha/Downloads/HAWKS_ML/RTJ.jpg'
    ssj = r'/home/harsha/Downloads/HAWKS_ML/SSJ.jpg'

    if(s==0):
        if age==0:
            return ['\t    Cooper\'s Hawk !',ch]
        else:
            return ['\t    Cooper\'s Hawk !',chj]
        
    elif(s==1):
        if age==0:
            return ['\t    Red Tailed Hawk !',rt]
        else:
            return ['\t    Red Tailed Hawk !',rtj]
            
    else:
        if age==0:
            return ['\t  Sharp-shinned Hawk !',ss]
        else:
            return ['\t  Sharp-shinned Hawk !',ssj]
        
    


#Gradio web application    
    
age = gr.inputs.Radio(["Adult", "Juvenile"],type = "index",label='AGE')    
wing = gr.inputs.Slider(minimum=30,maximum=700,default=300,label='WING SPAN')
weight = gr.inputs.Slider(minimum=50,maximum=2500,default=500,label='HAWK\'S WEIGHT')
culmen = gr.inputs.Slider(minimum=5,maximum=50,default=10,label='CULMEN LENGTH')
hallux = gr.inputs.Slider(minimum=5,maximum=80,default=20,label='HALLUX SIZE')   
tail = gr.inputs.Slider(minimum=30,maximum=500,default=200,label='TAIL LENGTH')

sp = gr.outputs.Textbox(type = "auto",label = "\tSpecies")
im = gr.outputs.Image(type = "auto",label = " ")

gr.Interface(predictSpecies,inputs=[age,wing,weight,culmen,hallux,tail],outputs=[sp,im],title="HAWK SPECIES",live=False,interpretation="default",flagging_dir='flags').launch(share=True)
 
    
 




 

