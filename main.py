from flask import Flask,render_template,request
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

app=Flask(__name__)
data=pd.read_csv("Bengaluru_House_Data_Cleaned_no_index.csv")

#ML MODEL:
X=data.drop(columns=["price"])
y=data["price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

column_trans=make_column_transformer((OneHotEncoder(sparse=False,),
                                      ["Availability","Locations"]),remainder='passthrough')

scaler=StandardScaler()

ridge=Ridge()
pipe2=make_pipeline(column_trans,scaler,ridge)

pipe2.fit(X_train,y_train)
y_pred_ridge=pipe2.predict(X_test)


@app.route("/")
def index():
    locations= sorted(data["Locations"].unique())
    baths=sorted(data["bath"].unique())
    availability=sorted(data["Availability"].unique())
    bhk=sorted(data["BHK"].unique())
    return render_template("index.html",locations=locations,
                           baths=baths,
                           availability=availability,
                           bhk=bhk)

@app.route("/predict",methods=["POST"])
def predict():
    locations_r=request.form.get('location')
    baths_r=request.form.get('bath')
    bhk_r=request.form.get('bhk')
    availability_r=request.form.get('availability')
    sqft_r=request.form.get('area')

    #print(locations_r,baths_r,bhk_r,availability_r,sqft_r)

    input=pd.DataFrame([[locations_r,float(baths_r),float(bhk_r),availability_r,float(sqft_r)]],
                       columns=['Locations','bath','BHK','Availability','total_sqft'])
    prediction=pipe2.predict(input)[0]

    output = data[(data['Locations'] == locations_r) & (data['BHK'] == float(bhk_r)) & (data['bath'] == float(baths_r))
                  & (data['Availability'] == availability_r) & (data["total_sqft"] >= float(sqft_r))]

    output["Ideal Market Price"]=prediction

    list0=["Area","Bath","Price (Lakhs)","Availability","BHK","Location","Market Price"]
    list1=output.values.tolist()
    list1.insert(0,list0)
    # row_count=output.shape[0]

    return list(list1)


if __name__ == "__main__":
    app.run(debug=True)