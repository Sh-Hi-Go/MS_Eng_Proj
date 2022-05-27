import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymongo
from pymongo import MongoClient
import sklearn
from datetime import date,timedelta
import random
from sklearn.model_selection import train_test_split
from fbprophet import Prophet

client = pymongo.MongoClient("mongodb+srv://automobile_123:automobile_456@cluster0.itzcp.mongodb.net/?retryWrites=true&w=majority")
db = client.test

def get_db(db,coll):
    database = client[db]
    collection = database[coll]
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    
    return df

car_dataset = get_db("cars","car_details")

customer_dataset = get_db("customer","customer_details")
count_company = get_db("customer","count_company")
count_body = get_db("customer","count_body")
count_fuel = get_db("customer","count_fuel")
count_transmission = get_db("customer","count_transmission")

sales_dataset = get_db("sales","sales_details")
sales_dataset.drop(["_id"],axis=1,inplace=True)

covid_effect = get_db("sales","covid_effect")
covid_effect.drop(["_id"],axis=1,inplace=True)

sales_dataset.dtypes

covid_effect.dtypes

"""### Price of a car"""

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def predict_price(make,model,variant):
  data = car_dataset.copy()
  data.dropna(inplace=True)

  # scaling numerical values
  num_features = [feature for feature in data.columns if (data[feature].dtypes!='O') and (feature not in ['Ex-Showroom_Price','Launch_Date'])]
  scaler = StandardScaler()
  data[num_features] = scaler.fit_transform(data[num_features])

  # enumerating categorical features
  categorical_features=[feature for feature in data.columns if (data[feature].dtype=='O') and (feature not in ['Body_Type','Fuel_Type','Type','_id','Make','Model','Variant','Launch_Date'])]
  cat_to_num = pd.get_dummies(categorical_features,drop_first=True)

  # train datasets
  X_train = pd.concat([data[num_features],cat_to_num], axis = 1)
  print(X_train)
  y_train = data['Ex-Showroom_Price']
  print(y_train)

  # test dataset = row of the car model
  X_test = data[(data['Make']==make) & (data['Model']==model) & (data['Variant']==variant)]
  X_test.drop(['Ex-Showroom_Price'],axis=1,inplace=True)

  # using lasso regression
  lasso = Lasso()
  params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
  folds = 5

  # cross validation
  model_cv = GridSearchCV(estimator = lasso, param_grid = params, scoring= 'neg_mean_absolute_error', cv = folds, return_train_score=True, verbose = 1)            
  model_cv.fit(X_train, y_train) 

  alpha =100000
  lasso = Lasso(alpha=alpha)     
  lasso.fit(X_train, y_train) 

  y_pred = lasso.predict(X_test)

  return y_pred


"""### Companies which give competition to a particular make with pie chart"""

def modulo(x):
  if x>=0:
    return x*3

  else:
    return (x + x*(-2))

def show_competition(mak,body,fuel,trans,seat,purpose):
  
  # finding models of similar car structure
  temp_df = car_dataset[(car_dataset['Body_Type']==str(body)) & (car_dataset['Fuel_Type']==str(fuel)) & (car_dataset['Type']==str(trans)) & (car_dataset['Seating_Capacity']==int(seat))]
  
  # creating 2 datasets for finding the correlation  
  df1 = pd.DataFrame().assign(Make = temp_df['Make'], Model = temp_df['Model'], Variant =temp_df['Variant'], A=temp_df['Competition'], 
                              B=temp_df['City_Mileage'], C=temp_df['Highway_Mileage'], D = temp_df['Minimum_Turning_Radius'])
  
  df2 = pd.DataFrame().assign(Make = temp_df['Make'], Model = temp_df['Model'], Variant =temp_df['Variant'], A=temp_df['Gears'], 
                              B=temp_df['Number_of_Airbags'], C=temp_df['Boot_Space'], D = temp_df['Ex-Showroom_Price'])


  df1['correlation'] = df1.corrwith(df2, axis = 1)

  # renaming columns to original and merging both the databases by sorting in descending order of correlation 
  df1.rename(columns = {'A':'Competition','B':'City_Mileage', 'C':'Highway_Mileage','D':'Minimum_Turning_Radius'}, inplace = True)
  df2.rename(columns = {'A':'Gears','B':'Number_of_Airbags', 'C':'Boot_Space','D':'Ex-Showroom_Price'}, inplace = True)
  
  final_data = pd.merge(df1,df2,on=['Make','Model','Variant'])
  final_data.sort_values("correlation",ascending=False,inplace=True)
  
  # using only make(company name) and correlation to plot a pie chart for market share/competition  
  plot_data = final_data.groupby('Make').mean()
  plot_data.reset_index(level=0, inplace=True)
  plot_data = plot_data[['Make','correlation']]
  plot_data['correlation']=plot_data.apply(lambda row : modulo(row['correlation']), axis = 1)
            
  fig = plt.figure()
  plot_data.plot(kind='pie', labels = plot_data['Make'],y='correlation', autopct='%1.0f%%')#,explode=tuple(explode))
  
  # returning the pie plot and final_data table
  if(purpose=='competition'): 
    return fig,final_data

  else:
    return fig,final_data.head(5)

fig,data = show_competition('Tata','Hatchback','Petrol','Manual',4,'customer')
data.head()

"""### Suggesting make of the car based on customer survey"""

def suggest_make(comp):
  # models made by the same company and the best model among them
  comp_models = car_dataset[car_dataset['Make']==comp]
  best_model = comp_models[comp_models['Competition']== comp_models['Competition'].max()]
  best_model.reset_index(level=0,inplace=True)

  # best category = highest probability
  body = count_body[count_body['Prob_buying'] == count_body['Prob_buying'].max()]
  body = body['Body_Type']
  fuel = count_fuel[count_fuel['Prob_buying'] == count_fuel['Prob_buying'].max()]
  fuel = fuel['Fuel_Type']
  transmission = count_transmission[count_transmission['Prob_buying'] == count_transmission['Prob_buying'].max()]
  transmission = transmission['transmission']

  # best company model which is in demand AND most desired make
  return best_model,body,fuel,transmission

best_model,body,fuel,transmission=suggest_make('Tata')

"""### Best car for customer based on their needs"""

def customer_suggest(mak,body,fuel,trans,seat):
  show_competition(mak,body,fuel,trans,seat,'customer')

customer_suggest('Toyota','Hatchback','Petrol','Manual',4)

"""### Sales and Launch time of a car using facebook prophet"""

# competition as additional regressor/ return launch date for launch prediction
def add_reg_comp(make,model,variant,purpose):
  row = car_dataset[(car_dataset['Make']==make) & (car_dataset['Model']==model) & (car_dataset['Variant']==variant)]
    
  if(purpose=='sales'):
    print(row['Competition'])
    return row['Competition']
    
  else:
    print(row['Launch_Date'])
    return row['Launch_Date']


def get_sale(make,model,variant,purpose):
    x= add_reg_comp(make,model,variant,'launch')
    y = add_reg_comp(make,model,variant,'sales')
    sale = sales_dataset.loc[(sales_dataset['ds']> x[0])]
    sale['competition'] = y[0]
    
    m = Prophet()
    m.add_regressor('competition')
    m.fit(sale)
    
    future = m.make_future_dataframe(periods=12, freq='M')
    future['competition'] = add_reg_comp(make,model,variant,'sales')
    
    forecast = m.predict(future)
    fig = m.plot_components(forecast)
    
    # for sales
    if(purpose=='sales'):
        forecast = forecast['ds','yhat','yhat_lower', 'yhat_upper']
    
    #for launch time
    else:
        forecast = forecast['ds','yhat','yhat_lower','yhat_upper','trend','trend_lower','trend_upper']
        forecast.sort_values("trend",ascending=False,inplace=True)

    # return the graph and forecast dataset    
    return fig,forecast


from flask import Flask,redirect,url_for,render_template,request

app = Flask(__name__)
app.run()

@app.route('/')
def welcome():
  return render_template('index.html')#include this file in a folder named 'templates'

@app.route('/submit',methods=['POST'])
def price_prediction():
  company=''
  model=''
  variant=''
  body=''
  fuel=''
  trans=''
  seats=0
  graph = None
  data = None
  comp_model= None
  best= None
  price=0

  choice = request.form['choice']

  if choice=='price':
    company = request.form['company']
    model = request.form['model']
    variant = request.form['variant']

    price = predict_price(company,model,variant)

  elif choice=='competition':
    company=request.form['company']
    body=request.form['body']
    fuel=request.form['fuel']
    trans=request.form['trans']
    seats=int(request.form['seats'])

    graph,data = show_competition(company,body,fuel,trans,seats)

  elif choice=='make':
    company=request.form['company']

    comp_model,best = suggest_make(company)

  elif choice=='launch':
    company = request.form['company']
    model = request.form['model']
    variant = request.form['variant']

    graph,data = get_sale(company,model,variant,'launch')

  elif choice=='sale':
    company = request.form['company']
    model = request.form['model']
    variant = request.form['variant']

    graph,data = get_sale(company,model,variant,'sales')

  else:
    company=request.form['company']
    body=request.form['body']
    fuel=request.form['fuel']
    trans=request.form['trans']
    seats=int(request.form['seats'])

    data = customer_suggest(company,body,fuel,trans,seats)

    return render_template('result.html')

if __name__ == '__main__' :
  app.run(debug=True)
