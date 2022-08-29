import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
start="2010-01-01"
end="2021-12-31"
st.title("STOCK CRUSHER")
st.subheader("DATA FOR AAPL Stock")
st.write("AAPL is the stock name for the Company APPLE INC. is an American multinational technology company that specializes in consumer electronics, software and online services headquartered in Cupertino, California, United States.")
df= data.DataReader("AAPL","yahoo",start, end)
st.subheader("Data from 2010 to 2021")
st.write(df.describe())
st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 50MA")
m=df.Close.rolling(50).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(m)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 50MA and 150MA")
m=df.Close.rolling(50).mean()
m1=df.Close.rolling(150).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(m,"r")
plt.plot(m1,"g")
plt.plot(df.Close,"b")
st.pyplot(fig)

dtrain=pd.DataFrame(df["Close"][0:int(len(df)*0.50)])
dtest=pd.DataFrame(df["Close"][int(len(df)*0.50):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
s=MinMaxScaler(feature_range=(0,1))

d_train_array=s.fit_transform(dtrain)


model=load_model("Keras_Mod1")
phundred_days=dtrain.tail(100)
final_df=phundred_days.append(dtest,ignore_index=True)

input_data=s.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)

y_pre=model.predict(x_test)
scaler=s.scale_
s_factor=1/scaler[0]
y_pre=y_pre*s_factor
y_test=y_test*s_factor

st.subheader("Pedictions vs Original Graph")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,"b",label="Original Price")
plt.plot(y_pre,"g",label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig2)
