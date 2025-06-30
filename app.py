import gradio as gr
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

n = 100

data = {
    "area" : np.random.randint(500, 3500, n),
    "bedrooms": np.random.randint(1,5,n),
    "age":  np.random.randint(0,30,n),

}

df = pd.DataFrame(data)
df["price"] = (df["area"] * 150) + (df["bedrooms"] * 10000) - (df["age"] * 2000) + np.random.randint(-10000, 10000, n)


X = df[["area","bedrooms","age"]]
y = df["price"]

reg_model = LinearRegression()
reg_model.fit(X,y)

def predict_price(area, bedrooms, age):
    input_df = pd.DataFrame([[area, bedrooms, age]], columns = ["area","bedrooms","age"])
    prediction = reg_model.predict(input_df)[0]

    return f"${prediction:,.2f}"

house_price_app = gr.Interface(
    fn = predict_price,
    inputs = [
        gr.Number(label = "area(sq. ft.)"),
        gr.Slider(1,10, step = 1, label = "Bedrooms"),
        gr.Slider(0,30, step = 1, label = "age of house")
    ],
    outputs = "text"

)

house_price_app.launch(share= True)
