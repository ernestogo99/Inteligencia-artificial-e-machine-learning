import torch
import torch.nn as nn
import pandas as pd
from model import Model


def celsius_to_farenheit(celsius:float)->float:
    return (1.8 * celsius +32)


temp_celsius=[-10,20,100]
temp_farenheit=[celsius_to_farenheit(temp) for temp in temp_celsius]

df =pd.DataFrame({'Celsius':temp_celsius,'Farenheit':temp_farenheit})
print(df)

x=torch.FloatTensor(df.Celsius.values.astype(float))
y=torch.FloatTensor(df.Farenheit.values.astype(float))
y=y.unsqueeze(1)


EPOCHS=1000
LR=0.2

model=Model()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=LR)


x=x.view(x.size(0),-1)


weights=[]
bias=[]

for epoch in range(EPOCHS):
    outputs = model.forward(x)

    loss=criterion(outputs,y)

    optimizer.zero_grad()

    loss.backward()
    
    optimizer.step()

    weights.append(model.input_layer.weight.item())
    bias.append(model.input_layer.bias.item())

print(f'Peso: {model.input_layer.weight.item():.2f} Bias: {model.input_layer.bias.item():.2f}  ' )