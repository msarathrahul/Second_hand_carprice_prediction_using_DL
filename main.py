import sys
import pickle
import torch
from torch import nn
import pandas as pd
import warnings
import numpy

warnings.filterwarnings('ignore')

class LinearRegressionModel(nn.Module):
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int):
        super().__init__()

        self.deep_neural_network = nn.Sequential(nn.Linear(in_features=input_nodes, out_features=hidden_nodes),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=hidden_nodes, out_features=output_nodes))

    def forward(self, X):
        return self.deep_neural_network(X)

with open('_models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('_models/ct.pkl', 'rb') as f:
    ct = pickle.load(f)

args = sys.argv[1:]
columns = ['on_road_age', 'age', 'km', 'rating', 'condition', 'economy','top_speed', 'hp']
dic = {}
for col in columns:
    index = args.index(col) + 1
    dic[col] = int(args[index])
df = pd.DataFrame(data=dic,index = range(1))
df = df[columns].copy()
features = torch.tensor(ct.transform(df)).type(torch.float)
with torch.inference_mode():
    pred = model(features)
print(f"The price of car is : {pred.item()}")
