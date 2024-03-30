import numpy as np
import matplotlib.pyplot as plt

clf_data = np.genfromtxt("CLF_data_1.csv", delimiter=",", skip_header=1, usecols=5)
dal_data = np.genfromtxt("DAL_data_1.csv", delimiter=",", skip_header=1, usecols=5)

clf_min = np.min(clf_data)
clf_max = np.max(clf_data)

clf_data = (clf_data - clf_min) / (clf_max - clf_min)

dal_min = np.min(dal_data)
dal_max = np.max(dal_data)

dal_data = (dal_data - dal_min) / (dal_max - dal_min)

diff_data = clf_data - dal_data[:-1]
total_data = diff_data
diff_data = diff_data[:-17]

interval = 100
start = 0
end = 100

monthly_data = []
weekly_data = []

while(end+10 <= 2500):
    tmp_month = diff_data[start:end]
    tmp_weekly = diff_data[end:end+10]
    monthly_data.append(tmp_month)
    weekly_data.append(tmp_weekly)
    start = start + 10
    end = end + 10

monthly_data = np.array(monthly_data)
weekly_data = np.array(weekly_data)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size) 
    
    def forward(self, x, hidden):
        out, hidden = self.gru(x)
        out = self.fc(out[-1,:])

        return out, hidden

hidden_size = 256
learning_rate = 0.0001
num_epochs = 10
input_size = 100
output_size = 10
layer_size = 8
batch_size = 32

monthly_tensor = torch.from_numpy(monthly_data)
weekly_tensor = torch.from_numpy(weekly_data)

dataset = TensorDataset(monthly_tensor, weekly_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = Model(input_size, output_size, hidden_size, layer_size)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    hidden = torch.zeros(layer_size, batch_size, hidden_size).float().detach()
    for month, week in zip(monthly_tensor, weekly_tensor):
        month = month.unsqueeze(0)
        optimizer.zero_grad()
        output, hidden = model(month.float(), hidden)
        loss = criterion(output, week.float())
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

last_month_data = total_data[2410:2510]
last_month_tensor = torch.from_numpy(last_month_data)
model.eval()
with torch.no_grad():
    last_month_tensor = last_month_tensor.unsqueeze(0)
    hidden = torch.zeros(layer_size, batch_size, hidden_size).float().detach()
    fin, hidden = model(last_month_tensor.float(), hidden)
    
    index1 = np.arange(len(total_data))
    plt.plot(index1, total_data, color="blue")
    index2 = np.arange(len(fin))
    index2 = index2 + 2510
    plt.plot(index2, fin, color="red")
    plt.show()
