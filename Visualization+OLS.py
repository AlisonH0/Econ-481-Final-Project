#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def import_data() -> pd.DataFrame:
    sheet_list = [
      'CO2 Emissions from Energy',
      'Oil Consumption - EJ',
      'Gas Consumption - EJ',
      'Coal Consumption - EJ'
               ]

    combined_df = download_individual_sheet(sheet_list[0])
    for i in range(1, len(sheet_list)):
        individual_df = download_individual_sheet(sheet_list[i])
        combined_df = pd.merge(combined_df, individual_df)
    return combined_df


def download_individual_sheet(sheet: str) -> pd.DataFrame:
    path = '/home/jovyan/Econ-481-Final-Project/Statistical Review of World Energy Data.xlsx'
    data = pd.read_excel(path, sheet_name = sheet, skiprows = 2, usecols = range(59))

    data.rename({data.columns[0]: 'Country'}, axis=1, inplace=True)
    data.dropna(inplace = True)

    data = data.loc[data[data.columns[0]].isin(['Mexico', 'US', 'Canada'])]

    data = data.melt(id_vars = 'Country', var_name = 'Year', value_name = sheet)
    data = data.sort_values(by=['Country', 'Year'])

    return data


df = import_data()
print(df)


# In[2]:


us = df[df.Country == 'US']
tes1 = us["CO2 Emissions from Energy"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("CO2 Emissions from Energy from US")


# In[3]:


mex = df[df.Country == 'Mexico']
tes2 = mex["CO2 Emissions from Energy"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("CO2 Emissions from Energy from Mexico")


# In[4]:


can = df[df.Country == 'Canada']
tes3 = can["CO2 Emissions from Energy"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("CO2 Emissions from Energy from Canada")


# In[5]:


year = {}
for i in range(58):
    year[i] = df[df.Year == i + 1965]    
print(year[0])


# In[6]:


for i in range(58):
    year[i]["TotalCO2"] = sum(year[i]["CO2 Emissions from Energy"])
print(year[57])


# In[7]:


print(year[57]["TotalCO2"].values[0])


# In[8]:


df1 = [*range(58)]
for i in range(58):
    df1[i] = year[i]["TotalCO2"].values[0]
print(df1)


# In[9]:


df2 = [*range(1965, 2023, 1)]
print(df2)


# In[10]:


plt.plot(df2, df1)
plt.title("Total CO2 Emissions from North America")
plt.xlabel('Year')
plt.ylabel('Total CO2')
plt.show()


# In[11]:


usoil = df[df.Country == 'US']
tes1 = usoil["Oil Consumption - EJ"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Oil Consumption from US (EJ)")


# In[12]:


mexoil = df[df.Country == 'Mexico']
tes2 = mexoil["Oil Consumption - EJ"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Oil Consumption from Mexico (EJ)")


# In[13]:


canoil = df[df.Country == 'Canada']
tes3 = canoil["Oil Consumption - EJ"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Oil Consumption from Canada (EJ)")


# ## Coal Consumption

# In[14]:


uscoal = df[df.Country == 'US']
tes1 = us["Coal Consumption - EJ"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Coal Consumption from US (EJ)")


# In[15]:


cancoal = df[df.Country == 'Canada']
tes3 = cancoal["Coal Consumption - EJ"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Coal Consumption from Canada (EJ)")


# In[16]:


mexcoal = df[df.Country == 'Mexico']
tes2 = mexcoal["Coal Consumption - EJ"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Coal Consumption from Mexico (EJ)")


# ## Gas Consumption

# In[17]:


mexgas = df[df.Country == 'Mexico']
tes2 = mexgas["Gas Consumption - EJ"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Gas Consumption from Mexico")


# In[18]:


cangas = df[df.Country == 'Canada']
tes3 = cangas["Gas Consumption - EJ"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Gas Consumption from Canada (EJ)")


# In[19]:


usgas = df[df.Country == 'US']
tes1 = usgas["Gas Consumption - EJ"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Gas Consumption from US (EJ)")


# ## Total Oil

# In[20]:


for i in range(58):
    year[i]["TotalOil"] = sum(year[i]["Oil Consumption - EJ"])
print(year[57])


# In[21]:


df3 = [*range(58)]
for i in range(58):
    df3[i] = year[i]["TotalOil"].values[0]
print(df3)


# In[22]:


plt.plot(df2, df3)
plt.title("Total Oil Consumption from North America")
plt.xlabel('Year')
plt.ylabel('Total Oil Consumption (EJ)')
plt.show()


# ## Total Coal

# In[23]:


for i in range(58):
    year[i]["TotalCoal"] = sum(year[i]["Coal Consumption - EJ"])
print(year[57])


# In[24]:


df4 = [*range(58)]
for i in range(58):
    df4[i] = year[i]["TotalCoal"].values[0]


# In[25]:


plt.plot(df2, df4)
plt.title("Total Coal Consumption from North America")
plt.xlabel('Year')
plt.ylabel('Total Coal Consumption (EJ)')
plt.show()


# ## Gas Consumption

# In[26]:


for i in range(58):
    year[i]["TotalGas"] = sum(year[i]["Gas Consumption - EJ"])


# In[27]:


df5 = [*range(58)]
for i in range(58):
    df5[i] = year[i]["TotalGas"].values[0]


# In[28]:


plt.plot(df2, df5)
plt.title("Total Gas Consumption from North America")
plt.xlabel('Year')
plt.ylabel('Total Gas Consumption (EJ)')
plt.show()


# ## OLS

# In[29]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def ols(df: pd.DataFrame):
    # Rename columns for easier reference
    df = df.rename(columns={
        'CO2 Emissions from Energy':'CO2_emissions', 
        'Oil Consumption - EJ':'oil_consumption', 
        'Gas Consumption - EJ':'gas_consumption', 
        'Coal Consumption - EJ':'coal_consumption'
    })

    countries = ["Canada" , "US", "Mexico"]

    for i in countries:
        subset = df[df["Country"] == i]
        subset_train, subset_test = train_test_split(
            subset,
            test_size=0.2,
            random_state=12
        )

        ols_model = smf.ols('CO2_emissions ~ oil_consumption + gas_consumption + coal_consumption', data=subset_train).fit()
        print(f"OLS Summary for {i}:")
        print(ols_model.summary())

        def normalized_rmse(y_true, y_pred):
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            std_dev = np.std(y_true)
            nrmse = rmse / std_dev
            return nrmse

        pred = ols_model.predict(subset_test)
        actual = subset_test['CO2_emissions']

        # Plotting the actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, pred, edgecolors=(0, 0, 0))
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted CO2 Emissions for {i}')
        plt.show()

# Example usage
# df = pd.read_csv('your_data.csv')
# ols(df)
ols(df)


# In[ ]:




