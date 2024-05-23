#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Downloads each sheet from sheet_list
#Merges cleaned data frames
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

'''
Takes a name of a sheet
Gets rid of null values
Subsets data for Mexico, US, and Canada
Sets country and year on the same axis
'''
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


# In[1]:


import pandas as pd

link = '/home/jovyan/Econ-481-Final-Project/Statistical Review of World Energy Data.xlsx'

#Downloads each cleaned sheet and merges into one dataframe
def import_data() -> pd.DataFrame:

    df1 = download_individual_sheet_oil()
    df2 = download_individual_sheet_gas()
    df3 = download_individual_sheet_coal()
    df4 = download_individual_sheet_CO2()
    df_list = [df1, df2, df3, df4]

    #Merges dataframes, with a common year index
    combined_df = pd.concat(df_list, axis=1)

    return combined_df

#Loads and cleans oil prices sheet
#Returns dataframe with a column of average of oil prices
#Null values in dataframe are due to the individual sheets covering different years
def download_individual_sheet_oil() -> pd.DataFrame:
    data = pd.read_excel(link, sheet_name = 'Oil - Spot crude prices', skiprows = 3, usecols = range(0, 5))

    data.rename({data.columns[0]: 'Year'}, axis=1, inplace=True)
    data = data.set_index(data.columns[0])
    data.drop(data.tail(4).index, inplace = True)

    data = data[data.index.notnull()]
    data.replace('-', None, inplace = True)

    #Creates column containing average of values
    data = data.mean(axis = 1, skipna = True).to_frame()
    data.rename({data.columns[0]: 'Avg Oil Prices'}, axis = 1, inplace = True)

    return data

#Loads and cleans gas price sheet
#Returns dataframe with a column of average of gas prices
def download_individual_sheet_gas() -> pd.DataFrame:
    data = pd.read_excel(link, sheet_name = 'Gas Prices ', skiprows = 4, usecols = range(0, 8))
    data.rename({data.columns[0]: 'Year'}, axis=1, inplace=True)
    data = data.set_index(data.columns[0])

    #drops unncessary rows
    data.drop(data.tail(8).index, inplace = True)

    data.replace('-', None, inplace = True)

    #Creates column containing average of values
    data = data.mean(axis = 1, skipna = True).to_frame()
    data.rename({data.columns[0]: 'Avg Gas Prices'}, axis = 1, inplace = True)

    return data

#Loads and cleans coal price sheet
#Returns dataframe with a column of average of coal prices
def download_individual_sheet_coal() -> pd.DataFrame:
    data = pd.read_excel(link, sheet_name = 'Coal Prices', skiprows = 2, usecols = range(0, 9))
    data.rename({data.columns[0]: 'Year'}, axis=1, inplace=True)
    data = data.set_index(data.columns[0])

    #drops unncessary rows
    data.drop(data.tail(8).index, inplace = True)

    data.replace('-', None, inplace = True)

    #Creates column containing average of values
    data = data.mean(axis = 1, skipna = True).to_frame()
    data.rename({data.columns[0]: 'Avg Coal Prices'}, axis = 1, inplace = True)


    return data

#Loads and cleans CO2 emissions sheet
#Returns column of average of CO2 emissions 
def download_individual_sheet_CO2() -> pd.DataFrame:
    data = pd.read_excel(link, sheet_name = 'CO2 Emissions from Energy', skiprows = 2, usecols = range(1, 59))
    data = data.iloc[[107]]

    data = data.transpose()
    data.index.name = 'Year'

    data.rename({data.columns[0]: 'Avg CO2 Emissions'}, axis=1, inplace=True)

    #Creates column containing average of values
    #92 is number of countries covered in original dataset
    data['Avg CO2 Emissions'] = data['Avg CO2 Emissions'].div(92)

    return data


df_prices = import_data()
print(df_prices)


# In[ ]:


us = df[df.Country == 'US']
tes1 = us["CO2 Emissions from Energy"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("CO2 Emissions from Energy from US")


# In[ ]:


mex = df[df.Country == 'Mexico']
tes2 = mex["CO2 Emissions from Energy"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("CO2 Emissions from Energy from Mexico")


# In[ ]:


can = df[df.Country == 'Canada']
tes3 = can["CO2 Emissions from Energy"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("CO2 Emissions from Energy from Canada")


# In[ ]:


year = {}
for i in range(58):
    year[i] = df[df.Year == i + 1965]    
print(year[0])


# In[ ]:


for i in range(58):
    year[i]["TotalCO2"] = sum(year[i]["CO2 Emissions from Energy"])
print(year[57])


# In[ ]:


print(year[57]["TotalCO2"].values[0])


# In[ ]:


df1 = [*range(58)]
for i in range(58):
    df1[i] = year[i]["TotalCO2"].values[0]
print(df1)


# In[ ]:


df2 = [*range(1965, 2023, 1)]
print(df2)


# In[ ]:


plt.plot(df2, df1)
plt.title("Total CO2 Emissions from North America")
plt.xlabel('Year')
plt.ylabel('Total CO2')
plt.show()


# In[ ]:


usoil = df[df.Country == 'US']
tes1 = usoil["Oil Consumption - EJ"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Oil Consumption from US (EJ)")


# In[ ]:


mexoil = df[df.Country == 'Mexico']
tes2 = mexoil["Oil Consumption - EJ"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Oil Consumption from Mexico (EJ)")


# In[ ]:


canoil = df[df.Country == 'Canada']
tes3 = canoil["Oil Consumption - EJ"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Oil Consumption from Canada (EJ)")


# ## Coal Consumption

# In[ ]:


uscoal = df[df.Country == 'US']
tes1 = us["Coal Consumption - EJ"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Coal Consumption from US (EJ)")


# In[ ]:


cancoal = df[df.Country == 'Canada']
tes3 = cancoal["Coal Consumption - EJ"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Coal Consumption from Canada (EJ)")


# In[ ]:


mexcoal = df[df.Country == 'Mexico']
tes2 = mexcoal["Coal Consumption - EJ"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Coal Consumption from Mexico (EJ)")


# ## Gas Consumption

# In[ ]:


mexgas = df[df.Country == 'Mexico']
tes2 = mexgas["Gas Consumption - EJ"]
fig, ax = plt.subplots()
tes2.plot(ax=ax, color="black")
plt.xticks(np.arange(58, 115, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Gas Consumption from Mexico")


# In[ ]:


cangas = df[df.Country == 'Canada']
tes3 = cangas["Gas Consumption - EJ"]
fig, ax = plt.subplots()
tes3.plot(ax=ax, color="black")
plt.xticks(np.arange(0, 57, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Gas Consumption from Canada (EJ)")


# In[ ]:


usgas = df[df.Country == 'US']
tes1 = usgas["Gas Consumption - EJ"]
fig, ax = plt.subplots()
tes1.plot(ax=ax, color="black")
plt.xticks(np.arange(116, 173, step=19), labels=[str(19*i+1965) for i in range(3)])
ax.set_title("Gas Consumption from US (EJ)")


# ## Total Oil

# In[ ]:


for i in range(58):
    year[i]["TotalOil"] = sum(year[i]["Oil Consumption - EJ"])
print(year[57])


# In[ ]:


df3 = [*range(58)]
for i in range(58):
    df3[i] = year[i]["TotalOil"].values[0]
print(df3)


# In[ ]:


plt.plot(df2, df3)
plt.title("Total Oil Consumption from North America")
plt.xlabel('Year')
plt.ylabel('Total Oil Consumption (EJ)')
plt.show()


# ## Total Coal

# In[ ]:


for i in range(58):
    year[i]["TotalCoal"] = sum(year[i]["Coal Consumption - EJ"])
print(year[57])


# In[ ]:


df4 = [*range(58)]
for i in range(58):
    df4[i] = year[i]["TotalCoal"].values[0]


# In[ ]:


plt.plot(df2, df4)
plt.title("Total Coal Consumption from North America")
plt.xlabel('Year')
plt.ylabel('Total Coal Consumption (EJ)')
plt.show()


# ## Gas Consumption

# In[ ]:


for i in range(58):
    year[i]["TotalGas"] = sum(year[i]["Gas Consumption - EJ"])


# In[ ]:


df5 = [*range(58)]
for i in range(58):
    df5[i] = year[i]["TotalGas"].values[0]


# In[ ]:


plt.plot(df2, df5)
plt.title("Total Gas Consumption from North America")
plt.xlabel('Year')
plt.ylabel('Total Gas Consumption (EJ)')
plt.show()


# ## OLS for Consumption

# In[ ]:


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


ols(df)


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def PricesReg(df_prices: pd.DataFrame):
  model = LinearRegression()
  df_prices = import_data()
  subset = df_prices[15:50]
  subset_train, subset_test = train_test_split(
        subset,
        test_size = 0.2,
        random_state = 12
    )
  predictors = ['Avg Oil Prices', 'Avg Gas Prices', 'Coal Prices']
  x_train = subset_train[predictors].to_numpy()
  x_test = subset_test[predictors].to_numpy()
  y_train = subset_train['Avg CO2 Emissions']
  model = model.fit(x_train, y_train)
  score = model.score(x_train, y_train)
  coef = model.coef_
  print(score)
  print(coef)

PricesReg(df_prices)

