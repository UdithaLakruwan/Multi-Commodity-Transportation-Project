
from pulp import *
import pandas as pd
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# File paths
toffee_costs_path = '/Users/senadhinipun/Documents/University/Project/EXcel data files/Toffee_Distribution_Costs.xlsx'
chocolate_costs_path = '/Users/senadhinipun/Documents/University/Project/EXcel data files/Chocolate_Distribution_Costs.xlsx'
biscuit_costs_path = '/Users/senadhinipun/Documents/University/Project/EXcel data files/Biscuit_Distribution_Costs.xlsx'
plant_capacity_path = '/Users/senadhinipun/Documents/University/Project/EXcel data files/Plant_Capacity_for_Commodities.xlsx'
demand_path = '/Users/senadhinipun/Documents/University/Project/EXcel data files/Commodity_Demand.xlsx'
fixed_cost_path = '/Users/senadhinipun/Documents/University/Project/EXcel data files/Fixed_Cost_Distribution.xlsx'
distribution_capacity_path = '/Users/senadhinipun/Documents/University/Project/EXcel data files/Capacity_Commodity_Distribution.xlsx'

# Load data
toffee_costs = pd.read_excel(toffee_costs_path, index_col='Customer')
chocolate_costs = pd.read_excel(chocolate_costs_path, index_col='Customer')
biscuit_costs = pd.read_excel(biscuit_costs_path, index_col='Customer')
plant_capacity = pd.read_excel(plant_capacity_path)
demand = pd.read_excel(demand_path)
fixed_costs = pd.read_excel(fixed_cost_path)
distribution_capacity = pd.read_excel(distribution_capacity_path)

# Data preprocessing
customers = list(demand['Customer'])
distribution_centers = list(fixed_costs['Distribution Center'])

commodities = ['Toffee', 'Chocolate', 'Biscuit']
cost_data = {'Toffee': toffee_costs, 'Chocolate': chocolate_costs, 'Biscuit': biscuit_costs}

# Extract relevant capacity data for each commodity at distribution centers
capacity_data = {
    'Toffee': distribution_capacity[['Distribution Center', 'Toffee']],
    'Chocolate': distribution_capacity[['Distribution Center', 'Chocolate']],
    'Biscuit': distribution_capacity[['Distribution Center', 'Biscuit']]
}

# Extract demand for each customer and commodity
demand_data = {
    'Toffee': demand[['Customer', 'Toffee']],
    'Chocolate': demand[['Customer', 'Chocolate']],
    'Biscuit': demand[['Customer', 'Biscuit']]
}

# Create optimization problem
model = LpProblem("Multi-Commodity_Transportation_Problem", LpMinimize)

# Decision variables
x = LpVariable.dicts("shipment", ((i, j, k) for i in commodities for j in distribution_centers for k in customers), lowBound=0, cat='Continuous')
z = LpVariable.dicts("open", (j for j in distribution_centers), cat='Binary')

# Objective function: Minimize transportation and fixed costs
model += (
    lpSum(
        cost_data[i].loc[k, j] * x[i, j, k]
        for i in commodities
        for j in distribution_centers
        for k in customers
    ) +
    lpSum(fixed_costs.loc[fixed_costs['Distribution Center'] == j, 'Fixed cost'].values[0] * z[j]
    for j in distribution_centers)
)

# Constraints
# 1. Customer demand must be met for each commodity
for i in commodities:
    for k in customers:
        model += lpSum(x[i, j, k] for j in distribution_centers) == demand_data[i].loc[demand_data[i]['Customer'] == k, i].values[0], f"Demand_{i}_{k}"

# 2. Distribution center capacity constraints
for i in commodities:
    for j in distribution_centers:
        model += lpSum(x[i, j, k] for k in customers) <= capacity_data[i].loc[capacity_data[i]['Distribution Center'] == j, i].values[0], f"Capacity_{i}_{j}"

# 3. Only open distribution centers can have shipments
M = 2750000  # Updated realistic large number based on maximum demand
for i in commodities:
    for j in distribution_centers:
        for k in customers:
            model += x[i, j, k] <= M * z[j], f"Open_center_{i}_{j}_{k}"

# Solve the problem using Branch and Bound
model.solve()

# Output the results
for v in model.variables():
    print(f"{v.name} = {v.varValue}")

print(f"Total cost: {model.objective.value()}")
