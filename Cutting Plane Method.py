import numpy as np
import pandas as pd
from scipy.optimize import linprog


# File paths (replace with your actual file paths)
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

capacity_data = {
    'Toffee': distribution_capacity[['Distribution Center', 'Toffee']].set_index('Distribution Center'),
    'Chocolate': distribution_capacity[['Distribution Center', 'Chocolate']].set_index('Distribution Center'),
    'Biscuit': distribution_capacity[['Distribution Center', 'Biscuit']].set_index('Distribution Center')
}

demand_data = {
    'Toffee': demand[['Customer', 'Toffee']].set_index('Customer'),
    'Chocolate': demand[['Customer', 'Chocolate']].set_index('Customer'),
    'Biscuit': demand[['Customer', 'Biscuit']].set_index('Customer')
}

# Cutting Plane Method implementation

def solve_cutting_plane():
    # Problem initialization
    num_customers = len(customers)
    num_centers = len(distribution_centers)
    num_commodities = len(commodities)

    # Variables

    shipment = np.zeros((num_commodities, num_centers, num_customers), dtype=int)
    open_center = np.zeros(num_centers)

    # Objective coefficients
    c = []
    for i in commodities:
        for j in distribution_centers:
            for k in customers:
                c.append(cost_data[i].loc[k, j])
    c += list(fixed_costs['Fixed cost'])

    # Constraints
    A_eq = []
    b_eq = []

    # Demand constraints
    for i, commodity in enumerate(commodities):
        for k, customer in enumerate(customers):
            row = [0] * (num_commodities * num_centers * num_customers + num_centers)
            for j, center in enumerate(distribution_centers):
                row[i * num_centers * num_customers + j * num_customers + k] = 1
            A_eq.append(row)
            b_eq.append(demand_data[commodity].loc[customer, commodity])

    # Capacity constraints
    A_ub = []
    b_ub = []
    for i, commodity in enumerate(commodities):
        for j, center in enumerate(distribution_centers):
            row = [0] * (num_commodities * num_centers * num_customers + num_centers)
            for k in range(num_customers):
                row[i * num_centers * num_customers + j * num_customers + k] = 1
            row[num_commodities * num_centers * num_customers + j] = -capacity_data[commodity].loc[center, commodity]
            A_ub.append(row)
            b_ub.append(0)

    # Link shipment to open centers (Big-M constraints)
    M = 2750000
    for i, commodity in enumerate(commodities):
        for j, center in enumerate(distribution_centers):
            for k in range(num_customers):
                row = [0] * (num_commodities * num_centers * num_customers + num_centers)
                row[i * num_centers * num_customers + j * num_customers + k] = 1
                row[num_commodities * num_centers * num_customers + j] = -M
                A_ub.append(row)
                b_ub.append(0)

    # Bounds
    bounds = [(0, None)] * (num_commodities * num_centers * num_customers) + [(0, 1)] * num_centers

    # Solve using linprog
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')

    # Extract results
    if res.success:
        solution = res.x
        shipment = solution[:num_commodities * num_centers * num_customers].reshape((num_commodities, num_centers, num_customers))
        open_center = solution[num_commodities * num_centers * num_customers:]

        # Force binary results for open centers
        open_center = np.round(open_center)

        # Output results
        for j, center in enumerate(distribution_centers):
            print(f"open_{center} = {int(open_center[j])}")

        for i, commodity in enumerate(commodities):
            for j, center in enumerate(distribution_centers):
                for k, customer in enumerate(customers):
                    print(f"shipment_('{commodity}',_'{center}',_{customer}) = {shipment[i, j, k]:.2f}")

        print(f"Total cost: {res.fun:.2f}")
    else:
        print("Optimization failed.")

# Run the solver
solve_cutting_plane()
