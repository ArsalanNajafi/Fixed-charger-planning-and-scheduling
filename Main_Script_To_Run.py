"""
Created on Thu Aug 14 11:21:15 2025

@author: arsalann
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PowerFlow import PowerFlow  # I solve the SP using this module
import os
import seaborn as sns
from build_master import build_masterOnlyFC
from GlobalData import GlobalData

[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, 
  NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

 # Initialize parameters
ParkNo = len(parking_to_bus)
sb = 10  # Base MVA

PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
PFV_Rob = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))


# Storage dictionaries
results = {
    'P_btot': {}, 'P_b_EV': {}, 'x': {}, 'z': {}, 'assign': {}, 
    'Ns': {}, 'P_ch_EV': {}, 'PeakPower': {}, 'Alpha': {}
}

P_btot_Parkings = {}  # Format: {(s,t): value}, I just add 'f' to the end of the variables to give the name to the dic formats
P_b_EVf = {}
xf = {}
zf = {}
assignf = {}
Nsf = {}
P_ch_EVf = {}
PeakPowerf = {}
Alphaf = {}

# Load data
current_directory = os.path.dirname(__file__)
current_directory = os.getcwd()
file_path         = os.path.join(current_directory, 'data.xlsx')
file_path3 = os.path.join(current_directory, 'day2PublicWork.xlsx')
parking_data = pd.read_excel(file_path3, sheet_name='clustered2')
Price = pd.read_excel(file_path, sheet_name='electricicty_price')
Pattern = pd.read_excel(file_path, sheet_name='DemandPattern')
Pattern = np.repeat(Pattern['Pattern'], SampPerH)
Price = np.repeat(Price['Price'], SampPerH)
 
# Models for each parking
parking_models = {}
for s in range(1, ParkNo+1):
    parking_models[s] = build_masterOnlyFC(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price) 

################# Benders loop #############
converged = False
kl = 0
max_iter = 10
volt_per_node = np.ones((33, SampPerH*24))  # Feasible voltage nodes

while True:
    kl += 1
    print(f"\n=== Iteration {kl} ===")
    
    # Solve all parking problems in the master
    P_btot_current = {}
    for s, model in parking_models.items():
        print(f"\nSolving parking {s} master...")
        solver = pyo.SolverFactory('gurobi')
        solver.options = {
            'Presolve': 2,          # Aggressive presolve
            'MIPGap': 0.05,         # duality gap
            'Heuristics': 0.8,      
            'Cuts': 2,              # Maximum cut generation
            'Threads': 8,           # Full CPU utilization
            'NodeMethod': 2,        # Strong branching
        }
        results = solver.solve(model, tee=True)
        
        # Store results
        for t in model.T:
            P_btot_current[(s,t)] = pyo.value(model.P_btot[t])
        print(f"Parking {s} Alpha: {pyo.value(model.Alpha):.2f}")
        
        for t in model.T:
            P_btot_Parkings[(s, t)] = pyo.value(model.P_btot[t])

        P_b_EVf.update({
            (k,i,s,t): pyo.value(model.P_b_EV[k,i,t]) 
            for (k,i,t) in model.x_indices
        })
        
        xf.update( {
            (k, i , s, t):  pyo.value(model.x[k, i, t]) 
            for (k, i, t) in model.x_indices  # Uses predefined sparse indices
        })
       
        for  i  in model.I:
            zf[(i,s)] = pyo.value(model.z[i])
        
        
        for k in model.K:
            for i in model.I:
                assignf[(k,i,s)] = pyo.value(model.assign[k, i])
                        
        Nsf[(s)] = pyo.value(model.Ns)
 
        for k in model.K:
            for t in model.T:
                P_ch_EVf[(k , s, t)] =   pyo.value(model.P_ch_EV[k, t])  
     
        print(f"Parking {s} Alpha: {pyo.value(model.Alpha):.2f}")
        PeakPowerf[(s)] = pyo.value(model.PeakPower)
        Alphaf[(s)] = pyo.value(model.Alpha)

    # Step 2: Solve global power flow
    print("\nSolving power flow subproblem...")
    min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj = PowerFlow(
        P_btot_current, np.repeat(Pattern, SampPerH), np.repeat(Price, SampPerH))
    if np.all(volt_per_node >= Vmin - 1e-3) or kl> max_iter:
        print("Voltage constraints satisfied. Optimal solution found.")
        break
    # Add cuts and check convergence
    #converged = True
    for s, model in parking_models.items():
        bus = parking_to_bus[s]       
        for t in model.T:
            if volt_per_node[bus-1, t-1] < Vmin - 1e-3:

                violation = Vmin - volt_per_node[bus-1, t-1]
                
                # Adding cuts
                model.cuts.add(
                    SPobj + (
                        -(duals_DevLow.get((bus,t), 0) + 
                          duals_balance_p.get((bus,t), 0) + 
                          duals_DevUp.get((bus,t), 0)) * 100 * violation *
                        (model.P_btot[t] - 0*P_btot_current[(s,t)])/sb
                    ) <= model.Alpha
                )
                print(f"Added cut for parking {s} (bus {bus}) at t={t}")

        # Update lower bound
        current_alpha = pyo.value(model.Alpha)
        if current_alpha > model.AlphaDown.value:
            model.AlphaDown.value = current_alpha
            print(f"Updated AlphaDown[{s}] = {current_alpha:.2f}")

    print(f"\nVoltage status: Min={np.min(volt_per_node):.4f}")
    if converged:
        print("\n*** All voltages feasible - convergence achieved! ***")
        break
     
################### Visualization  ########################
Ncharger_val = {(s): Nsf[s] for s in range(1, ParkNo+1)}
print(Ncharger_val)

##### First figure: Purchased electricity
plt.figure(figsize=(10, 8))  

# --- Overall Purchased Electricity ---
plt.subplot(2, 1, 1)
time = range(1, SampPerH * 24 + 1)
total_power = [
    sum(P_btot_Parkings.get((s,t), 0) for s in range(1, ParkNo+1))  # Sum across all parkings
    for t in model.T
]
plt.plot(time, total_power, 'b-', linewidth=1.5)
plt.xlabel('Time period', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.title('Overall Purchased Electricity', fontsize=14)
plt.grid()

# --- Per-Parking Purchased Electricity ---
plt.subplot(2, 1, 2)
colors = ['r', 'g', 'b', 'm', 'c']  # Different colors for each parking
for s in range(1, ParkNo+1):
    P_Tot_Purch = [
        P_btot_Parkings.get((s,t), 0)  # Get from saved results
        for t in model.T
    ]
    plt.plot(time, P_Tot_Purch, 
             label=f'Parking {s}',
             color=colors[s-1],  # different color for each parking
             linewidth=1.5)

plt.xlabel('Time period', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.title('Purchased Electricity by Parking', fontsize=14)
plt.grid()
plt.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('PurchasePower.png', dpi=600, bbox_inches='tight')
 
#### end of the first figure


# Create DataFrames for x and y variables
x_data = []
for (k, i, s, t), val in xf.items():  # we use xf dictionary
    x_data.append({
        'EV': k,
        'Charger': i,
        'Parking': s,
        'Time': t,
        'Value': val  
    })

 
# Converting to DataFrames
df_x = pd.DataFrame(x_data)
 
# Calculate costs using saved values
TotalChargerCost = sum(Ch_cost * Nsf[s] for s in range(1, ParkNo+1))
print("Total Charger Cost = ", TotalChargerCost)

TotalPurchaseCost = sum(Price.iloc[t - 1] * (0.001) * P_btot_Parkings[(s,t)] 
                   for s in range(1, ParkNo+1) for t in model.T)
print("Total cost of electricity purchased = ", TotalPurchaseCost)

TotalPeakCost = (1 / 30) * PeakPrice * sum(PeakPowerf[s] for s in range(1, ParkNo+1))
print("Total cost of Peak Power = ", TotalPeakCost)


# Calculate objective value from components
TotalCost = (TotalChargerCost + TotalPurchaseCost + TotalPeakCost)
print("Total Cost = ", TotalCost)


ObjectiveFun = (PFV_Charger*TotalChargerCost + (1/SampPerH)*TotalPurchaseCost + TotalPeakCost)
print("Objective function = ", ObjectiveFun)



# Create and plot charger utilization heatmap
charger_utilization = df_x.pivot_table(index='Time', 
                                      columns='Charger', 
                                      values='Value',
                                      aggfunc='sum',
                                      fill_value=0)

 
# Parking-specific charger utilization plots
parkings = sorted(df_x['Parking'].unique())

for parking in parkings:
    df_parking = df_x[df_x['Parking'] == parking].copy()
    original_charger_ids = sorted(df_parking['Charger'].unique())
    charger_id_mapping = {orig_id: new_id + 1 
                         for new_id, orig_id in enumerate(original_charger_ids)}
    df_parking['Charger_Sequential'] = df_parking['Charger'].map(charger_id_mapping)
    
    charger_utilization = df_parking.pivot_table(
        index='Time', 
        columns='Charger_Sequential', 
        values='Value',
        aggfunc='sum',
        fill_value=0
    )
    
    # Skip if no active chargers
    if charger_utilization.empty:
        print(f"No active chargers in Parking {parking}. Skipping.")
        continue
    
    # Create complete time index
    full_index = range(1, model.HORIZON + 1)
    charger_utilization = charger_utilization.reindex(full_index, fill_value=0)
    
    # heatmap plot
    plt.figure(figsize=(15, 6))
    ax = sns.heatmap(
        charger_utilization.T,
        cmap=['white', 'green'],
        linewidths=0.5,
        linecolor='lightgray',
        cbar=False
    )
    
    # x-axis labels
    n = max(1, len(full_index) // 10)
    xticks = [i for i in full_index if i % n == 1 or i == full_index[-1]]
    ax.set_xticks([x - 1.5 for x in xticks])
    ax.set_xticklabels(xticks)
    
    plt.title(f'Parking {parking} - Charger Utilization', fontsize=14)
    plt.xlabel('Time Period', fontsize=14)
    plt.ylabel('Charger ID', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'Parking_{parking}_Charger_Utilization.png', dpi=300, bbox_inches='tight')
     
    plt.show()

###############################
######################### SAVING VARS FOR FUTURE NEEDS #########################
MyresultsMM_FC = {
    # Power variables
    'P_b_EV': P_b_EVf,          # Dict: (k,i,s,t) → value
    'x': xf,                    # Dict: (k,i,s,t) → value
    'z': zf,                    # Dict: (i,s) → value
    'assign': assignf,          # Dict: (k,i,s) → value
    'Ns': Nsf,                  # Dict: (s) → value
    'P_btot': P_btot_Parkings,  # Dict: (s,t) → value
    'P_ch_EV': P_ch_EVf,        # Dict: (k,s,t) → value
    'PeakPower': PeakPowerf,    # Dict: (s) → value
    'Alpha': Alphaf,  # Single value
   
}

import pickle

with open('pyomo_resultsMM.pkl', 'wb') as f:

    pickle.dump(MyresultsMM_FC, f)
