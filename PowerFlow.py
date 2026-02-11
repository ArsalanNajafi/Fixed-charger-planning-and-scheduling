"""
Created on Tue May 17 16:59:40 2022

@author: arsalann
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import Suffix
from GlobalData import GlobalData



[parking_to_node, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, 
 RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()


def PowerFlow(ParkingDemand, Pattern, Price):

    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')    
    
    vb = 12.66
    sb = 10
    zb = vb**2/sb
    PenF = 1
    model=pyo.ConcreteModel()
      
    
    #model.BRANCH  = no_branches # We have |N| branches in radial system
    #model.BUS     = no_buses # We have |N|+1 nodes in radial system but we dismiss root node
    model.HORIZON = SampPerH*24 
    
    model.L     =  pyo.Set(initialize= data['L'])
    model.LINES = pyo.Set(initialize= data['lineCon']) # Lines Set
    model.N = pyo.Set(initialize= data['N']) # Bus Set
    model.T = pyo.Set(initialize=[x+1 for x in range(model.HORIZON)])
    
    
    model.P = pyo.Var(model.L, model.T) # Branch/Line Active power
    model.Q = pyo.Var(model.L, model.T) # Branch/Line Reactive power
    
    def Limit_Volt(model,n,t):
        return (0.5,1.05)
    model.V = pyo.Var(model.N, model.T, bounds=Limit_Volt )
    model.PS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.QS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.Dev = pyo.Var(model.N, model.T, within= pyo.NonNegativeReals)
    model.abs_dev = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, 
                           doc="Absolute voltage deviation from 1.0 p.u.")    
    
    # Network Characteristics
    volt = 12.66
    
   
    # Defining model az concrete one
    model.Activedemand = pyo.Param(model.N, model.T, initialize=0, mutable=True)
    model.Reactivedemand = pyo.Param(model.N, model.T, initialize=0, mutable=True)
      
    
    for node in model.N:
        for t in model.T:
            model.Reactivedemand[node, t] = data['bus_Qd'][node] * Pattern.iloc[t-1]# Power Network characteristics
            model.Activedemand[node, t] = data['bus_Pd'][node] * Pattern.iloc[t-1]# Power Network characteristics
    
    
    for t in model.T:
        for p, n in parking_to_node.items():
            # Add parking demand to node demand
            model.Activedemand[n, t] += ParkingDemand[p, t]
    
    # Active power balance
    def lines_active_power_rule(model, n, t):
        # First Sum: if there is line which connect bus n to bus m "Loads from BUS n to BUS m"
        # Second Sum: if there is line from any bus m to bus n,
        # Active load as input to node n minus sum of the output load from node n and resistance of the line must be zero
        return ( 
                sum(model.P[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - 
                    sum(model.P[l,t]  for l in model.L for m in model.N if (l,n,m) in model.LINES) + 
                    0.001*model.Activedemand[n, t]/sb - model.PS[n,t] == 0
                )
    model.lines_active_power_con = pyo.Constraint(model.N, model.T, rule= lines_active_power_rule)
    
    # RTeactive power balance
    def lines_reactive_power_rule(model, n , t):
        return (
                sum(model.Q[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - 
                sum(model.Q[l,t]  for l in model.L for m in model.N if (l,n,m) in model.LINES) + 
                0.001*model.Reactivedemand[n, t]/sb - model.QS[n,t] == 0
                 )
    model.lines_reactive_power_con = pyo.Constraint(model.N,  model.T, rule=lines_reactive_power_rule)
    
    
    # # kirchhoff_voltage_rule
    def kirchhoff_voltage_rule(model, l, t):
         return (
                 sum(-model.V[n,t] - ((data['R'][l]/zb)*model.P[l,t] + (data['X'][l]/zb)*model.Q[l,t]) + 
                     model.V[m,t] for n in model.N for m in model.N if (l,n,m) in model.LINES) == 0
                 )
    model.kirchhoff_voltage_con = pyo.Constraint(model.L, model.T, rule=kirchhoff_voltage_rule) 
    
    # generatos active power rule
    def gen_active_power_rule(model, n, t):
        if n == 0:
            return pyo.Constraint.Skip
        return model.PS[n,t] == 0
    model.gen_active_power_con = pyo.Constraint(model.N,  model.T, rule=gen_active_power_rule)
    
    
    # generatos reactive power rule
    def gen_reactive_power_rule(model, n, t):
        if n == 0:
            return pyo.Constraint.Skip
        return model.QS[n,t] == 0
    model.gen_reactive_power_con = pyo.Constraint(model.N,  model.T, rule=gen_reactive_power_rule)
    
    
    def Voltage_Pen1(model, n, t):
        return model.V[n, t] >= Vmin - model.Dev[n,t]
    model.conV_dev1 = pyo.Constraint(model.N, model.T, rule = Voltage_Pen1)
    
    
    def Voltage_Pen2(model, n, t):
        return model.Dev[n,t] <= 0.3
    model.conV_dev2 = pyo.Constraint(model.N, model.T, rule = Voltage_Pen2)
    
    
    def abs_dev_upper(model, n, t):
        """Absolute deviation: |V[n,t] - 1.0| <= abs_dev[n,t]"""
        return model.abs_dev[n,t] >= model.V[n,t] - 1.0
    
    def abs_dev_lower(model, n, t):
        return model.abs_dev[n,t] >= -(model.V[n,t] - 1.0)
    model.con_abs_upper = pyo.Constraint(model.N, model.T, rule=abs_dev_upper)
    model.con_abs_lower = pyo.Constraint(model.N, model.T, rule=abs_dev_lower)

    model.obj = pyo.Objective(
        expr=sum(model.abs_dev[n,t] for n in model.N for t in model.T) +
        sum(model.Dev[n,t]*PenF for n in model.N for t in model.T),
        sense=pyo.minimize
    )


####################################### SOLVING MODEL ##############    
     
    SOLVER_NAME="gurobi"
    TIME_LIMIT=100000
    
    solver=pyo.SolverFactory(SOLVER_NAME)
    
    if SOLVER_NAME == 'cplex':
        solver.options['timelimit'] = TIME_LIMIT
    elif SOLVER_NAME == 'glpk':         
        solver.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':           
        solver.options['TimeLimit'] = TIME_LIMIT
    solver.options = {
    'NumericFocus': 3,  # Highest numerical stability
    'ScaleFlag': 2,      # Aggressive scaling
    'FeasibilityTol': 1e-6,
    'OptimalityTol': 1e-6
    }
    # solver.options['NonConvex'] = 2
    model.dual = Suffix(direction=Suffix.IMPORT)
    results = solver.solve(model)
    
    
    duals_Dev = {
        (n, t): model.dual[model.conV_dev1[n, t]]
        for n in model.N 
        for t in model.T
        if model.conV_dev1[n, t] in model.dual
    }
    
    duals_DevLow = {
        (n, t): model.dual[model.con_abs_lower[n, t]]
        for n in model.N 
        for t in model.T
        if model.con_abs_lower[n, t] in model.dual
    }
    duals_DevUp = {
        (n, t): model.dual[model.con_abs_upper[n, t]]
        for n in model.N 
        for t in model.T
        if model.con_abs_upper[n, t] in model.dual
    }
    
    duals_balance_p = {
    (n,t): model.dual[model.lines_active_power_con[n,t]]
    for n in model.N for t in model.T
    if model.lines_active_power_con[n,t] in model.dual
    }

    
    with open('Power_Network_Constraints.txt', 'w') as f:
        f.write("Description of the power network constraints:\n")
        model.pprint(ostream=f)
        
           
    print("Voltage in each branch:")
    for l in model.L:
        print(pyo.value(model.V[l,1]))
       
    
    volt = [pyo.value(model.V[l,1]) for l in model.L]
    min_volt_per_node = {
        l: min(pyo.value(model.V[l, t]) for t in model.T)
        for l in model.L
    }
    
    
    nodes = list(min_volt_per_node.keys())
    min_voltages = list(min_volt_per_node.values())
    

    volt_per_node = [pyo.value(model.V[l, t]) for l in model.L for t in model.T]
    
    volt_per_node = np.zeros((len(model.L), len(model.T)))

# Fill with Pyomo values
    for i, l in enumerate(model.L):
        for j, t in enumerate(model.T):
            volt_per_node[i, j] = pyo.value(model.V[l, t])
    
    
    
    ################# Plot
    plt.figure(figsize=(8, 4))
    plt.plot(nodes, min_voltages, color='red')
    plt.xlabel('Node')
    plt.ylabel('Minimum Voltage')
    plt.title('Minimum Voltage per Node over Time Horizon')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("MinVoltage.png", dpi=600)

    
    PS  = [pyo.value(model.PS[n,1]) for n in model.N]
    QS  = [pyo.value(model.QS[n,1]) for n in model.N]

    
    df = pd.DataFrame({"V":volt})   
    with pd.ExcelWriter('output.xlsx') as writer:
        df.to_excel(writer,  sheet_name='Voltage')
        
    df2 = pd.DataFrame({"P_Active":PS,
                       "Q_Reactive":QS})    
    with pd.ExcelWriter('output2.xlsx') as writer:
        df2.to_excel(writer,  sheet_name='Optimal_Power')    
        
    plt.figure(2)    
    plt.plot(volt)
    plt.title('Voltage over node')
    plt.xlabel('Node')
    plt.ylabel('Voltage (pu)')
    plt.grid(True)
    plt.show()
    
    SPobj = pyo.value(model.obj)

    return min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj
    print(volt_per_node)    