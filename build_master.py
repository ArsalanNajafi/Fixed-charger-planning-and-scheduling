"""
Created on Tue Aug 12 12:01:57 2025

@author: arsalann
"""
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from MaxOverlap import max_overlaps_per_parking
from GlobalData import GlobalData
  
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##########  ONLY FIXED CHARGER ###################
def build_masterOnlyFC(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
    [parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, 
          robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()
    
    modelFC = pyo.ConcreteModel()

    EVdata = parking_data[parking_data['ParkingNo'] == s].reset_index(drop=True)
     
    # Calculate and print results
    max_overlaps = max_overlaps_per_parking(EVdata)
    print('max_overlaps=', max_overlaps)
    print('s ==',s)
    modelFC.nCharger = max_overlaps[s]
    
    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')   
   
    #################
    ATT = EVdata['AT']
    DTT = EVdata['DT']
        
    plt.subplot(2,1,1)
    plt.hist(ATT, bins=20, color='skyblue', edgecolor='black',label = 'Arrival time')
    plt.legend()
    plt.grid()
    plt.ylabel('Frequency')
    plt.xlim(1,24*SampPerH)
    
    plt.subplot(2,1,2)
    plt.hist(DTT, bins=20, color='red', edgecolor='black', label = 'Departure time')
    plt.xlim(1,24*SampPerH)
    plt.grid()
    plt.legend()
    plt.xlabel('Time sample')
    plt.ylabel('Frequency')
    plt.savefig('Histogram.png', dpi = 300)
    
    plt.show()

    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')
    M = 50 #Big M
    # present value factors
    PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
    print(len(EVdata))
    modelFC.HORIZON = SampPerH * 24
    modelFC.nEV = len(EVdata) - 1
    modelFC.Nodes = pyo.Set(initialize=range(33)) # Buses 0 to 32
    modelFC.T = pyo.Set(initialize=[x + 1 for x in range(modelFC.HORIZON)])
    modelFC.I = pyo.Set(initialize=[x + 1 for x in range(modelFC.nCharger)])
    modelFC.K = pyo.Set(initialize=[x + 1 for x in range(modelFC.nEV)])
    
    # x = binary varible for CS, z = binary variable for choosing CS
    modelFC.x_indices = pyo.Set(dimen=3, initialize=lambda modelFC: (
    (k, i, t)
    for k in modelFC.K
    for i in modelFC.I
    for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
    ))
    # I limit the index of x to not search over unnecessary space
     
    #charger usage tracking
    modelFC.z = pyo.Var(modelFC.I,  within=pyo.Binary)
    modelFC.assign = pyo.Var(modelFC.K, modelFC.I,  within=pyo.Binary) # EV-charger assignment
    modelFC.occupancy = pyo.Var(
        modelFC.I, modelFC.T,
        within=pyo.NonNegativeReals,
        bounds=(0, 1)  
    )
    modelFC.charger_used = pyo.Var(modelFC.I, within=pyo.Binary)  # Charger is installed
    # u = binary variable to buy either from the grid or from the robots
    modelFC.u = pyo.Var(modelFC.K, within=pyo.Binary)
    modelFC.P_btotBar = pyo.Var( modelFC.T, within=pyo.NonNegativeReals)
    # Ns = number of chargers
    modelFC.Ns = pyo.Var( within=pyo.NonNegativeIntegers)
    # P_btot = P_buy_total from the grid to charge the EVs, P_b_EV= Purchased electricity to charge EVS, 
    modelFC.P_btot = pyo.Var( modelFC.T, within=pyo.NonNegativeReals)
    modelFC.P_b_EV = pyo.Var(modelFC.K, modelFC.I,  modelFC.T, within=pyo.NonNegativeReals)
    # P_ch_EV= charging to EV, P_dch_EV = discharging from the EV, SOC_EV = state of charge of the EV
    modelFC.P_ch_EV = pyo.Var(modelFC.K, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.P_dch_EV = pyo.Var(modelFC.K, modelFC.I, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.SOC_EV = pyo.Var(modelFC.K, modelFC.T, within=pyo.NonNegativeReals)
    # Capacity of the robot
    modelFC.PeakPower = pyo.Var( within=pyo.NonNegativeReals) # Peak grid power
    modelFC.Alpha = pyo.Var(within=pyo.Reals)  # Subproblem approximation
    modelFC.AlphaDown = pyo.Param(initialize=-100, mutable=True)  # Lower bound
    
    modelFC.x = pyo.Expression(
        modelFC.K, modelFC.I, modelFC.T,
        initialize=0.0  # Default value
    )
    
    ################ Energy constraints ###########
    def PowerPurchased(modelFC, t):
        return modelFC.P_btot[t] == sum(
            modelFC.P_ch_EV[k, t] 
            for k in modelFC.K 
            if EVdata['AT'][k] <= t <= EVdata['DT'][k]
        )
            
    modelFC.ConPurchasedPower = pyo.Constraint( modelFC.T, rule=PowerPurchased)
    
    def PowerPurchasedLimit(modelFC, t):
        return modelFC.P_btot[ t] <= PgridMax
    modelFC.ConPowerPurchasedLimit = pyo.Constraint( modelFC.T, rule=PowerPurchasedLimit)
    
    ##################### Charger assignment constraints ###############
    
    def NoCharger(modelFC):
        return modelFC.Ns == sum(modelFC.z[i] for i in modelFC.I)
    modelFC.ConNoCharger = pyo.Constraint( rule=NoCharger)
    # Each EV assigned to <= 1 charger
    
    def SingleChargerAssignment(modelFC, k):
        return sum(modelFC.assign[k, i] for i in modelFC.I) <= 1
    modelFC.ConSingleAssign = pyo.Constraint(modelFC.K,  rule=SingleChargerAssignment)
    
    def occupancy_limit(modelFC, i, t):
            """Occupancy cannot exceed 1"""
            return modelFC.occupancy[i, t] <= 1       
    modelFC.ConOccupancyLimit = pyo.Constraint(
        modelFC.I, modelFC.T, rule=occupancy_limit
    )


    def charger_installation(modelFC, i):
        """Charger must be installed if any EV is assigned to it"""
        # Sum of all assignments to charger i
        total_assignments = sum(modelFC.assign[k, i] for k in modelFC.K)       
        # If any EV is assigned, charger must be installed
        return modelFC.z[i] >= total_assignments / M  
        
    modelFC.ConChargerInstallation = pyo.Constraint(
        modelFC.I, rule=charger_installation
    )

    def simplified_power_constraint(modelFC, k, t):
        """Simplified linear power constraint"""
        at = EVdata['AT'][k]
        dt = EVdata['DT'][k]
        
        if t < at or t > dt:
            return modelFC.P_ch_EV[k, t] == 0
        
        # Power limited by whether EV is assigned to ANY charger
        total_assignment = sum(modelFC.assign[k, i] for i in modelFC.I)
        return modelFC.P_ch_EV[k, t] <= ChargerCap * total_assignment
    modelFC.ConSimplePower = pyo.Constraint(modelFC.K, modelFC.T, rule=simplified_power_constraint)
   
    def charger_occupancy_new(modelFC, k, i):
            """If EV k is assigned to charger i, then charger must be occupied during EV's stay"""
            at = EVdata['AT'][k]
            dt = EVdata['DT'][k]
            if at > dt:
                return pyo.Constraint.Skip
            
            # Option A: Sum of occupancy over EV's stay must equal duration if assigned
            # This ensures the charger is fully occupied during the EV's stay
            occupancy_sum = sum(modelFC.occupancy[i, t] for t in range(at, dt + 1)  )
            
            return occupancy_sum >= (dt - at + 1) * modelFC.assign[k, i]
        
    modelFC.ConChargerOccupancyNew = pyo.Constraint(modelFC.K, modelFC.I, rule=charger_occupancy_new)
    
# Create constraints at times when something changes
    def get_critical_times(EVdata):
        """Get all arrival and departure times"""
        critical_times = set()
        for idx, ev in EVdata.iterrows():
            critical_times.add(ev['AT'])
            critical_times.add(ev['DT'])
        return sorted(critical_times)
    
    critical_times = get_critical_times(EVdata)
    modelFC.CRITICAL_TIMES = pyo.Set(initialize=critical_times)
    
    def critical_time_constraint(modelFC, i, ct):
        """Constraint only at critical times"""
        active_at_ct = sum(
            modelFC.assign[k, i]
            for k in modelFC.K
            if EVdata['AT'][k] <= ct <= EVdata['DT'][k]
        )
        return active_at_ct <= 1

    modelFC.ConCriticalTimes = pyo.Constraint(modelFC.I, modelFC.CRITICAL_TIMES, rule=critical_time_constraint )


    ########## charging constraints###################
    def Charging_UpLimit(modelFC, k, t):
        return modelFC.P_ch_EV[k, t] <= ChargerCap
    modelFC.ConCharging_UpLimit = pyo.Constraint(modelFC.K, modelFC.T, rule=Charging_UpLimit)
    
    def SOC_EV_f1(modelFC, k, t):
        if t < EVdata['AT'][k]:
            return modelFC.SOC_EV[k, t] == 0
        elif t == EVdata['AT'][k]:
            return modelFC.SOC_EV[k, t] == EVdata['SOCin'][k] * EVdata['EVcap'][k]
        elif t > EVdata['AT'][k] and t <= EVdata['DT'][k]:
            return modelFC.SOC_EV[k, t] == modelFC.SOC_EV[k, t - 1] + (1 / SampPerH) * modelFC.P_ch_EV[k, t]
        else:
            return pyo.Constraint.Skip # nothing to enforce outside the relevant time range
    def SOC_EV_f2(modelFC, k, t):
        if t == EVdata['DT'][k]:
            return modelFC.SOC_EV[k, t] == 1 * EVdata['SOCout'][k] * EVdata['EVcap'][k]
        else:
            return pyo.Constraint.Skip # nothing to enforce outside the relevant time range
    modelFC.ConSOC_EV_f1 = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_EV_f1)
    modelFC.ConSOC_EV_f2 = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_EV_f2)
   
    def SOC_Charge_limit1(modelFC, k, i, t):
        return modelFC.P_b_EV[k, i, t] <= EVdata['EVcap'][k] * modelFC.assign[k, i]
    modelFC.ConSOC_Charge_limit1 = pyo.Constraint(modelFC.K, modelFC.I,  modelFC.T, rule=SOC_Charge_limit1)

    
    def charging_power_limit(modelFC, k, t):
        """EV charging power cannot exceed charger capacity"""
        return modelFC.P_ch_EV[k, t] <= ChargerCap
    modelFC.ConChargingPowerLimit = pyo.Constraint(modelFC.K, modelFC.T, rule=charging_power_limit )
       
    def SOC_Limit(modelFC, k, t):
        return modelFC.SOC_EV[k, t] <= EVdata['EVcap'][k]
    modelFC.ConSOC_Limit = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_Limit)


    ##################### PARKING CONSTRAINTS ################## 
    def PeakPowerConstraint(modelFC, t):
        return modelFC.PeakPower >= modelFC.P_btot[t]
    modelFC.ConPeakPower = pyo.Constraint( modelFC.T, rule=PeakPowerConstraint)
    
    def AlphaFun(modelFC):
        return modelFC.Alpha >= modelFC.AlphaDown    
    modelFC.ConAlphaFun = pyo.Constraint( rule=AlphaFun)

    ############## OBJECTIVE FUNCTION ##################
    modelFC.obj = pyo.Objective(expr= (PFV_Charger * modelFC.Ns * Ch_cost) +
    (1/SampPerH)*sum(Price.iloc[t - 1] * (0.001) * modelFC.P_btot[t] for t in modelFC.T) +
    (1 / 30) * PeakPrice *modelFC.PeakPower + (modelFC.Alpha) , sense=pyo.minimize)
        
    modelFC.cuts = pyo.ConstraintList()      
   
    ### creating x indexed to 3 sets again
    x_indices = []
    for k in modelFC.K:
        at = int(EVdata['AT'][k])
        dt = int(EVdata['DT'][k])
        for i in modelFC.I:
            for t in range(at, min(dt, modelFC.HORIZON) + 1):
                x_indices.append((k, i, t))
    
    modelFC.x_sparse_indices = pyo.Set(initialize=x_indices, dimen=3)
    modelFC.x = pyo.Expression(modelFC.x_sparse_indices)
    
    for (k, i, t) in modelFC.x_sparse_indices:
        modelFC.x[k, i, t] = modelFC.assign[k, i] * modelFC.occupancy[i, t]
    
    print(f"Created x expression with {len(x_indices)} entries")
          

    return modelFC
