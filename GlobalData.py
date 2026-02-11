"""
Created on Fri May 16 11:06:36 2025

@author: arsalann
"""

def GlobalData():
    parking_to_bus = {1: 28, 2: 17}
    #parking_to_bus = {1: 17}
    ChargerCap = 22 #kW
    SampPerH = 1
    Vmin = 0.9
    Ch_cost = 5500 #euro
    nChmax = 1000 #upper limit of number of chargers based on the area limit
    PgridMax = 2000 #kW
    #RobotCapMax = 500 #kWh
    NYearCh = 15
    NYearRob = 10
    BesskWh = 250
    RobotCost = 4000
    ConvertorCost = 2000   
    RobotTypes = [50, 75, 100, 125]

 
    robotCC = [50*BesskWh + RobotCost +  ConvertorCost, 75*BesskWh + RobotCost +  ConvertorCost, 100*BesskWh + RobotCost +  ConvertorCost, 125*BesskWh + RobotCost +  ConvertorCost]
    MaxRobot = 30
    DCchargerCap = 50
    PeakPrice = 13.5
    IR = 0.05
    NevSame = 1
    

    return parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame



## Battery Size	    ESS Cost	Robot Cost	On Borad converter	 Total Capital Cost
#    50 kWh	        $7,500	    $4,000	    $1500	            ~$13000
#    75 kWh	        $11,250	    $4,000	    $1500	            ~$16750
#    100 kWh	    $15,000	    $4,000	    $1500	            ~$20500
#    125 kWh	    $18,750	    $4,000	    $1500	            ~$24250
#    150kWh	        22500	    4000	    $1500	             $28000

