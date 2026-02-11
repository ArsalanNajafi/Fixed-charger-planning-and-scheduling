"""
Created on Wed Jun 25 15:44:11 2025

@author: arsalann
"""

def max_overlaps_per_parking(df):
    parking_nos = df['ParkingNo'].unique()
    max_overlaps = {}
    
    for parking_no in parking_nos:
        # Filter EVs for the current parking station
        parking_evs = df[df['ParkingNo'] == parking_no]
        
        # Get all possible time slots (min AT to max DT)
        min_time = parking_evs['AT'].min()
        max_time = parking_evs['DT'].max()
        
        max_overlap = 0
        
        # Check each time slot in the range
        for t in range(min_time, max_time + 1):
            # Count EVs present at time t
            overlap = sum((ev['AT'] <= t) & (t <= ev['DT']) for _, ev in parking_evs.iterrows())
            
            # Update max overlap if current count is higher
            if overlap > max_overlap:
                max_overlap = overlap
        
        max_overlaps[parking_no] = max_overlap
    
    return max_overlaps

