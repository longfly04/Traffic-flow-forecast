import pandas as pd
import re
import sys
sys.path.append(
    'C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\Traffic-flow-forecast')

taxi = pd.read_csv('data\\trainTaxiGPS\\trainTaxiGPS_1.csv', usecols=[
                   'taxiID', 'lng', 'lat', 'GPSspeed', 'direction', 'timestamp'])
traffic_flow = pd.read_csv('data\\trainCrossroadFlow\\train_trafficFlow_1.csv', usecols=[
                           'timestamp', 'crossroadID', 'vehicleID'])


taxi_id = taxi['taxiID'].unique()
v_id = traffic_flow['vehicleID'].unique()
taxi_code = pd.Series([i[2:] for i in taxi_id])
v_code = [i[3:] for i in v_id]
count = 0
for i in v_code:
#     print("finding %s" %i)
    for j in range(len(taxi_code)):
        if re.search(i, taxi_code.iloc[j]):
            print("got one! ", i, taxi_code.iloc[j])
            count += 1

print(count)
