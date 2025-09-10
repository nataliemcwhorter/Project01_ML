import pandas as pd
data_p = pd.read_csv('../data/drivers.csv')

for i in range(len(data_p)):
    print('\'' + data_p.loc[i, 'forename'] + ' ' + data_p.loc[i, 'surname'] + '\' : ' + str(data_p.loc[i, 'driverId']) + ',')