# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:27:57 2022

@author: mxkep
"""
import pandas as pd

savant_data = pd.read_csv("C:\\Users\\mxkep\\OneDrive\\HU\\699\\project\\data\\savant_data.csv")



# choose important columns
svdata = savant_data[['pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z', 
                    'release_pos_y', 'events', 'balls', 'strikes', 'plate_x', 'plate_y', 
                    'outs_when_up', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_spin_rate']]




svdata.drop(svdata[svdata['release_speed'] < 50].index, inplace = True)

svdata.drop(svdata[svdata['events'] == 'strikeout_double_play'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'fielders_choice_out'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'double_play'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'grounded_into_double_play'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'force_out'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'sac_fly'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'sac_bunt'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'sac_fly_double_play'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'sac_bunt_double_play'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'fielders_choice'].index, inplace = True)
svdata = svdata.dropna()

############
"""
svdata.drop(svdata[svdata['events'] == 'triple'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'double'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'home_run'].index, inplace = True)
svdata.drop(svdata[svdata['events'] == 'single'].index, inplace = True)
"""

############


target = svdata['events']


# Make a dataset of only fastballs, since they are the most frequent pitch
fastballs = svdata[svdata['pitch_type'] == 'FF']



data_bin = svdata

################################################
### This block converts to binary classification
################################################

data_bin.loc[data_bin['events'] == 'field_out', 'events'] = 'out'
data_bin.loc[data_bin['events'] == 'strikeout', 'events'] = 'out'
data_bin.loc[data_bin['events'] == 'force_out', 'events'] = 'out'
data_bin.loc[data_bin['events'] == 'grounded_into_double_play', 'events'] = 'out'
data_bin.loc[data_bin['events'] == 'double_play', 'events'] = 'out'
data_bin.loc[data_bin['events'] == 'fielders_choice_out', 'events'] = 'out'
data_bin.loc[data_bin['events'] == 'strikeout_double_play', 'events'] = 'out'
data_bin.loc[data_bin['events'] != 'out', 'events'] = 'hit'

# Prepare for statistical analysis i.e. remove categorical variables
hits = data_bin[data_bin['events'] == 'hit']
hitsstat = hits
hitsstat = hitsstat.drop(['pitch_type', 'release_pos_x', 'release_pos_z',
                          'release_pos_y', 'plate_x', 'plate_y',
                          'outs_when_up', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'events'], axis=1)


outs = data_bin[data_bin['events'] == 'out']
outsstat = outs
outsstat = outsstat.drop(['pitch_type', 'release_pos_x', 'release_pos_z',
                          'release_pos_y', 'plate_x', 'plate_y',
                          'outs_when_up', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'events'], axis=1)

print("HITS SUMMARY:\n", hitsstat.describe(include='all'))
print("\nOUTS SUMMARY:\n", outsstat.describe(include='all'))

################
# Make an equal about of hit/out observations


hits = svdata[svdata['events'] == 'hit']
outs = svdata[svdata['events'] == 'out'].sample(n=10410, random_state=123)

svdata = hits
svdata = svdata.append(outs)

################
# Convert to 0/1
################
svdata['events'] = np.where(svdata['events'] == 'out', 0, 1) # 0 for out, 1 for hit















