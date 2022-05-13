# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:37:00 2022

@author: mxkep
"""

# Check Correlation

corr = svdata.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(svdata.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
plt.title("Correlation Plot")
ax.set_yticks(ticks)
ax.set_xticklabels(svdata.columns)
ax.set_yticklabels(svdata.columns)
plt.show()




# scatterplot with groups
# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt

"""
data.events.value_counts()
data_counts = data.events.value_counts()
pitch_type_counts = data.pitch_type.value_counts()
data_counts.plot(kind='pie', title='Events value count')
data_counts.plot.bar(title='Events bar plot')
pitch_type_counts.plot(kind='pie', title='Pitch type value count')
pitch_type_counts.plot.bar(title='Pitch type bar plot')
"""

fig, ax = plt.subplots()
ax.hist(svdata['release_speed'], range=(60,110), bins=100)
ax.set_title('Distribution of Pitch Speed')
ax.set_xlabel('MPH')
ax.set_ylabel('Count')
plt.show()

fig, ax = plt.subplots()
ax.hist(svdata['release_spin_rate'],  bins=100)
ax.set_title('Distribution of Pitch Spin Rate')
ax.set_xlabel('RPM')
ax.set_ylabel('Count')
plt.show()

#sns.countplot(data = data_bin, x = 'events',hue ='pitch_type')
#sns.catplot(data=data_binary, kind="swarm", x="events", y="release_speed", hue="pitch_type")
#sns.pairplot(data=data_binary, hue="events")

# Use the 'hue' argument to provide a factor variable
sns.lmplot(x="release_pos_x", y="release_pos_y", data=svdata, fit_reg=False, hue='events', legend=True).set(title='Cartesian Plot of Release Point')
sns.lmplot(x="release_pos_x", y="release_pos_y", data=svdata, fit_reg=False, hue='pitch_type', legend=True).set(title='Cartesian Plot of Release Point')

sns.lmplot( x="plate_x", y="plate_y", data=svdata, fit_reg=False, hue='events', legend=True).set(title='At-Bat Outcome by Plate Position')
sns.lmplot( x="plate_x", y="plate_y", data=svdata, fit_reg=False, hue='pitch_type', legend=True).set(title='Pitch Type by Plate Position')

sns.lmplot( x="release_pos_x", y="release_pos_y", data=fastballs, fit_reg=False, hue='events', legend=True).set(title='At-Bat Outcomes Against Fastballs (Pitcher)')
sns.lmplot( x="plate_x", y="plate_y", data=fastballs, fit_reg=False, hue='events', legend=True).set(title='At-Bat Outcomes Against Fastballs (Batter)')

sns.histplot(data=svdata, x="pitch_type", hue="events", multiple="stack").set(title='Histogram of Pitch Type')
sns.histplot(data=svdata, x="events", hue="pitch_type", multiple="stack").set(title='Histogram of At-Bat Outcomes')


sns.barplot(data=svdata, x="pitch_type", y="release_spin_rate", hue="events").set(title='Pitch Type by Spin Rate')
sns.barplot(data=svdata, x="pitch_type", y="release_speed", hue="events").set(title='Pitch Type by Speed')


sns.catplot(x="events", 
                       y="release_speed", 
                       hue="events", 
                       data=svdata, 
                       palette="colorblind",
                       kind='box',
                       height = 5,
                       aspect = 1.5,
                       legend=False).set(title='At-Bat Outcomes by Pitch Speed')

sns.catplot(x="events", 
                       y="release_spin_rate", 
                       hue="events", 
                       data=svdata, 
                       palette="colorblind",
                       kind='box',
                       height = 5,
                       aspect = 1.5,
                       legend=False).set(title='At-Bat Outcomes by Spin Rate')

# Move the legend to an empty part of the plot
#plt.legend(loc='upper_right')
#plt.show()


#ax = plt.axes(projection='3d', xlim=(-5.0, 5.0), ylim=(50.0,60.0), zlim=(3,8))
#ax.scatter3D(data.release_pos_x, data.release_pos_y, data.release_pos_z, c=data_binary, cmap='Greens');
