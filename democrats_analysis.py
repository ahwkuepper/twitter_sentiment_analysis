#!/usr/bin/env python
# -*- coding: utf-8 -*-

#for reading and writing
import pickle

#for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

#for statistics on tweeted words
import numpy as np
import pandas as pd


"""Sentiment analysis of keyword lists using the ANEW ratings (Bradley & Lang 1999)
    """


#############
# functions #
#############

#I/O of python objects via pickle files
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#get value of a 2d Gaussian distribution with mean mu's and stddev sigma's
def gaussian2d(x, y, mux, muy, sigmax, sigmay, norm):
    return np.exp(-0.5*(np.power((x - mux)/sigmax, 2.) + np.power((y-muy)/sigmay, 2.0)))/(2.0*np.pi*sigmax*sigmay)



#################
# data analysis #
#################

#read in pickle files with counter arrays
clinton_counter = load_obj("clinton_democrats_debate_03072016")
sanders_counter = load_obj("sanders_democrats_debate_03072016")

#create dictionaries from arrays
dict_clinton = {}
dict_sanders = {}

for word,count in clinton_counter:
    dict_clinton[word] = count
for word,count in sanders_counter:
    dict_sanders[word] = count


#read ANEW csv file with Pandas and create dictionary
dict_anew = {}
df = pd.read_csv('anew-1999/all.csv')
for i in np.arange(len(df)):
    dict_anew[df.loc[i,'Description']] = i


#create a 10x10 units grid for the valence and arousal ratings
xvec = np.arange(0,10,0.1)+0.5
yvec = np.arange(0,10,0.1)+0.5
x_grid, y_grid = np.meshgrid(xvec, yvec)


#select the dictionaries to use in the analysis
candidates = ['clinton', 'sanders']


#match the candidate dictionaries with the ANEW dict and sum up contributions of the words in the grids
for candidate in candidates:
    emo_grid = np.zeros(np.shape(x_grid)) #iniitalize as 0
    exec("dict_plot = dict_"+ candidate)
    for word in dict_plot.keys():
        if word in dict_anew.keys():
            valence = df.loc[dict_anew[word], 'Valence Mean']
            dvalence = df.loc[dict_anew[word], 'Valence SD']
            arousal = df.loc[dict_anew[word], 'Arousal Mean']
            darousal = df.loc[dict_anew[word], 'Arousal SD']
            dominance = df.loc[dict_anew[word], 'Dominance Mean']
            ddominance = df.loc[dict_anew[word], 'Dominance SD']
            for i in np.arange(len(x_grid)):
                emo_grid[i] = emo_grid[i] + gaussian2d(x_grid[i],y_grid[i],valence,arousal,dvalence,darousal,dict_plot[word]) #add contribution of word across the grid (normalized by occurence of key word)
    #                emo_grid[i] = emo_grid[i] + gaussian2d(x_grid[i],y_grid[i],valence,dominance,dvalence,ddominance,dict_plot[word]) #switch if dominance is wanted instead of arousal

    #normalize each grid to 1 (total sum of fields in the grid)
    exec("emo_"+ candidate + "_grid = emo_grid")
    total = np.sum(emo_grid)
    norm_grid = emo_grid/total
    exec("norm_"+ candidate + "_grid = norm_grid")

#create mean grid
total_grid = norm_sanders_grid + norm_clinton_grid
total_grid = total_grid/2.0


#subtract mean grid from individual grids
flat_sanders_grid = norm_sanders_grid-total_grid
flat_clinton_grid = norm_clinton_grid-total_grid




################################################
# Plot 2x1 comparison of differential ratings  #
################################################

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.4))

#standard deviation of variations from 0 (=mean)
sig_tot = np.std([flat_sanders_grid,flat_clinton_grid])

#set font and increase font size
mpl.rcParams.update({'font.size': 17, 'font.family':'sans-serif'})


#first subplot (Clinton)
ax1 = plt.subplot(1, 2, 1)

#show Trump difference map with fixed z-range
im1 = ax1.imshow(flat_clinton_grid, cmap="seismic")

#set contour levels to multiples of standard deviation
levels1 = np.arange(-20.0,20.0,1.0)*sig_tot
CS1 = ax1.contour(flat_clinton_grid, levels1, linewidths=2, colors='white', linestyles='solid')

#axis labels
ax1.set_xlabel('calm', weight='ultralight')
ax1u = ax1.twiny()
ax1u.set_xlabel('excited', weight='ultralight')
ax1.set_xticks([])
ax1u.set_xticks([])
ax1.set_ylabel('unpleasant', weight='ultralight', rotation=0, labelpad=50, ha='center', va='center')
ax1r = ax1.twinx()
ax1r.set_ylabel('')
ax1.set_yticks([])
ax1r.set_yticks([])

#label
ax1.text(5,10, 'Clinton', style='italic', weight='roman')



#second subplot (Sanders)
ax2 = plt.subplot(1, 2, 2)

im2 = ax2.imshow(flat_sanders_grid, cmap="seismic")
levels2 = np.arange(-20.0,20.0)*sig_tot
CS2 = ax2.contour(flat_sanders_grid, levels2, linewidths=2, colors='white', linestyles='solid')

ax2.set_xlabel('calm', weight='ultralight')
ax2u = ax2.twiny()
ax2u.set_xlabel('excited', weight='ultralight')
ax2.set_xticks([])
ax2u.set_xticks([])
ax2.set_ylabel('')
ax2r = ax2.twinx()
ax2r.set_ylabel('pleasant', weight='ultralight', rotation=0, labelpad=42.5, ha='center', va='center')
ax2.set_yticks([])
ax2r.set_yticks([])

ax2.text(5,10, 'Sanders', style='italic', weight='roman')

#save figure in dir 'plots'
plt.savefig('plots/difference_maps_democrats.png', format='png', dpi=150, bbox_inches='tight')




