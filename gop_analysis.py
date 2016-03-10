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
trump_counter = load_obj("trump_gop_debate_03032016")
rubio_counter = load_obj("rubio_gop_debate_03032016")
cruz_counter = load_obj("cruz_gop_debate_03032016")
kasich_counter = load_obj("kasich_gop_debate_03032016")

#create dictionaries from arrays
dict_trump = {}
dict_rubio = {}
dict_cruz = {}
dict_kasich = {}

for word,count in trump_counter:
    dict_trump[word] = count
for word,count in rubio_counter:
    dict_rubio[word] = count
for word,count in cruz_counter:
    dict_cruz[word] = count
for word,count in kasich_counter:
    dict_kasich[word] = count


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
candidates = ['trump', 'cruz', 'rubio', 'kasich']

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
total_grid = norm_trump_grid + norm_cruz_grid + norm_rubio_grid + norm_kasich_grid
total_grid = total_grid/4.0

#subtract mean grid from individual grids
flat_trump_grid = norm_trump_grid-total_grid
flat_rubio_grid = norm_rubio_grid-total_grid
flat_cruz_grid = norm_cruz_grid-total_grid
flat_kasich_grid = norm_kasich_grid-total_grid



################################################
# Plot 2x2 comparison of differential ratings  #
################################################

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9.7))

#standard deviation of variations from 0 (=mean)
sig_tot = np.std([flat_trump_grid,flat_cruz_grid,flat_rubio_grid,flat_kasich_grid])

#set font and increase font size
mpl.rcParams.update({'font.size': 17, 'font.family':'sans-serif'})


#first subplot (Trump)
ax1 = plt.subplot(2, 2, 1)

#show Trump difference map with fixed z-range
im1 = ax1.imshow(flat_trump_grid, cmap="seismic", clim=(-0.000030,0.000030))

#set contour levels to multiples of standard deviation
levels1 = np.arange(-20.0,20.0,1.0)*sig_tot
CS1 = ax1.contour(flat_trump_grid, levels1, linewidths=2, colors='white', linestyles='solid')

#axis labels
ax1.set_xlabel('')
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
ax1.text(5,10, 'Trump', style='italic', weight='roman')



#second subplot (Cruz)
ax2 = plt.subplot(2, 2, 2)

im2 = ax2.imshow(flat_cruz_grid, cmap="seismic", clim=(-0.000030,0.000030))
levels2 = np.arange(-20.0,20.0)*sig_tot
CS2 = ax2.contour(flat_cruz_grid, levels2, linewidths=2, colors='white', linestyles='solid')

ax2.set_xlabel('')
ax2u = ax2.twiny()
ax2u.set_xlabel('excited', weight='ultralight')
ax2.set_xticks([])
ax2u.set_xticks([])
ax2.set_ylabel('')
ax2r = ax2.twinx()
ax2r.set_ylabel('pleasant', weight='ultralight', rotation=0, labelpad=42.5, ha='center', va='center')
ax2.set_yticks([])
ax2r.set_yticks([])

ax2.text(5,10, 'Cruz', style='italic', weight='roman')



#third subplot (Rubio)
ax3 = plt.subplot(2, 2, 3)

im3 = ax3.imshow(flat_rubio_grid, cmap="seismic", clim=(-0.000030,0.000030))
levels3 = np.arange(-20.0,20.0)*sig_tot
CS3 = ax3.contour(flat_rubio_grid, levels3, linewidths=2, colors='white', linestyles='solid')

ax3.set_xlabel('calm', weight='ultralight')
ax3u = ax3.twiny()
ax3u.set_xlabel('')
ax3.set_xticks([])
ax3u.set_xticks([])
ax3.set_ylabel('unpleasant', weight='ultralight', rotation=0, labelpad=50, ha='center', va='center')
ax3r = ax3.twinx()
ax3r.set_ylabel('')
ax3.set_yticks([])
ax3r.set_yticks([])

ax3.text(5,10, 'Rubio', style='italic', weight='roman')



#fourth subplot (Kasich)
ax4 = plt.subplot(2, 2, 4)

im4 = ax4.imshow(flat_kasich_grid, cmap="seismic", clim=(-0.000030,0.000030))
levels4 = np.arange(-20.0,20.0)*sig_tot
CS4 = ax4.contour(flat_kasich_grid, levels4, linewidths=2, colors='white', linestyles='solid')

ax4.set_xlabel('calm', weight='ultralight')
ax4u = ax4.twiny()
ax4u.set_xlabel('')
ax4.set_xticks([])
ax4u.set_xticks([])
ax4.set_ylabel('')
ax4r = ax4.twinx()
ax4r.set_ylabel('pleasant', weight='ultralight', rotation=0, labelpad=42.5, ha='center', va='center')
ax4.set_yticks([])
ax4r.set_yticks([])

ax4.text(5,10, 'Kasich', style='italic', weight='roman')


#create axis for colorbar and anchor far right of the 2x2 plot
cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], anchor=(4,1), label='deviation from mean', format="")
plt.colorbar(im2, cax=cax, **kw)

#labels for colorbar
ax2.text(152,-2,'more', weight='ultralight')
ax4.text(154,107,'less', weight='ultralight')


#save figure in dir 'plots'
plt.savefig('plots/difference_maps.png', format='png', dpi=150, bbox_inches='tight')




