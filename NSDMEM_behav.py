# NSDMEM_behav
# analyze behavioral data and generate figures
# Futing Zou 6/26/22

## load packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import math
from pymer4.models import Lmer
import sklearn

## import data
subjlist = range(1, 9)
df_behav_recog = pd.read_csv("behav_recog.csv")
df_behav_timeline = pd.read_csv("behav_timeline.csv")

### recognition memory
df_recog = []
for subj in subjlist:
    df_sub = df_behav_recog[df_behav_recog['SUBJECT']==subj]

    hit = df_sub.RECOGRESP.astype('bool')[df_sub.STIM.astype('bool')]
    miss = ~df_sub.RECOGRESP.astype('bool')[df_sub.STIM.astype('bool')]
    fa = df_sub.RECOGRESP.astype('bool')[~df_sub.STIM.astype('bool')]
    cr = ~df_sub.RECOGRESP.astype('bool')[~df_sub.STIM.astype('bool')]

    dprime = stats.norm.ppf(hit.mean()) - stats.norm.ppf(fa.mean())
    c = -(stats.norm.ppf(hit.mean()) + stats.norm.ppf(fa.mean()))/2.0

    df_recog.append({'subjix': subj,
                        'hit': sum(hit),
                        'miss':sum(miss),
                        'fa':sum(fa),
                        'cr':sum(cr),
                        'hitRate': hit.mean(),
                        'faRate': fa.mean(),
                        'dprime': dprime,
                        'c':c
                        })
df_recog = pd.DataFrame(df_recog)

Z = norm.ppf
def SDT(hits, misses, fas, crs):
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1:
        fa_rate = 1 - half_fa
    if fa_rate == 0:
        fa_rate = half_fa

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))

    return(out)

df_recog_byconf = []
for subj in subjlist:
    for conf in ([1,6], [2,5], [3,4]):
        df_sub = df_behav_recog.loc[(df_behav_recog['SUBJECT']==subj)&
                                    ((df_behav_recog['RECOGBUTTON']==conf[0])|(df_behav_recog['RECOGBUTTON']==conf[1]))]

        hit = df_sub.RECOGRESP.astype('bool')[df_sub.STIM.astype('bool')]
        fa = df_sub.RECOGRESP.astype('bool')[~df_sub.STIM.astype('bool')]
        miss = ~df_sub.RECOGRESP.astype('bool')[df_sub.STIM.astype('bool')]
        cr = ~df_sub.RECOGRESP.astype('bool')[~df_sub.STIM.astype('bool')]
        dprime = stats.norm.ppf(hit.mean()) - stats.norm.ppf(fa.mean())

        df_recog_byconf.append({'subjix': subj,
                                'conf':['high', 'med', 'low'][conf[0]-1],
                                'dprime': SDT(sum(hit),sum(miss),sum(fa),sum(cr))['d']
                                })
df_recog_byconf = pd.DataFrame(df_recog_byconf)

###Figure 1a&1b###
df_recog_plot = df_recog[["subjix", "hitRate", "faRate"]]
df_recog_plot = pd.melt(df_recog_plot, id_vars='subjix')

f, (ax1,ax2) = plt.subplots(1, 2, figsize=(2.8, 2), sharey=False, sharex=False, gridspec_kw = {'width_ratios':[.8, 1]})
f.subplots_adjust(wspace=.6)
for subj in subjlist:
    ax1.plot(df_recog_plot.loc[df_recog_plot['subjix']==subj, 'variable'],
             df_recog_plot.loc[df_recog_plot['subjix']==subj, 'value'],
             color="#969699", zorder = 0,
             alpha=.6, lw=.68, ls='-')
sns.pointplot(y='value',x='variable', data = df_recog_plot,
              dodge = .6, size = 3, join = False, ci=68, legend=False, ax = ax1)
sns.stripplot(y='value',x='variable', data = df_recog_plot, s=1.5, ax = ax1, jitter = 0, zorder = 100)
sns.despine()
ax1.legend_ = None
ax1.set(ylabel='Proportion',xlabel='',title='',ylim=[0,1], xticklabels=["Hit Rate", "FA Rate"], yticklabels = [0,.2,.4,.6,.8,1])

sns.pointplot('conf',y='dprime', data = df_recog_byconf,
              size = 2.8, order=['low', 'med', 'high'],
              join = True, ci=68,legend=False, ax = ax2)
ax2.legend_ = None
sns.despine()
ax2.set(ylabel="Recognition sensitivity (d')",xlabel='Confidence level',title='',xticklabels = ["Low", "Med", "High"], ylim=[0,2.5])


### temporal memory

### group-level performance
behav_timeline = Lmer("FIRST_RANK ~ TLSESS_RANK + (1|SUBJECT)", data=df_behav_timeline)
behav_timeline.fit()

###Figure 1c###
f, ax1 = plt.subplots(1, 1, figsize=(2.02, 2), sharex=False)
for subj in subjlist:
    sns.regplot(x="FIRST_RANK", y="TLSESS_RANK", data=df_behav_timeline.loc[df_behav_timeline.SUBJECT==subj],
                ax=ax1, scatter=None, ci=68, line_kws={'lw': 1})
sns.despine()
plt.yticks([0, 30, 60, 90, 120])
plt.xticks([0, 30, 60, 90, 120])
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.xlabel('Actual position')
plt.ylabel('Subjective position')

### permutation for individual subject
n=1000
perm_sub = pd.DataFrame()
for subj in subjlist:
    eS_ind =[]
    for s in range(0, n):
        tlsess_shuffle = sklearn.utils.shuffle(behav_timeline.loc[behav_timeline['SUBJECT']==subj, 'TLSESSEST']).rank().tolist()
        error_shuffled = tlsess_shuffle - behav_timeline.loc[behav_timeline['SUBJECT']==subj,'FIRST_RANK']
        eS_ind.append(stats.sem(error_shuffled))
    t = pd.DataFrame({"subj":subj,
                      "perm_sem":eS_ind})
    perm_sub = perm_sub.append(t)

###Figure 1d###
# Initialize the FacetGrid object
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(perm_sub, row="subj", hue="subj",
                  aspect=10, height=.4,
                  row_order=subjlist, hue_order=sub_order,
                  palette=sns.cubehelix_palette(start=.8, rot=-.7, n_colors=36)[14:22])
# Draw the densities in a few steps
g.map(sns.kdeplot, "perm_sem",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "perm_sem", clip_on=False, color="w", lw=2.25, bw_adjust=.5, zorder=0)
g.map(plt.axhline, y=0, lw=1.5, clip_on=False)
for s,ax in zip(subjlist, g.axes.ravel()):
    ax.vlines(stats.sem(behav_timeline.loc[behav_timeline['SUBJECT']==s, 'TLERROR_RANK']),
              ymin=-.01, ymax=1.25, linewidth=1.5, zorder=1,
              color='#db9b3c', linestyle='-')
# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)
# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], xticks=[3, 4, 5, 6])
g.despine(bottom=True, left=True)


###Supp. figure 2###
for subj in subjlist:
    sess_max = max(df_corr.loc[df_corr['SUBJECT']==subj, 'FIRST_SESS'])
    g = sns.jointplot(data=behav_timeline.loc[behav_timeline['SUBJECT']==subj],
                      y="TLSESSEST2", x="FIRST_POS", hue="TLERROR_GROUP",
                      edgecolor="1",linewidth=0, height=1.8, s=5.5, legend=False,
                      xlim=[-8, sess_max+8], ylim=[-8, sess_max+8])
    g.plot_joint(sns.scatterplot, s=6, legend=False, edgecolor="1",linewidth=0)
    g.plot_marginals(sns.histplot, bins=10, legend=False, edgecolor=".3", linewidth=.5)
    g.ax_joint.tick_params(labelsize=6)
    g.ax_joint.set_xticks([0, sess_max/2, sess_max])
    g.ax_joint.set_yticks([0, sess_max/2, sess_max])
    g.ax_marg_y.tick_params(left=None)
    g.ax_marg_x.tick_params(bottom=None)
    g.ax_marg_x.spines["bottom"].set_linewidth(0.6)
    g.ax_marg_y.spines["left"].set_linewidth(0.65)
    g.ax_joint.xaxis.set_tick_params(length=3)
    g.ax_joint.yaxis.set_tick_params(length=3)
    g.ax_joint.set_ylabel('Raw subjective position', fontsize=6)
    g.ax_joint.set_xlabel('Raw actual position', fontsize=6)
    g.fig.subplots_adjust(top=.92)