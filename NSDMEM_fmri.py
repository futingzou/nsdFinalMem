# NSDMEM_fmri
# analyze fmri data and generate figures
# Futing Zou 6/26/22


## load packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn
from pymer4.models import Lmer

## import data
subjlist = range(1, 9)
df_ps_recog = pd.read_csv("rsa_recog.csv")
df_ps_timeline = pd.read_csv("rsa_timeline.csv")
roi_list = ['CA1','DGCA23', 'ERC', 'PRC', 'PHC', 'V1']

def pointplot_err(x, y, xerr=None, yerr=None, data=None, **kwargs):
    _data = []
    for _i in data.index:
        _data_i = pd.concat([data.loc[_i:_i]]*3, ignore_index=True, sort=False)
        _row = data.loc[_i]
        if xerr is not None:
            _data_i[x] = [_row[x]-_row[xerr], _row[x], _row[x]+_row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y]-_row[yerr], _row[y], _row[y]+_row[yerr]]
        _data.append(_data_i)
    _data = pd.concat(_data, ignore_index=True, sort=False)

    _ax = sns.pointplot(x=x, y=y,data=_data, ci='sd',
                        edgecolor=(0,0,0), linewidth=0, **kwargs)
    return _ax

### Similarity across repeated exposures

###Figure 3c###
ps_by_sub = pd.DataFrame()
for i in roi_list:
    ds = df_ps_timeline.loc[df_ps_timeline['roi']==i]
    ds.index = range(len(ds))
    for subj in subjlist:
        for j in [1,0]:
            ps_mean = pd.DataFrame({"subj":subj,
                                    "roi": i,
                                    "mem":[1,0][j],
                                    "mean_ps":np.mean(ds.loc[(ds.subj==subj)&
                                                             (ds.mem==j)]['r123'])}, index=[0])
            ps_by_sub = ps_by_sub.append(ps_mean, ignore_index=True)

palatte_roi = ['#746297', '#C2463E', '#e9c46a', '#29658c', '#176533', '#757575',
               "#9081ac", "#CE6B65", "#edd088", "#5484a3", "#45845c", '#9E9E9E']

f, axes = plt.subplots(1, 6, figsize=(6.4, 1.8), sharey=False, sharex=False)
f.subplots_adjust(wspace=0.35)
axes = axes.flatten()
for ax,roi in zip(axes, range(len(roi_list))):
    sns.barplot(
        data=ps_by_sub.loc[ps_by_sub.roi==roi_list[roi]],
        x="mem", y="mean_ps", ax=ax,
        alpha=.9, edgecolor=".3", linewidth=.65,
        ci=68, palette=[palatte_roi[roi], palatte_roi[roi+6]]
    )
    sns.stripplot(y='mean_ps',x='mem',
                  data=ps_by_sub.loc[ps_by_sub.roi==roi_list[roi]], ax=ax,
                  palette=[palatte_roi[roi], palatte_roi[roi+6]], s=1.6, jitter=0, zorder = 100, alpha=.65)
    ax.axhline(y = 0,linewidth=1,linestyle = '--', color='#949494', zorder=200)
    ax.legend_ = None
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_box_aspect(5.5/len(ax.patches))
    if roi == 0:
        ax.set(ylabel="Similarity (z)",xlabel='',title=f'{roi_list[roi]}', xticklabels=["High", "Low"])
    else:
        ax.set(ylabel="",xlabel='',title=f'{roi_list[roi]}', xticklabels=["High", "Low"])
    sns.despine()
f.text(0.5, -.066, 'Temporal memory precision', ha='center')

###mixed-effects models
# temporal memory (high/low) ~ avg(r12,r23,r13) + lag0 + lag1 + lag2 + lag3 + (1|subj)
model_psa_tl = pd.DataFrame()
for i in roi_list:
    ds = betas_psa_tl.loc[betas_psa_tl['roi']==i].reset_index(drop=True)
    mod_logit = Lmer("mem ~ r123 + lag0 + lag1 + lag2 + lag3 + (1|subj)",
                     data=ds.dropna(), family="binomial")
    mod_logit.fit()
    t_logit = mod_logit.coefs[['Estimate', 'SE', 'P-val', 'Sig']][1:]
    t_logit['ROI'] = i
    t['task'] = 'time'
    model_psa_tl = model_psa_tl.append(t_logit)
model_psa_tl = model_psa_tl.reset_index()

# recog confidence (1-6) ~ avg(r12,r23,r13) + lag0 + lag1 + lag2 + lag3 + (1|subj)
model_psa_recog = pd.DataFrame()
for i in roi_list:
    ds = betas_psa_recog.loc[betas_psa_recog['roi']==i].reset_index(drop=True)
    ds['mean_r'] = (ds[['r12','r13','r23']]).mean(axis=1)
    mod = Lmer("recogconf ~ mean_r + lag0 + lag1 + lag2 + lag3 + (1|subj)",
               data=ds.dropna())
    mod.fit()
    t = mod.coefs[['Estimate', 'SE', 'P-val', 'Sig']][1:]
    t['ROI'] = i
    t['task'] = 'recog'
    model_psa_recog = model_psa_recog.append(t)
model_psa_recog = model_psa_recog.reset_index()

###Figure 3d&3e###
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 1.8), sharex=True, sharey=False)
f.subplots_adjust(wspace=0.35)
pointplot_err(x="ROI", y="Estimate", ax=ax1,
              data=model_psa_tl.loc[(model_psa_tl['index']=="r123")],
              errwidth=1.2, yerr="SE",
              palette=palatte_roi, order=roi_list)
ax1.set(ylabel='Pattern similarity effect (β)', xlabel='',
        title='Temporal memory precision', ylim=[-1,4.9])
ax1.axhline(y = 0,linewidth=1,linestyle = '--', color='#989898', zorder=0)
ax1.set_xticklabels(rotation =0, labels = roi_label)
sns.despine()
pointplot_err(x="ROI", y="Estimate", ax=ax2,
              data=model_psa_recog.loc[(model_psa_recog['index']=="mean_r")],
              yerr="SE", errwidth=1.2,
              palette=palatte_roi, order=roi_list)
ax2.set(ylabel='Pattern similarity effect (β)', xlabel='',
        title='Recognition confidence', ylim=[-.5,1.9])
ax2.axhline(y = 0,linewidth=1,linestyle = '--', color='#989898', zorder=0)
ax2.set_xticklabels(rotation = 0, labels = roi_label)
sns.despine()

### Similarity for each pair of exposures

###Figure 4a###
# temporal memory (high/low) ~ r12 + r23 + r13 + lag0 + lag1 + lag2 + lag3 + (1|subj)
model_psa_tl_by_exp = pd.DataFrame()
for i in roi_list:
    ds = betas_psa_tl.loc[betas_psa_tl['roi']==i].reset_index(drop=True)
    mod_logit = Lmer("mem ~ r12 + r23 + r13 + lag0 + lag1 + lag2 + lag3 + (1|subj)",
                         data=ds.dropna(), family="binomial")
    mod_logit.fit()
    t_logit = mod_logit.coefs[['Estimate', 'SE', 'P-val', 'Sig']][1:]
    t_logit['ROI'] = i
    model_psa_tl_by_exp = model_psa_tl_by_exp.append(t_logit)
model_psa_tl_by_exp = model_psa_tl_by_exp.reset_index()

# recog confidence (1-6) ~ r12 + r23 + r13 + lag0 + lag1 + lag2 + lag3 + (1|subj)
model_psa_recog_by_exp = pd.DataFrame()
for i in roi_list:
    ds = betas_psa_recog.loc[betas_psa_recog['roi']==i].reset_index(drop=True)
    mod = Lmer("recogconf ~ r12 + r23 + r13 + lag0 + lag1 + lag2 + lag3 + (1|subj)",
               data=ds.dropna())
    mod.fit()
    t = mod.coefs[['Estimate', 'SE', 'P-val', 'Sig']][1:]
    t['ROI'] = i
    model_psa_recog_by_exp = model_psa_recog_by_exp.append(t)
model_psa_recog_by_exp = model_psa_recog_by_exp.reset_index()

###Figure 4c###
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 1.8), sharey=False, sharex=False,gridspec_kw = {'width_ratios':[1, 1]})
f.subplots_adjust(wspace=0.35)
pointplot_err(x="index", y="Estimate", yerr="SE", errwidth=1.2, ax=ax1,
              data=model_psa_tl_by_exp.loc[(model_psa_tl_by_exp['ROI']=="CA1")], order=['r12','r23','r13'])
ax1.set(ylabel='Pattern similarity effect (β)', xlabel='', title='CA1', ylim=[-1,2.6])
ax1.axhline(y = 0,linewidth=1,linestyle = '--', color='#989898', zorder=0)
ax1.set_xticklabels(rotation = 0, labels = ["E1-E2", "E2-E3", "E1-E3"])
sns.despine()

pointplot_err(x="index", y="Estimate", yerr="SE", errwidth=1.2, ax=ax2,
              data=model_psa_tl_by_exp.loc[(model_psa_tl_by_exp['ROI']=="ERC")], order=['r12','r23','r13'])
ax2.set(ylabel='', xlabel='', title='ERC', ylim=[-1,2.6])
ax2.axhline(y = 0,linewidth=1,linestyle = '--', color='#989898', zorder=0)
ax2.set_xticklabels(rotation = 0, labels = ["E1-E2", "E2-E3", "E1-E3"])
sns.despine()

###Figure 4d###
f, ax1 = plt.subplots(1, 2, figsize=(2, 1.8))
pointplot_err(x="index", y="Estimate", yerr="SE", errwidth=1.2, ax=ax1,
              data=model_psa_recog_by_exp.loc[(model_psa_recog_by_exp['ROI']=="PHC")], order=['r12','r23','r13'])
ax1.set(ylabel='Pattern similarity effect (β)', xlabel='', title='PHC', ylim=[-.2, .8])
ax1.axhline(y = 0,linewidth=1,linestyle = '--', color='#989898', zorder=0)
ax1.set_xticklabels(rotation = 0, labels = ["E1-E2", "E2-E3", "E1-E3"])
sns.despine()

###Figure 5a###
model_item = pd.read_csv("rsa_item_specificity.csv")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 1.8), sharey=False, sharex=False,gridspec_kw = {'width_ratios':[1, 1]})
f.subplots_adjust(wspace=0.35)
n, bins, patches = ax1.hist(model_item.loc[model_item['roi']=="CA1", 'perm_beta'], bins=30, color = ["#969699"], alpha=.66, linewidth=0)
ax1.set(ylabel='\n# of permutations',xlabel='', title='CA1',ylim=[0,115],xlim=[None,2.1])
ax1.axvline(x = model_item.loc[(model_item['ROI']=="CA1")&(model_item['index']=="r12"), "Estimate"].values[0],
            ymax=.86, linewidth=1.5, color='#746297', linestyle = '--')
ax1.legend_ = None
ax1.set_xticks([-1, 0, 1])
sns.despine()
n, bins, patches = ax2.hist(model_item.loc[model_item['roi']=="ERC", 'perm_beta'], bins=30, color = ["#969699"], alpha=.66, linewidth=0)
ax2.set(ylabel='',xlabel='', title='ERC',ylim=[0,115])#,xlim=[1.3,2.1])
ax2.axvline(x = model_item.loc[(model_item['ROI']=="ERC")&(model_item['index']=="r12"), "Estimate"].values[0],
            ymax=.86, linewidth=1.5, color='#e9c46a', linestyle = '--')
ax2.legend_ = None
sns.despine()
f.text(0.5, -.066, 'Similarity effect of E1-E2 on temporal memory precision (β)', ha='center')