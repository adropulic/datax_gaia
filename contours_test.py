#!/usr/bin/env python
# coding: utf-8

# In[2]:


from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.optimize as so
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import LogNorm
import TransformCoords
import argparse, ast
#import tensorflow as tf
#from tensorflow.keras import initializers
#import keras
#from keras import backend as K

matplotlib.rcParams.update({'font.family':'cmr10','font.size': 13})
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams['axes.labelsize']=15
plt.rcParams['figure.figsize']=(4,4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
import multiprocessing as mp
#parser = argparse.ArgumentParser()
#parser.add_argument("--nnodes", nargs="+", default = 30, type=int)
#parser.add_argument("--ncores", action="store", dest="ncores", default=16, type=int)
#results = parser.parse_args()
#ncores = results.ncores
#nnodes = results.nnodes
#print(ncores)
#config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=ncores, inter_op_parallelism_threads=ncores,  allow_soft_placement=True,device_count = {'CPU': ncores})
#session = tf.compat.v1.Session(config=config)
#K.set_session(session)
# In[2]:


def find_confidence_interval(x, pdf, confidence_level, area, sigma_string):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, xbins2d,ybins2d, ax=None,pred = True, MC = False, **contour_kwargs):
    
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=[xbins2d,ybins2d], range = [[xbins2d.min(),xbins2d.max()],[ybins2d.min(),ybins2d.max()]],density=True)
    xcenters = (xedges[1:]+ xedges[:-1])/2
    ycenters = (yedges[1:]+ yedges[:-1])/2
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,len(xcenters)))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((len(ycenters),1))
    area = (x_bin_sizes*y_bin_sizes)
    #print(area)
    pdf = (H*(x_bin_sizes*y_bin_sizes))
    # 0.39346934, 0.67534753, 0.86466472
    low_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.15, area,'low_sigma'))
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.39, area,'one_sigma'))
    med_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.66, area,'med_sigma'))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.86, area,'two_sigma'))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99, area,'three_sigma'))
    levels = [three_sigma, two_sigma, med_sigma, one_sigma, low_sigma]
    levels_plot = [three_sigma, two_sigma,one_sigma]
    #print(levels)
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T
    print(np.shape(Z))

    if ax == None and MC == False:
        contour = plt.contour(X, Y, Z, levels=levels_plot, origin="lower", **contour_kwargs)
    else:
        if pred == True and MC == False:
            contour = ax.contour(X, Y, Z, levels=levels_plot, origin="lower", colors = ['maroon','red','salmon'],  **contour_kwargs)
            #p1 = contour.collections[0].get_paths()
            #coor_p1 = p1[-1].vertices
        if pred == False and MC == False:
            contour = ax.contour(X, Y, Z, levels=levels_plot, origin="lower", colors = ['darkblue','blue','deepskyblue'],  **contour_kwargs)
    if MC == False: return levels, Z
    if MC == True: return levels, Z

def density_contour_MC(H, xedges, yedges, xbins2d,ybins2d, ax=None,pred = True, MC = False, **contour_kwargs):
    # H must be a histogram density
    #H must be not flipped or rotated!
    xcenters = (xedges[1:]+ xedges[:-1])/2
    ycenters = (yedges[1:]+ yedges[:-1])/2
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,len(xcenters)))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((len(ycenters),1))
    area = (x_bin_sizes*y_bin_sizes)
    #print(area)
    pdf = (H*(x_bin_sizes*y_bin_sizes))
    # 0.39346934, 0.67534753, 0.86466472
    low_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.15, area,'low_sigma'))
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.39, area,'one_sigma'))
    med_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.66, area,'med_sigma'))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.86, area,'two_sigma'))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99, area,'three_sigma'))
    levels = [three_sigma, two_sigma, med_sigma, one_sigma, low_sigma]
    levels_plot = [three_sigma, two_sigma,one_sigma]
    #print(levels)
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    if ax == None and MC == False:
        contour = plt.contour(X, Y, Z, levels=levels_plot, origin="lower", **contour_kwargs)
    else:
        if pred == True and MC == False:
            contour = ax.contour(X, Y, Z, levels=levels_plot, origin="lower", colors = ['maroon','red','salmon'],  **contour_kwargs)
        if pred == False and MC == False:
            contour = ax.contour(X, Y, Z, levels=levels_plot, origin="lower", colors = ['darkblue','blue','deepskyblue'],  **contour_kwargs)
    if MC == True: return levels, Z
    elif MC == False: return levels, Z
    


# In[3]:


folder = '/tigress/dropulic/G_train_2it_10000000_tanh_D30_nodropout_seed1test1_lbppp_log2d_10M'
print(folder)
ellipse_df = np.load(folder+'/ellipse_df_wmet.npz')
spec = ""
print(spec)
ellipse_df = ellipse_df['arr_0']
data_cols_vel_ellipse = ['vr_pred','vth_pred','vphi_pred','sigma_vr','sigma_vth','sigma_vphi','vlos_pred','sigma_los','vr_true','vth_true','vphi_true', 'vlos_true','feh','z','l', 'b', 'ra', 'dec', 'parallax','pmra','pmdec']
ellipse_df = pd.DataFrame(ellipse_df, columns=data_cols_vel_ellipse)


# In[4]:


import sys
import cv2
import math
from scipy.interpolate import interp2d, interp1d
from shapely.geometry import Polygon
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
def plot_contours(ellipse_df):
    from skimage.draw import polygon
    import numpy.ma as ma
    idx_lowmet = np.where((ellipse_df.feh <= -1.5))
    ellipse_df_lowmet = ellipse_df.loc[idx_lowmet]
    print(ellipse_df_lowmet.shape)

    idx_highmet = np.where((ellipse_df.feh > -1))
    ellipse_df_highmet = ellipse_df.loc[idx_highmet]
    print(ellipse_df_highmet.shape)

    vrthbins = np.linspace(-250,250,31)
    vphibins = np.linspace(-450,250,31)
    bin_area = (vrthbins[1]-vrthbins[0])*(vphibins[1]-vphibins[0])

    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False,figsize=(20,15))    

    ratio_highmet_vrvphi = percent_in_prob_volume(ellipse_df_highmet,vrthbins,vphibins,'vr', 'vphi',ax,0,"high")
    plot_roc(ratio_highmet_vrvphi,ax, 0,"high")
    
    ratio_lowmet_vrvphi = percent_in_prob_volume(ellipse_df_lowmet,vrthbins,vphibins,'vr', 'vphi',ax,0,"low")
    plot_roc(ratio_lowmet_vrvphi,ax, 0,"low")
    
    ratio_highmet_vrvth = percent_in_prob_volume(ellipse_df_highmet,vrthbins,vrthbins,'vr', 'vth',ax,1,"high")
    plot_roc(ratio_highmet_vrvth,ax, 1,"high")
    
    ratio_lowmet_vrvth = percent_in_prob_volume(ellipse_df_lowmet,vrthbins,vrthbins,'vr', 'vth',ax,1,"low")
    plot_roc(ratio_lowmet_vrvth,ax, 1,"low")
    
    ratio_highmet_vthvphi = percent_in_prob_volume(ellipse_df_highmet,vrthbins,vphibins,'vth', 'vphi',ax,2,"high")
    plot_roc(ratio_highmet_vthvphi,ax, 2,"high")
    
    ratio_lowmet_vthvphi = percent_in_prob_volume(ellipse_df_lowmet,vrthbins,vphibins,'vth', 'vphi',ax,2,"low")
    plot_roc(ratio_lowmet_vthvphi,ax, 2,"low")
    
    ax[0,0].set_title("High [Fe/H] ($> -1$), "+ str(ellipse_df_highmet.shape[0])+" stars")
    ax[0,1].set_title("High [Fe/H] ($> -1$), "+ str(ellipse_df_highmet.shape[0])+" stars")
    ax[0,2].set_title("Low [Fe/H] ($< -1.5$), "+ str(ellipse_df_lowmet.shape[0])+" stars")
    ax[0,3].set_title("Low [Fe/H] ($< -1.5$), "+ str(ellipse_df_lowmet.shape[0])+" stars")

    red_patch = mpatches.Patch(color='red', label='Predicted')
    blue_patch = mpatches.Patch(color='blue', label='Truth')
    ax[0,0].legend(handles=[red_patch,blue_patch], ncol = 1)
    fig.savefig(folder+'/countors_roc_'+spec+'.png')


# In[5]:


def plot_contours_MC(ellipse_df):
    vrthbins = np.linspace(-250,250,31)
    vphibins = np.linspace(-450,250,31)
    z_idx = 2.52
    idx_lowmet = np.where((ellipse_df.feh <= -1.5))
    ellipse_df_lowmet = ellipse_df.loc[idx_lowmet]
    print(ellipse_df_lowmet.shape)
    
    idx_highmet = np.where((ellipse_df.feh > -1))
    ellipse_df_highmet = ellipse_df.loc[idx_highmet]
    print(ellipse_df_highmet.shape)
    
    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False,figsize=(20,15))   
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    
    vr_vphi_lowmet, vr_vth_lowmet, vth_vphi_lowmet = monte_carlo(ellipse_df_lowmet, lowmet = True)
    vr_vphi_highmet, vr_vth_highmet, vth_vphi_highmet = monte_carlo(ellipse_df_highmet, lowmet = False)
    
    ratio_highmet_vrvphi,ratio_highmet_vrvphi_MClist = percent_in_prob_volume_MC(ellipse_df_highmet,vrthbins,vphibins,'vr', 'vphi',ax,0,"high",vr_vphi_highmet )
    plot_roc_MC(ratio_highmet_vrvphi,ratio_highmet_vrvphi_MClist,ax, 0,"high")
    
    ratio_lowmet_vrvphi, ratio_lowmet_vrvphi_MClist = percent_in_prob_volume_MC(ellipse_df_lowmet,vrthbins,vphibins,'vr', 'vphi',ax,0,"low",vr_vphi_lowmet)
    plot_roc_MC(ratio_lowmet_vrvphi,ratio_lowmet_vrvphi_MClist,ax, 0,"low")
    
    ratio_highmet_vrvth, ratio_highmet_vrvth_MClist = percent_in_prob_volume_MC(ellipse_df_highmet,vrthbins,vrthbins,'vr', 'vth',ax,1,"high",vr_vth_highmet)
    plot_roc_MC(ratio_highmet_vrvth,ratio_highmet_vrvth_MClist,ax, 1,"high")
    
    ratio_lowmet_vrvth, ratio_lowmet_vrvth_MClist = percent_in_prob_volume_MC(ellipse_df_lowmet,vrthbins,vrthbins,'vr', 'vth',ax,1,"low",vr_vth_lowmet)
    plot_roc_MC(ratio_lowmet_vrvth,ratio_lowmet_vrvth_MClist,ax, 1,"low")
    
    ratio_highmet_vthvphi, ratio_highmet_vthvphi_MClist = percent_in_prob_volume_MC(ellipse_df_highmet,vrthbins,vphibins,'vth', 'vphi',ax,2,"high",vth_vphi_highmet)
    plot_roc_MC(ratio_highmet_vthvphi,ratio_highmet_vthvphi_MClist,ax, 2,"high")
    
    ratio_lowmet_vthvphi,ratio_lowmet_vthvphi_MClist = percent_in_prob_volume_MC(ellipse_df_lowmet,vrthbins,vphibins,'vth', 'vphi',ax,2,"low",vth_vphi_lowmet)
    plot_roc_MC(ratio_lowmet_vthvphi,ratio_lowmet_vthvphi_MClist,ax, 2,"low")
    
    ax[0,0].set_title("High [Fe/H] ($> -1$), "+ str(ellipse_df_highmet.shape[0])+" stars")
    ax[0,1].set_title("High [Fe/H] ($> -1$), "+ str(ellipse_df_highmet.shape[0])+" stars")
    ax[0,2].set_title("Low [Fe/H] ($< -1.5$), "+ str(ellipse_df_lowmet.shape[0])+" stars")
    ax[0,3].set_title("Low [Fe/H] ($< -1.5$), "+ str(ellipse_df_lowmet.shape[0])+" stars")

    red_patch = mpatches.Patch(color='red', label='Predicted')
    blue_patch = mpatches.Patch(color='blue', label='Truth')
    ax[0,0].legend(handles=[red_patch,blue_patch], ncol = 1)
    fig.savefig(folder+'/MC_countors_roc_band_all_50MC_50stars_'+spec+'.png')


# In[6]:


def percent_in_prob_volume_MC(df,xbins2d,ybins2d,xstr, ystr,ax,ax_x, met_string, MC_array):
    ## FOR MONTE-CARLO
    if met_string == "high": ax_y = 0
    if met_string == "low": ax_y = 2
    global contour_interp_func 
    h1_test, xedges, yedges = np.histogram2d(df[xstr+'_true'].values,df[ystr+'_true'].values, bins=[xbins2d,ybins2d],range=[[xbins2d.min(),xbins2d.max()],[ybins2d.min(),ybins2d.max()]], density = True)
    #probably need to do this for each MC in MC_array and then return list of lists for ratio
    #MC_array = np.mean(MC_array, axis = 0)
    #MC array must not be flipped or rotated before going into density_contour
    ratio_MC_list = []
    for MC_i in MC_array:
        dca, Za = density_contour_MC(MC_i,xedges, yedges, xbins2d,ybins2d, ax=ax[ax_x,ax_y], pred = True, MC = True)
        MC_i = np.rot90(MC_i)  # rotate
        MC_i = np.flipud(MC_i) # flip
        X, Y = np.meshgrid(xedges, yedges)
        dcb, Zb = density_contour(df[xstr+'_true'].values,df[ystr+'_true'].values, xbins2d,ybins2d, ax=ax[ax_x, ax_y], pred = False, MC = True)
        num_stars = len(df[xstr+'_pred'].values)
        ratio = []
        for level_i in dca:
            print("level_i", level_i)
            h1_test_bool = np.zeros_like(Za)
            for i in range(np.shape(Za)[0]):
                for j in range(np.shape(Za)[1]):
                    if Za[i][j] >= level_i:
                        h1_test_bool[i][j] = 1
            #if level_i == dca[0]: ax[ax_x, ax_y].pcolormesh(X, Y,h1_test_bool, norm = LogNorm(), alpha = 0.5)
            xbins2d_centers = (xbins2d[1:]+xbins2d[:-1])/2
            ybins2d_centers = (ybins2d[1:]+ybins2d[:-1])/2
            contour_function = interp2d(xbins2d_centers,ybins2d_centers,h1_test_bool,kind='linear')
            num_true_in_pred = 0
            num_pred_in_pred = 0
            ratio_i = 0
            interp_val_true = [np.round(contour_function(star_x,star_y)[0]) for star_x,star_y in zip(df[xstr+'_true'].values,df[ystr+'_true'].values)]
            num_true_in_pred = sum(interp_val_true)
            ratio_i = num_true_in_pred/num_stars
            print(ratio_i)
            ratio.append(ratio_i)
        ratio_MC_list.append(ratio)
        
        
        
    MC_array = np.mean(MC_array, axis = 0)
    dca, Za = density_contour_MC(MC_array,xedges, yedges, xbins2d,ybins2d, ax=ax[ax_x,ax_y], pred = True, MC = False)
    MC_array = np.rot90(MC_array)  # rotate
    MC_array = np.flipud(MC_array) # flip
    X, Y = np.meshgrid(xedges, yedges)
    pc = ax[ax_x,ax_y].pcolormesh(X, Y,MC_array, norm = LogNorm(), alpha = 0.5, cmap = "inferno")    

    dcb, Zb = density_contour(df[xstr+'_true'].values,df[ystr+'_true'].values, xbins2d,ybins2d, ax=ax[ax_x, ax_y], pred = False)
    num_stars = len(df[xstr+'_pred'].values)
    ratio_mean = []
#    for level_i in dca:
#        print("level_i", level_i)
#        h1_test_bool = np.zeros_like(Za)
#        for i in range(np.shape(Za)[0]):
#            for j in range(np.shape(Za)[1]):
#                if Za[i][j] >= level_i:
#                    h1_test_bool[i][j] = 1
#        #if level_i == dca[0]: ax[ax_x, ax_y].pcolormesh(X, Y,h1_test_bool, norm = LogNorm(), alpha = 0.5)
#        xbins2d_centers = (xbins2d[1:]+xbins2d[:-1])/2
#        ybins2d_centers = (ybins2d[1:]+ybins2d[:-1])/2
#        contour_function = interp2d(xbins2d_centers,ybins2d_centers,h1_test_bool,kind='linear')
#        num_true_in_pred = 0
#        num_pred_in_pred = 0
#        ratio_i = 0
#        for star_i in tqdm(range(len(df['vr_true'].values))):
#            interp_val_true = np.round(contour_function(df[xstr+'_true'].values[star_i],df[ystr+'_true'].values[star_i])[0])
#            interp_val_pred = np.round(contour_function(df[xstr+'_pred'].values[star_i],df[ystr+'_pred'].values[star_i])[0])
#            if interp_val_true == 1.0:
#                num_true_in_pred =  num_true_in_pred + 1
#            if interp_val_pred == 1.0:
#                num_pred_in_pred =  num_pred_in_pred + 1
#        ratio_i = num_true_in_pred/num_stars
#        print(ratio_i)
#        ratio_mean.append(ratio_i)
    ax[ax_x,0].set_xlim(xbins2d.min(),xbins2d.max());
    ax[ax_x,0].set_ylim(ybins2d.min(),ybins2d.max());
    if xstr == "vr": ax[ax_x,0].set_xlabel(r'$v_{r}$')
    if xstr == "vth": ax[ax_x,0].set_xlabel(r'$v_{\Theta}$')
    if ystr == "vphi": ax[ax_x,0].set_ylabel('$v_{\phi}$', rotation = 360)
    if ystr == "vth": ax[ax_x,0].set_ylabel('$v_{\Theta}$', rotation = 360)
        
        
        
    return ratio_mean, ratio_MC_list


# In[7]:


def percent_in_prob_volume(df,xbins2d,ybins2d,xstr, ystr,ax,ax_x, met_string):
    if met_string == "high": ax_y = 0
    if met_string == "low": ax_y = 2
    h1_test, xedges, yedges = np.histogram2d(df[xstr+'_true'].values,df[ystr+'_true'].values, bins=[xbins2d,ybins2d],range=[[xbins2d.min(),xbins2d.max()],[ybins2d.min(),ybins2d.max()]], density = True)
    h1_test = np.rot90(h1_test)  # rotate
    h1_test = np.flipud(h1_test) # flip
    X, Y = np.meshgrid(xedges, yedges)
    pc = ax[ax_x,ax_y].pcolormesh(X, Y,h1_test, norm = LogNorm(), alpha = 0.5)
    dca, Za = density_contour(df[xstr+'_pred'].values,df[ystr+'_pred'].values,xbins2d,ybins2d, ax=ax[ax_x, ax_y], pred = True)
    dcb, Zb = density_contour(df[xstr+'_true'].values,df[ystr+'_true'].values, xbins2d,ybins2d, ax=ax[ax_x, ax_y], pred = False)
    num_stars = len(df[xstr+'_pred'].values)
    ratio = []
    for level_i in dca:
        print("level_i", level_i)
        h1_test_bool = np.zeros_like(Za)
        for i in range(np.shape(Za)[0]):
            for j in range(np.shape(Za)[1]):
                if Za[i][j] >= level_i:
                    h1_test_bool[i][j] = 1
        #ax[ax_x, ax_y].pcolormesh(X, Y,h1_test_bool, norm = LogNorm(), alpha = 0.5)
        xbins2d_centers = (xbins2d[1:]+xbins2d[:-1])/2
        ybins2d_centers = (ybins2d[1:]+ybins2d[:-1])/2
        contour_function = interp2d(xbins2d_centers,ybins2d_centers,h1_test_bool,kind='linear')
        num_true_in_pred = 0
        num_pred_in_pred = 0
        ratio_i = 0
        for star_i in tqdm(range(len(df['vr_true'].values))):
            interp_val_true = np.round(contour_function(df[xstr+'_true'].values[star_i],df[ystr+'_true'].values[star_i])[0])
            interp_val_pred = np.round(contour_function(df[xstr+'_pred'].values[star_i],df[ystr+'_pred'].values[star_i])[0])
            if interp_val_true == 1.0:
                num_true_in_pred =  num_true_in_pred + 1
            if interp_val_pred == 1.0:
                num_pred_in_pred =  num_pred_in_pred + 1
        ratio_i = num_true_in_pred/num_stars
        print(ratio_i)
        ratio.append(ratio_i)
    ax[ax_x,0].set_xlim(xbins2d.min(),xbins2d.max());
    ax[ax_x,0].set_ylim(ybins2d.min(),ybins2d.max());
    if xstr == "vr": ax[ax_x,0].set_xlabel(r'$v_{r}$')
    if xstr == "vth": ax[ax_x,0].set_xlabel(r'$v_{\Theta}$')
    if ystr == "vphi": ax[ax_x,0].set_ylabel('$v_{\phi}$', rotation = 360)
    if ystr == "vth": ax[ax_x,0].set_ylabel('$v_{\Theta}$', rotation = 360)
    return ratio


# In[8]:


def plot_roc(ratio_list,ax, ax_x, met_string):
    if met_string == "high": ax_y = 1
    if met_string == "low": ax_y = 3
    true_prob_array = [0.99,0.86,0.66,0.39, 0.15, 0.0]
    ratio_list.append(0.0)
    ax[ax_x,ax_y].scatter(true_prob_array, ratio_list, color = "lightgreen")
    x_calib = np.linspace(0,1,20)
    y_calib = np.linspace(0,1,20)
    ax[ax_x,ax_y].plot(x_calib, y_calib, linestyle = '--', color = "black")
    f = interp1d(true_prob_array, ratio_list, kind = "cubic")
    xnew = np.linspace(0,.99,100)
    if ax_x == 0: label_string = "_vr_vphi_"+spec+"_"
    if ax_x == 1: label_string = "_vr_vth_"+spec+"_"
    if ax_x == 2:label_string = "_vth_vphi_"+spec+"_"
    np.savez(folder+'/y_i_list'+label_string+met_string+'notMC.npz',f(xnew[1:]))
    ax[ax_x,ax_y].plot(xnew[1:], f(xnew[1:]), color = "green")
    ax[ax_x,ax_y].set_xlabel("Percentage of Probability Volume", fontsize=10)
    ax[ax_x,ax_y].set_ylabel("Percent of Stars with True Value in the Volume", fontsize=10)


# In[9]:


def plot_roc_MC(ratio_list_avg,MC_ratio_list, ax, ax_x, met_string):
    if met_string == "high": ax_y = 1
    if met_string == "low": ax_y = 3
    true_prob_array = [0.99,0.86,0.66,0.39, 0.15, 0.0]
    y_i_list = []
    for list_i in MC_ratio_list:
        list_i.append(0.0)
        f_i = interp1d(true_prob_array, list_i, kind = "cubic")
        xnew_i = np.linspace(0,.99,100)
        y_i = f_i(xnew_i[1:])
        #ax[ax_x,ax_y].plot(xnew_i[1:], f_i(xnew_i[1:]))
        y_i_list.append(y_i)
    xnew = np.linspace(0,.99,100)
    min_y_i_list = np.min(y_i_list, axis = 0)
    mean_y_i_list = np.mean(y_i_list, axis = 0)
    max_y_i_list = np.max(y_i_list, axis = 0)
    if ax_x == 0: label_string = "_vr_vphi_"+spec+"_" 
    if ax_x == 1: label_string = "_vr_vth_"+spec+"_"
    if ax_x == 2:label_string = "_vth_vphi_"+spec+"_"
    np.savez(folder+'/min_y_i_list'+label_string+met_string+'.npz',min_y_i_list)
    np.savez(folder+'/max_y_i_list'+label_string+met_string+'.npz',max_y_i_list)
    np.savez(folder+'/mean_y_i_list'+label_string+met_string+'.npz',mean_y_i_list)
    ax[ax_x,ax_y].fill_between(xnew[1:], min_y_i_list, max_y_i_list, where=max_y_i_list>=min_y_i_list, alpha = 0.5, color = "mediumaquamarine")
    ratio_list_avg.append(0.0)
    #ax[ax_x,ax_y].scatter(true_prob_array, ratio_list_avg, color = "lightgreen")
    x_calib = np.linspace(0,1,20)
    y_calib = np.linspace(0,1,20)
    ax[ax_x,ax_y].plot(x_calib, y_calib, linestyle = '--', color = "black")
    #f = interp1d(true_prob_array, ratio_list_avg, kind = "cubic")
    #ax[ax_x,ax_y].plot(xnew[1:], f(xnew[1:]), color = "green")
    ax[ax_x,ax_y].plot(xnew[1:], mean_y_i_list, color = "green")
    ax[ax_x,ax_y].set_xlabel("Percentage of Probability Volume", fontsize=10)
    ax[ax_x,ax_y].set_ylabel("Percent of Stars with True Value in the Volume", fontsize=10)


# In[10]:


def monte_carlo(ellipse_df_full, lowmet = False):
    from tqdm import tqdm
    from matplotlib.colors import LogNorm
    global mc_i_func
    num_MC = 10
    y_low = -250
    y_high = 250
    mc_vr_pred_list = []
    mc_pred_list_vr = []
    mc_pred_list_vth = []
    mc_pred_list_vphi = []
    resample_test_list = []
    bin_values_list = []
    min_array = []
    max_array = []
    min_array_r = []
    max_array_r = []
    min_array_th = []
    max_array_th = []
    min_array_phi = []
    max_array_phi = []

    hb_list = []
    hist_list_vr_vphi = []
    hist_list_vr_vth = []
    hist_list_vth_vphi = []
    hex_centers = []

    bin_values_list_r = []
    bin_values_list_th = []
    bin_values_list_phi = []

    N = len((ellipse_df_full['vlos_pred']).values)
    x_range = np.linspace(y_low,y_high,N)
    test_preds =ellipse_df_full[['vlos_pred', 'sigma_los']].to_numpy()
    if lowmet == True: num_stars_draw = 50
    else: num_stars_draw = 1
    indices_mc = np.repeat(ellipse_df_full.index, num_stars_draw)
    ellipse_df_full = ellipse_df_full.loc[indices_mc, data_cols_vel_ellipse]
    ellipse_df_full = ellipse_df_full.reset_index(drop=True)
    #for mc_i in tqdm(range(0,10)):
    def mc_i_func():
        mc_vr_pred = []
        for star_i in range(0,len(test_preds)):
            mc_vr_pred.append(np.random.normal(test_preds[star_i,0],test_preds[star_i,1],num_stars_draw))
            
        mc_vr_pred = np.reshape(mc_vr_pred,(1,-1))
        mc_vr_pred_list.append(mc_vr_pred)
        n, bins = np.histogram(mc_vr_pred,bins=50,range=(y_low,y_high), density = True)
        n_test_preds, bins_test_preds = np.histogram((ellipse_df_full['vlos_true']).values, bins=50, range=(y_low,y_high))

        plt.figure(2)
        #hb = plt.hexbin((ellipse_df_full['vlos_true']).values, mc_vr_pred,gridsize=80, norm = LogNorm(),extent=[-200, 200, -200, 200]);
        #hb_list.append(hb.get_array());
        #bin_values_list.append(n)

        #now for the coordinate-transformed histograms
        #need to increase number of rows to equal increased number of stars drawn 
        
        vel_sph_coord = get_coord_transform(ellipse_df_full, np.array(mc_vr_pred).flatten().astype('float'))
        n_r , bins_r = np.histogram(vel_sph_coord[:,0], bins=50, range=(-250,250), density = True)
        n_th , bins_th = np.histogram(vel_sph_coord[:,1], bins=50, range=(-250,250), density = True)
        n_phi , bins_phi = np.histogram(vel_sph_coord[:,2], bins=50, range=(-450,0), density = True)
        bin_values_list_r.append(n_r)
        bin_values_list_th.append(n_th)
        bin_values_list_phi.append(n_phi)
        mc_pred_list_vr.append(vel_sph_coord[:,0])
        mc_pred_list_vth.append(vel_sph_coord[:,1])
        mc_pred_list_vphi.append(vel_sph_coord[:,2])

        hist_vr_vphi, xedges_vr_vphi ,yedges_vr_vphi = np.histogram2d(vel_sph_coord[:,0],vel_sph_coord[:,2],bins = (30,30) ,range=[[-250, 250],[-450, 250]], density = True);
        #hist_list_vr_vphi.append(hist_vr_vphi);


        hist_vr_vth, xedges_vr_vth,yedges_vr_vth = np.histogram2d(vel_sph_coord[:,0], vel_sph_coord[:,1],bins = (30,30),range =[[-250, 250],[-250, 250]], density = True);
        #hist_list_vr_vth.append(hist_vr_vth);


        hist_vth_vphi, xedges_vth_vphi,yedges_vth_vphi = np.histogram2d(vel_sph_coord[:,1], vel_sph_coord[:,2],bins = (30,30),range=[[-250, 250], [-450, 250]], density = True);
        #hist_list_vth_vphi.append(hist_vth_vphi);
        plt.close(2)
        return np.array([hist_vr_vphi, hist_vr_vth, hist_vth_vphi])
    #for mc_i in tqdm(range(0,10)):
    pool = mp.Pool(mp.cpu_count()-1)
    print("cpu count",mp.cpu_count())
    results = [pool.apply_async(mc_i_func) for _ in tqdm(range(0,num_MC))]
    print("shape results ", np.shape(results))
    pool.close()
    res = [f.get() for f in tqdm(results)]
    print(np.shape(list(res[0])))
    
    print("res shape",np.shape(res))
    for mc_i in range(0,num_MC):
        hist_list_vr_vphi.append(res[mc_i][0])
        hist_list_vr_vth.append(res[mc_i][1])
        hist_list_vth_vphi.append(res[mc_i][2])
    
    return hist_list_vr_vphi, hist_list_vr_vth, hist_list_vth_vphi


# In[11]:


def get_coord_transform(df, train_preds):
    #needs only vr values of train_preds (maybe...need to see what to do about error)

    v_LSR = [11.1, 239.08, 7.25]
    r_LSR = [-8.,0.,0.015]

    sin_theta_gc, sin_phi_gc ,cos_theta_gc, cos_phi_gc= TransformCoords.calc_theta_phi(np.deg2rad(df['ra'].values),np.deg2rad(df['dec'].values),np.deg2rad(df['b'].values),np.deg2rad(df['l'].values), df['parallax'].values ,train_preds.astype(np.float32))
    vr_gc, vth_gc, vphi_gc = TransformCoords.cart_to_galcen(np.deg2rad(df['ra'].values),np.deg2rad(df['dec'].values), np.deg2rad(df['b'].values) ,np.deg2rad(df['l'].values), df['parallax'].values,train_preds.astype(np.float32), df['pmra'].values, df['pmdec'].values,sin_theta_gc, sin_phi_gc, cos_theta_gc, cos_phi_gc)

    vels_sph_pred_train = np.array([vr_gc, vth_gc, vphi_gc]).T
    return vels_sph_pred_train


# In[ ]:


plot_contours_MC(ellipse_df)


# In[ ]:


#plot_contours(ellipse_df)

