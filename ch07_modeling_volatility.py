import numpy as np
import pandas as pd

from . import black_merton_scholes as bms
from .hypotheses import moneyness, define_moneyness

from .plot_utils import (mpl, plt, mtick, mdates, gridspec,
                         set_plt_defaults, set_payoff_axes, set_time_axis, with_style)
  
def panel_prices(ax, mny, calls, puts, otm_calls, color='#666666', loc='best'):
    ax.plot(mny, calls, label='calls', color=color, linestyle='--')
    ax.plot(mny, puts, label='puts', color=color, linestyle='-', linewidth=0.5)
        
    M,otm = mny[otm_calls], calls[otm_calls]
    ax.plot(M, otm, linewidth=3, color=color, linestyle='--', label='OTM calls') # 

    M,otm = mny[~otm_calls], puts[~otm_calls]
    ax.plot(M, otm, linewidth=3, color=color, linestyle='-', label='OTM puts') # 

    ax.set_xlabel('Moneyness (%s)'%moneyness.mny_str)
    ax.set_ylabel('Price')
    ax.legend(loc=loc)

def figure_impvol(mny, IV, calls, puts, otm_calls, price_color='#666666', loc='upper center'):
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.plot(mny, 100*IV, linewidth=3)
    ax.set_xlabel('Moneyness (%s)'%moneyness.mny_str)
    ax.set_ylabel('Implied Volatility')
    ax.set_ylim([0, 60])

    axx = ax.twinx()
    panel_prices(axx, mny, calls, puts, otm_calls, color=price_color, loc=loc)
    return ax, axx
    
def example_symmetric_smile(mny,S,K,r,y,T,sigma0):
    atm = np.median(mny)
    IV = sigma0 + 1.25*(mny-atm)**2
    calls = bms.option_price(S, K, r, y, T, IV, is_call=True)
    puts = bms.option_price(S, K, r, y, T, IV, is_call=False)
    assert np.all(calls == np.sort(calls)[::-1]), repr(calls)
    assert np.all(puts == np.sort(puts)), repr(puts)
    return calls, puts

def example_asymmetric_smile(mny,S,K,r,y,T,sigma0):
    atm = np.median(mny)
    IV = sigma0 + 1.25*(mny-atm)**2 - 0.5*(mny-atm)  
    calls = bms.option_price(S, K, r, y, T, IV, is_call=True)
    puts = bms.option_price(S, K, r, y, T, IV, is_call=False)
    assert np.all(calls == np.sort(calls)[::-1]), repr(calls)
    assert np.all(puts == np.sort(puts)), repr(puts)
    return calls, puts

__smiles = np.genfromtxt('ch07_smiles.csv', delimiter=',')

def fig_no_smile(S, r, y, T, sigma):
    if S==100 and r==0.05 and y==0 and T==0.25 and sigma==0.2:
        K = __smiles[0,:]   
    else:
        mny,K0 = moneyness.range(S,0.70,0.99,0.01)
        mny,K1 = moneyness.range(S,0.99,1.01,0.001)
        mny,K2 = moneyness.range(S,1.01,1.301,0.01)
        K = np.sort(np.array(list(set(K0).union(K1).union(K2))))            
    mny = moneyness(S,K)
        
    calls = bms.option_price(S, K, r, y, T, sigma, is_call=True)
    puts = bms.option_price(S, K, r, y, T, sigma, is_call=False)
    
    # Use the OTM option prices
    otm_calls = mny > 1
    opt_price = np.where(otm_calls, calls, puts)
    
    # And invert the BMS price function around these prices
    IV = bms.implied_volatility(opt_price, S, K, r, y, T, otm_calls)
    
    ax,axx = figure_impvol(mny, IV, calls, puts, otm_calls)
    ax.set_ylim([0,50])
    return K, calls, puts

def fig_with_smile(S, K, r, y, T, sigma, smile):
    if S==100 and r==0.05 and y==0 and T==0.25 and sigma==0.2:
        assert np.all(K==__smiles[0,:]) 
        mny = moneyness(S,K)
        if smile=='symmetric':
            IV = __smiles[1,:]
        elif smile=='asymmetric':
            IV = __smiles[2,:]
        calls = bms.option_price(S, K, r, y, T, IV, True)
        puts = bms.option_price(S, K, r, y, T, IV, False)
        #import pdb; pdb.set_trace()
    else:
        mny,K0 = moneyness.range(S,0.70,0.99,0.01)
        mny,K1 = moneyness.range(S,0.99,1.01,0.001)
        mny,K2 = moneyness.range(S,1.01,1.301,0.01)
        K = np.sort(np.array(list(set(K0).union(K1).union(K2))))            
        mny = moneyness(S,K)
        calls, puts = smile(mny, S,K,r,y,T,sigma+0.025)

    otm_calls = mny > 1
    price = np.where(otm_calls, calls, puts)

    IV = bms.implied_volatility(price, S, K, r, y, T, otm_calls)
    ax,axx = figure_impvol(mny, IV, calls, puts, otm_calls, price_color='#444444', loc='upper center')
    ax.axhline(100*sigma, color='k', linewidth=0.5)
    ax.set_ylim([0,50])

    light = '#AAAAAA'
    opt_price = lambda is_call: bms.option_price(S,K,r,y,T,sigma,is_call=is_call)
    axx.plot(mny, opt_price(True), color=light, linestyle='--')
    axx.plot(mny, opt_price(False), color=light, linestyle='-', linewidth=0.5)

    return K, calls, puts

fig_symmetric_smile = lambda S, K, r, y, T, sigma: \
                      fig_with_smile(S, K, r, y, T, sigma, 'symmetric')

fig_asymmetric_smile = lambda S, K, r, y, T, sigma: \
                       fig_with_smile(S, K, r, y, T, sigma, 'asymmetric')

def BL78_risk_neutral_density(exp_rT, strikes, call_prices):
    d_strikes = strikes[1:] - strikes[:-1]
    dK = np.mean(d_strikes)
    assert np.max(np.abs(dK - d_strikes)) < 1e-6, 'dK is assumed to be unique'

    b_0 = call_prices[2:] - 2*call_prices[1:-1] + call_prices[:-2]
    return exp_rT*b_0 / (dK**2)     

def risk_neutral_density(call_prices, S, K, r, y, T):
    from scipy.interpolate import UnivariateSpline
    IV = bms.implied_volatility(call_prices, S, K, r, y, T, is_call=True)
    smile = UnivariateSpline(K, IV, ext=3)

    strikes = np.linspace(np.min(K), np.max(K), max(len(K),1000))    
    call_prices = bms.option_price(S, strikes, r, y, T, smile(strikes), is_call=True)
    return strikes[1:-1], BL78_risk_neutral_density(np.exp(r*T), strikes, call_prices)

def fig_extrapolate_sym_rnd():
    pass



