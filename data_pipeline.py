"""
SPY Options Data Pipeline
Fetches option chain data from Yahoo Finance, extracts forwards via put-call parity,
and transforms to SVI inputs (log-moneyness k, total variance w).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


# --- Black's Model (Forward) ---

def black_price(F, K, T, sigma, r, opt='call'):
    """
    European option price under Black's model.
    C = exp(-rT) * [F*N(d1) - K*N(d2)]
    P = exp(-rT) * [K*N(-d2) - F*N(-d1)]
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0) if opt == 'call' else max(K - F, 0)
        return np.exp(-r * T) * intrinsic
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    df = np.exp(-r * T)
    
    if opt == 'call':
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, r, opt='call', bounds=(0.001, 5.0)):
    """Back out IV from option price via Brent's method using Black's model."""
    if T <= 0:
        return None
    
    df = np.exp(-r * T)
    
    # Arbitrage bounds
    intrinsic = df * max(F - K, 0) if opt == 'call' else df * max(K - F, 0)
    if price < intrinsic - 0.01:
        return None
    
    max_price = df * F if opt == 'call' else df * K
    if price > max_price:
        return None
    
    def obj(sigma):
        return black_price(F, K, T, sigma, r, opt) - price
    
    try:
        if obj(bounds[0]) * obj(bounds[1]) > 0:
            return None
        return brentq(obj, bounds[0], bounds[1], xtol=1e-6)
    except (ValueError, RuntimeError):
        return None


# --- Forward Extraction ---

def extract_forward(calls, puts, S, T, r):
    """Extract forward price via put-call parity: F = K + exp(rT) * (C - P)"""
    calls_sub = calls[['strike', 'mid', 'spread_pct', 'openInterest']].copy()
    calls_sub.columns = ['strike', 'call_mid', 'call_spread', 'call_oi']
    
    puts_sub = puts[['strike', 'mid', 'spread_pct', 'openInterest']].copy()
    puts_sub.columns = ['strike', 'put_mid', 'put_spread', 'put_oi']
    
    merged = pd.merge(calls_sub, puts_sub, on='strike', how='inner')
    
    if len(merged) == 0:
        return S
    
    # Filter for liquid strikes
    merged = merged[
        (merged['call_spread'] < 0.20) & 
        (merged['put_spread'] < 0.20) &
        (merged['call_oi'] > 10) &
        (merged['put_oi'] > 10)
    ]
    
    if len(merged) == 0:
        merged = pd.merge(calls_sub, puts_sub, on='strike', how='inner')
        merged = merged[(merged['call_spread'] < 0.50) & (merged['put_spread'] < 0.50)]
    
    if len(merged) == 0:
        return S
    
    discount = np.exp(-r * T)
    merged['F_implied'] = merged['strike'] + (merged['call_mid'] - merged['put_mid']) / discount
    
    # Weight by inverse spread and ATM proximity
    merged['w'] = 1.0 / (merged['call_spread'] + merged['put_spread'] + 0.01)
    atm_K = merged['strike'].iloc[(merged['strike'] - S).abs().argmin()]
    merged['w'] *= np.exp(-0.5 * ((merged['strike'] - atm_K) / (0.1 * S))**2)
    
    return np.average(merged['F_implied'], weights=merged['w'])


# --- Data Container ---

@dataclass
class VolSurfaceData:
    ticker: str
    spot: float
    fetch_time: datetime
    slices: dict      # expiry -> DataFrame
    forwards: dict    # expiry -> F
    expiries: list
    rates: dict


# --- Pipeline ---

class SPYDataPipeline:
    def __init__(self, r=0.045, min_dte=3, max_dte=365, 
                 k_min=-0.5, k_max=0.5, max_spread=0.50, min_oi=10):
        self.r = r
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.k_min = k_min
        self.k_max = k_max
        self.max_spread = max_spread
        self.min_oi = min_oi
        
    def fetch(self, ticker='SPY'):
        print(f"Fetching options data for {ticker}...")
        
        yf_ticker = yf.Ticker(ticker)
        spot = yf_ticker.info.get('regularMarketPrice') or yf_ticker.info.get('previousClose')
        if spot is None:
            spot = yf_ticker.history(period='1d')['Close'].iloc[-1]
        
        print(f"Spot price: ${spot:.2f}")
        
        expiries_raw = yf_ticker.options
        today = date.today()
        
        valid_expiries = []
        for exp in expiries_raw:
            dte = (datetime.strptime(exp, '%Y-%m-%d').date() - today).days
            if self.min_dte <= dte <= self.max_dte:
                valid_expiries.append(exp)
        
        print(f"Found {len(valid_expiries)} expiries in [{self.min_dte}, {self.max_dte}] DTE range")
        
        slices, forwards, rates = {}, {}, {}
        
        for exp in valid_expiries:
            T = (datetime.strptime(exp, '%Y-%m-%d').date() - today).days / 365.0
            
            try:
                chain = yf_ticker.option_chain(exp)
            except Exception as e:
                print(f"  Skipping {exp}: {e}")
                continue
            
            calls = self._add_mid(chain.calls.copy())
            puts = self._add_mid(chain.puts.copy())
            
            if len(calls) == 0 or len(puts) == 0:
                continue
            
            F = extract_forward(calls, puts, spot, T, self.r)
            
            calls_proc = self._to_surface_inputs(calls, F, T, self.r, 'call')
            puts_proc = self._to_surface_inputs(puts, F, T, self.r, 'put')
            
            # Use OTM options only
            combined = pd.concat([
                puts_proc[puts_proc['strike'] < F],
                calls_proc[calls_proc['strike'] >= F]
            ]).sort_values('strike').reset_index(drop=True)
            
            combined = self._filter(combined)
            
            if len(combined) < 5:
                print(f"  {exp}: insufficient data ({len(combined)} points)")
                continue
            
            slices[exp] = combined
            forwards[exp] = F
            rates[exp] = self.r
            
            print(f"  {exp}: {len(combined)} strikes, F={F:.2f}, T={T:.3f}y")
        
        return VolSurfaceData(
            ticker=ticker, spot=spot, fetch_time=datetime.now(),
            slices=slices, forwards=forwards, expiries=sorted(slices.keys()), rates=rates
        )
    
    def _add_mid(self, df):
        df = df.copy()
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid'].replace(0, np.nan)
        df['spread_pct'] = df['spread_pct'].fillna(1.0)
        return df
    
    def _to_surface_inputs(self, df, F, T, r, opt_type):
        out = df[['strike', 'bid', 'ask', 'mid', 'spread_pct', 'openInterest', 'volume']].copy()
        out['k'] = np.log(out['strike'] / F)
        out['iv'] = out.apply(
            lambda row: implied_vol(row['mid'], F, row['strike'], T, r, opt_type), axis=1
        )
        out['w'] = out['iv']**2 * T
        out['T'] = T
        out['option_type'] = opt_type
        return out
    
    def _filter(self, df):
        mask = (
            df['iv'].notna() &
            (df['iv'] > 0.01) & (df['iv'] < 3.0) &
            (df['spread_pct'] <= self.max_spread) &
            (df['openInterest'] >= self.min_oi) &
            (df['k'] >= self.k_min) & (df['k'] <= self.k_max)
        )
        return df[mask].copy()


# --- Utilities ---

def fetch_spy_surface(aggressive=False):
    if aggressive:
        pipeline = SPYDataPipeline(max_spread=0.20, min_oi=100, k_min=-0.3, k_max=0.3)
    else:
        pipeline = SPYDataPipeline()
    return pipeline.fetch()


def surface_to_dataframe(data):
    dfs = []
    for exp, df in data.slices.items():
        slice_df = df.copy()
        slice_df['expiry'] = exp
        slice_df['F'] = data.forwards[exp]
        dfs.append(slice_df)
    return pd.concat(dfs, ignore_index=True)


# --- Main ---

if __name__ == '__main__':
    data = fetch_spy_surface()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Ticker: {data.ticker}")
    print(f"Spot: ${data.spot:.2f}")
    print(f"Fetch time: {data.fetch_time}")
    print(f"Expiries: {len(data.expiries)}")
    
    print("\nExpiry breakdown:")
    for exp in data.expiries[:10]:
        df = data.slices[exp]
        F = data.forwards[exp]
        T = df['T'].iloc[0]
        print(f"  {exp}: {len(df):3d} strikes, F=${F:.2f}, T={T:.3f}y, k∈[{df['k'].min():.3f}, {df['k'].max():.3f}]")
    
    if len(data.expiries) > 10:
        print(f"  ... and {len(data.expiries) - 10} more")
    
    # Sample around ATM
    nearest = data.expiries[0]
    df = data.slices[nearest]
    F = data.forwards[nearest]
    atm_idx = (df['strike'] - F).abs().argmin()
    sample = df[['strike', 'k', 'mid', 'iv', 'w', 'openInterest']].iloc[max(0,atm_idx-5):atm_idx+6]
    print(f"\nSample for {nearest} (around ATM, F=${F:.2f}):")
    print(sample.to_string(index=False))
    
    # Export ts
    full_df = surface_to_dataframe(data)
    full_df['spot'] = data.spot
    full_df['fetch_time'] = data.fetch_time
    
    cols = ['fetch_time', 'expiry', 'T', 'spot', 'F', 'strike', 'k', 
            'option_type', 'bid', 'ask', 'mid', 'spread_pct', 'iv', 'w', 
            'volume', 'openInterest']
    full_df = full_df[[c for c in cols if c in full_df.columns]]
    
    filename = f"spy_vol_surface_{data.fetch_time.strftime('%Y%m%d_%H%M%S')}.csv"
    full_df.to_csv(filename, index=False)
    print(f"\n✓ Saved {len(full_df)} data points to {filename}")