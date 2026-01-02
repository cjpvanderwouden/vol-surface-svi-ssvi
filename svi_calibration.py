"""
SVI Calibration
Fits the raw SVI parameterization to individual expiry slices.

SVI: w(k) = a + b * (ρ(k-m) + √((k-m)² + σ²))
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass
from typing import Optional, Dict
import glob
import warnings

warnings.filterwarnings('ignore')


# --- SVI Parameterization ---

@dataclass
class SVIParams:
    a: float      # level
    b: float      # angle  
    rho: float    # skew
    m: float      # translation
    sigma: float  # smoothness
    
    def to_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])
    
    @classmethod
    def from_array(cls, arr):
        return cls(a=arr[0], b=arr[1], rho=arr[2], m=arr[3], sigma=arr[4])
    
    def __repr__(self):
        return f"SVI(a={self.a:.6f}, b={self.b:.6f}, ρ={self.rho:.4f}, m={self.m:.4f}, σ={self.sigma:.4f})"


def svi_raw(k, params):
    """Compute total variance w(k) under raw SVI."""
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def svi_raw_arr(k, p):
    """Same as svi_raw but takes array."""
    a, b, rho, m, sigma = p
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


# --- Arbitrage Check ---

def check_butterfly(k, params):
    """Check butterfly arbitrage via g(k) >= 0 condition."""
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    
    km = k - m
    sqrt_term = np.sqrt(km**2 + sigma**2)
    
    w = a + b * (rho * km + sqrt_term)
    w_p = b * (rho + km / sqrt_term)
    w_pp = b * sigma**2 / sqrt_term**3
    
    w = np.maximum(w, 1e-10)
    
    g = (1 - k * w_p / (2 * w))**2 - (w_p**2 / 4) * (1/w + 0.25) + w_pp / 2
    
    return np.all(g >= -1e-8), g


# --- Initial Guess ---

def initial_guess(k, w):
    """Heuristic initial parameters from data."""
    atm_idx = np.argmin(np.abs(k))
    w_atm = w[atm_idx]
    
    left = k < -0.05
    right = k > 0.05
    
    slope_l = np.polyfit(k[left], w[left], 1)[0] if np.sum(left) >= 2 else 0.1
    slope_r = np.polyfit(k[right], w[right], 1)[0] if np.sum(right) >= 2 else 0.05
    
    a = w_atm * 0.9
    b = np.clip((abs(slope_l) + abs(slope_r)) / 2, 0.01, 1.0)
    rho = np.clip((slope_r - slope_l) / (slope_r + slope_l + 1e-6), -0.95, 0.95) if slope_l + slope_r != 0 else -0.3
    
    return SVIParams(a=a, b=b, rho=rho, m=0.0, sigma=0.1)


# --- Calibration ---

@dataclass
class SVIResult:
    params: SVIParams
    expiry: str
    T: float
    rmse: float
    max_error: float
    n_points: int
    is_arbitrage_free: bool
    success: bool
    message: str
    
    def __repr__(self):
        status = "✓" if self.success else "✗"
        arb = "arb-free" if self.is_arbitrage_free else "ARB"
        return f"{status} {self.expiry}: RMSE={self.rmse:.6f}, max_err={self.max_error:.6f}, {arb}"


class SVICalibrator:
    BOUNDS = [
        (-0.5, 0.5),     # a
        (0.001, 2.0),    # b
        (-0.999, 0.999), # rho
        (-0.5, 0.5),     # m
        (0.001, 1.0)     # sigma
    ]
    
    def __init__(self, method='differential_evolution'):
        self.method = method
    
    def fit(self, k, w, expiry='', T=0.0):
        k, w = np.asarray(k), np.asarray(w)
        weights = np.ones(len(k)) / len(k)
        
        def objective(p):
            w_fit = svi_raw_arr(k, p)
            mse = np.sum(weights * (w_fit - w)**2)
            if np.any(w_fit < 0):
                mse += 100.0 * np.sum(np.maximum(-w_fit, 0)**2)
            return mse
        
        x0 = initial_guess(k, w).to_array()
        
        try:
            if self.method == 'differential_evolution':
                res = differential_evolution(objective, self.BOUNDS, seed=42, maxiter=500, tol=1e-8)
            else:
                res = minimize(objective, x0, method=self.method, bounds=self.BOUNDS)
            x_opt = res.x
            msg = str(res.message) if hasattr(res, 'message') else ''
        except Exception as e:
            return SVIResult(SVIParams.from_array(x0), expiry, T, np.inf, np.inf, len(k), False, False, str(e))
        
        params = SVIParams.from_array(x_opt)
        w_fit = svi_raw(k, params)
        
        rmse = np.sqrt(np.mean((w_fit - w)**2))
        max_err = np.max(np.abs(w_fit - w))
        arb_free, _ = check_butterfly(k, params)
        
        return SVIResult(params, expiry, T, rmse, max_err, len(k), arb_free, rmse < 0.01, msg)


def calibrate_surface(data, method='differential_evolution', verbose=True):
    """Calibrate SVI to all expiries."""
    calibrator = SVICalibrator(method=method)
    results = {}
    
    if verbose:
        print(f"Calibrating SVI to {len(data.expiries)} expiries...")
        print("-" * 60)
    
    for exp in data.expiries:
        df = data.slices[exp]
        result = calibrator.fit(df['k'].values, df['w'].values, exp, df['T'].iloc[0])
        results[exp] = result
        if verbose:
            print(result)
    
    if verbose:
        print("-" * 60)
        n_ok = sum(r.success for r in results.values())
        n_arb = sum(r.is_arbitrage_free for r in results.values())
        avg_rmse = np.mean([r.rmse for r in results.values() if r.success])
        print(f"Success: {n_ok}/{len(results)}, Arb-free: {n_arb}/{len(results)}, Avg RMSE: {avg_rmse:.6f}")
    
    return results


# --- Utilities ---

def results_to_dataframe(results):
    rows = []
    for exp, r in results.items():
        rows.append({
            'expiry': exp, 'T': r.T,
            'a': r.params.a, 'b': r.params.b, 'rho': r.params.rho, 
            'm': r.params.m, 'sigma': r.params.sigma,
            'rmse': r.rmse, 'max_error': r.max_error, 'n_points': r.n_points,
            'is_arbitrage_free': r.is_arbitrage_free, 'success': r.success
        })
    return pd.DataFrame(rows)


def compute_fitted_surface(data, results):
    rows = []
    for exp in data.expiries:
        if exp not in results:
            continue
        df = data.slices[exp]
        params = results[exp].params
        T = df['T'].iloc[0]
        F = data.forwards[exp]
        
        k = df['k'].values
        w_mkt = df['w'].values
        w_fit = svi_raw(k, params)
        
        iv_mkt = np.sqrt(w_mkt / T)
        iv_fit = np.sqrt(np.maximum(w_fit, 0) / T)
        
        for i in range(len(k)):
            rows.append({
                'expiry': exp, 'T': T, 'F': F, 'strike': df['strike'].iloc[i], 'k': k[i],
                'iv_market': iv_mkt[i], 'iv_fitted': iv_fit[i], 'iv_error': iv_fit[i] - iv_mkt[i],
                'w_market': w_mkt[i], 'w_fitted': w_fit[i]
            })
    return pd.DataFrame(rows)


def load_surface_from_csv(path):
    """Load surface data from pipeline CSV."""
    df = pd.read_csv(path)
    fetch_time = pd.to_datetime(df['fetch_time'].iloc[0])
    spot = df['spot'].iloc[0]
    
    slices, forwards = {}, {}
    expiries = sorted(df['expiry'].unique())
    
    for exp in expiries:
        slice_df = df[df['expiry'] == exp].copy()
        slices[exp] = slice_df
        forwards[exp] = slice_df['F'].iloc[0]
    
    class Data: pass
    data = Data()
    data.ticker, data.spot, data.fetch_time = 'SPY', spot, fetch_time
    data.slices, data.forwards, data.expiries = slices, forwards, expiries
    return data


def find_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return max(files)


# --- Main ---

if __name__ == '__main__':
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_latest("spy_vol_surface_*.csv")
    
    print(f"Loading {csv_path}...")
    data = load_surface_from_csv(csv_path)
    print(f"  Spot: ${data.spot:.2f}, Expiries: {len(data.expiries)}\n")
    
    results = calibrate_surface(data)
    
    timestamp = data.fetch_time.strftime('%Y%m%d_%H%M%S')
    
    params_df = results_to_dataframe(results)
    params_df.to_csv(f"svi_params_{timestamp}.csv", index=False)
    print(f"\n✓ Saved svi_params_{timestamp}.csv")
    
    fitted_df = compute_fitted_surface(data, results)
    fitted_df.to_csv(f"svi_fitted_{timestamp}.csv", index=False)
    print(f"✓ Saved svi_fitted_{timestamp}.csv")
