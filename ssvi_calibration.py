"""
SSVI Calibration
Fits the entire volatility surface jointly using the SSVI parameterization.

SSVI: w(k,θ) = (θ/2) * (1 + ρφ(θ)k + √((φ(θ)k + ρ)² + 1-ρ²))
where φ(θ) = η / (θ^γ (1+θ)^(1-γ))
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Tuple
import glob
import warnings

warnings.filterwarnings('ignore')


# --- SSVI Parameterization ---

@dataclass
class SSVIParams:
    rho: float    # skew
    eta: float    # vol-of-vol
    gamma: float  # power-law exponent
    
    def to_array(self):
        return np.array([self.rho, self.eta, self.gamma])
    
    @classmethod
    def from_array(cls, arr):
        return cls(rho=arr[0], eta=arr[1], gamma=arr[2])
    
    def __repr__(self):
        return f"SSVI(ρ={self.rho:.4f}, η={self.eta:.4f}, γ={self.gamma:.4f})"


def phi(theta, eta, gamma):
    """Power-law phi function."""
    theta = np.maximum(theta, 1e-8)
    return eta / (theta**gamma * (1 + theta)**(1 - gamma))


def ssvi_w(k, theta, params):
    """SSVI total variance."""
    k = np.asarray(k)
    p = phi(theta, params.eta, params.gamma)
    pk = p * k
    return (theta / 2) * (1 + params.rho * pk + np.sqrt((pk + params.rho)**2 + 1 - params.rho**2))


def ssvi_w_arr(k, theta, p_arr):
    """SSVI total variance (array params)."""
    rho, eta, gamma = p_arr
    theta = max(theta, 1e-8)
    p = eta / (theta**gamma * (1 + theta)**(1 - gamma))
    pk = p * k
    return (theta / 2) * (1 + rho * pk + np.sqrt((pk + rho)**2 + 1 - rho**2))


# --- ATM Term Structure ---

def extract_atm(data) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ATM total variance term structure."""
    T_list, theta_list = [], []
    
    for exp in data.expiries:
        df = data.slices[exp]
        T = df['T'].iloc[0]
        atm_idx = df['k'].abs().argmin()
        theta_list.append(df['w'].iloc[atm_idx])
        T_list.append(T)
    
    return np.array(T_list), np.array(theta_list)


def fit_atm_curve(T, theta, method='linear'):
    """Fit monotonic ATM variance curve."""
    idx = np.argsort(T)
    T_s, theta_s = T[idx], theta[idx]
    theta_mono = np.maximum.accumulate(theta_s)  # enforce monotonicity
    return interp1d(T_s, theta_mono, kind=method, fill_value='extrapolate')


# --- Calibration ---

@dataclass
class SSVIResult:
    params: SSVIParams
    theta_interp: callable
    T_values: np.ndarray
    theta_values: np.ndarray
    rmse: float
    max_error: float
    n_points: int
    success: bool
    
    def __repr__(self):
        return f"SSVI({self.params}, RMSE={self.rmse:.6f})"


class SSVICalibrator:
    BOUNDS = [
        (-0.999, 0.0),  # rho (negative for equity)
        (0.01, 5.0),    # eta
        (0.01, 0.99)    # gamma
    ]
    
    def __init__(self, method='differential_evolution'):
        self.method = method
    
    def fit(self, data, verbose=True):
        if verbose:
            print("Fitting SSVI surface...")
            print("-" * 60)
            print("Stage 1: ATM term structure")
        
        T_vals, theta_vals = extract_atm(data)
        theta_interp = fit_atm_curve(T_vals, theta_vals)
        
        if verbose:
            print(f"  {len(T_vals)} expiries, θ ∈ [{theta_vals.min():.6f}, {theta_vals.max():.6f}]")
        
        # Collect all data
        all_k, all_w, all_theta = [], [], []
        for exp in data.expiries:
            df = data.slices[exp]
            T = df['T'].iloc[0]
            theta = float(theta_interp(T))
            all_k.extend(df['k'].values)
            all_w.extend(df['w'].values)
            all_theta.extend([theta] * len(df))
        
        all_k = np.array(all_k)
        all_w = np.array(all_w)
        all_theta = np.array(all_theta)
        
        if verbose:
            print(f"  {len(all_k)} total points")
            print("\nStage 2: Global fit")
        
        def objective(p):
            w_fit = np.array([ssvi_w_arr(all_k[i], all_theta[i], p) for i in range(len(all_k))])
            mse = np.mean((w_fit - all_w)**2)
            if p[0] >= 0:  # penalize positive rho
                mse += 0.1 * p[0]**2
            return mse
        
        if self.method == 'differential_evolution':
            res = differential_evolution(objective, self.BOUNDS, seed=42, maxiter=500, tol=1e-8)
        else:
            res = minimize(objective, [-0.5, 1.0, 0.5], method=self.method, bounds=self.BOUNDS)
        
        params = SSVIParams.from_array(res.x)
        
        w_fit = np.array([ssvi_w_arr(all_k[i], all_theta[i], res.x) for i in range(len(all_k))])
        rmse = np.sqrt(np.mean((w_fit - all_w)**2))
        max_err = np.max(np.abs(w_fit - all_w))
        
        if verbose:
            print(f"  {params}")
            print(f"  RMSE: {rmse:.6f}, Max error: {max_err:.6f}")
            print("-" * 60)
        
        return SSVIResult(params, theta_interp, T_vals, theta_vals, rmse, max_err, len(all_k), rmse < 0.01)


# --- Utilities ---

def compute_ssvi_surface(data, result):
    """Compute fitted surface for all market points."""
    rows = []
    for exp in data.expiries:
        df = data.slices[exp]
        T = df['T'].iloc[0]
        F = data.forwards[exp]
        theta = float(result.theta_interp(T))
        
        k = df['k'].values
        w_mkt = df['w'].values
        w_fit = ssvi_w(k, theta, result.params)
        
        iv_mkt = np.sqrt(w_mkt / T)
        iv_fit = np.sqrt(np.maximum(w_fit, 0) / T)
        
        for i in range(len(k)):
            rows.append({
                'expiry': exp, 'T': T, 'F': F, 'theta': theta,
                'strike': df['strike'].iloc[i], 'k': k[i],
                'iv_market': iv_mkt[i], 'iv_fitted': iv_fit[i], 'iv_error': iv_fit[i] - iv_mkt[i],
                'w_market': w_mkt[i], 'w_fitted': w_fit[i]
            })
    return pd.DataFrame(rows)


def generate_smooth_surface(result, T_grid=None, k_grid=None):
    """Generate smooth surface on regular grid."""
    if T_grid is None:
        T_grid = np.linspace(result.T_values.min(), result.T_values.max(), 50)
    if k_grid is None:
        k_grid = np.linspace(-0.4, 0.2, 100)
    
    rows = []
    for T in T_grid:
        theta = float(result.theta_interp(T))
        w = ssvi_w(k_grid, theta, result.params)
        iv = np.sqrt(np.maximum(w, 0) / T)
        for i, k in enumerate(k_grid):
            rows.append({'T': T, 'k': k, 'theta': theta, 'w': w[i], 'iv': iv[i]})
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
    
    calibrator = SSVICalibrator()
    result = calibrator.fit(data)
    
    timestamp = data.fetch_time.strftime('%Y%m%d_%H%M%S')
    
    # Save params
    pd.DataFrame([{
        'rho': result.params.rho, 'eta': result.params.eta, 'gamma': result.params.gamma,
        'rmse': result.rmse, 'max_error': result.max_error, 'n_points': result.n_points
    }]).to_csv(f"ssvi_params_{timestamp}.csv", index=False)
    print(f"\n✓ Saved ssvi_params_{timestamp}.csv")
    
    # Save ATM curve
    pd.DataFrame({
        'T': result.T_values, 'theta': result.theta_values,
        'atm_iv': np.sqrt(result.theta_values / result.T_values)
    }).to_csv(f"ssvi_atm_term_structure_{timestamp}.csv", index=False)
    print(f"✓ Saved ssvi_atm_term_structure_{timestamp}.csv")
    
    # Save fitted surface
    fitted_df = compute_ssvi_surface(data, result)
    fitted_df.to_csv(f"ssvi_fitted_{timestamp}.csv", index=False)
    print(f"✓ Saved ssvi_fitted_{timestamp}.csv")
