"""
Volatility Surface Visualization
Generates plots from pre-computed SVI and SSVI calibration results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple
import glob
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
C = {'market': '#2E86AB', 'svi': '#A23B72', 'ssvi': '#F18F01', 'error': '#C73E1D'}


# --- Individual Smiles ---

def plot_smile(df, expiry, figsize=(10, 6)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True, gridspec_kw={'hspace': 0.05})
    
    k = df['k'].values
    iv_mkt = df['iv_market'].values * 100
    iv_fit = df['iv_fitted'].values * 100
    iv_err = df['iv_error'].values * 100
    
    axes[0].scatter(k, iv_mkt, s=30, c=C['market'], alpha=0.7, label='Market', zorder=3)
    axes[0].plot(k, iv_fit, '-', c=C['ssvi'], lw=2, label='Fitted', zorder=2)
    axes[0].set_ylabel('IV (%)')
    axes[0].legend(loc='upper right')
    axes[0].axvline(0, color='gray', ls='--', alpha=0.5, lw=1)
    axes[0].set_title(f"{expiry} (T={df['T'].iloc[0]:.3f}y)")
    
    axes[1].bar(k, iv_err, width=0.01, color=C['error'], alpha=0.7)
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_xlabel('Log-moneyness (k)')
    axes[1].set_ylabel('Error (%)')
    axes[1].set_title(f"RMSE: {np.sqrt(np.mean(iv_err**2)):.3f}%", fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_multiple_smiles(fitted_df, expiries=None, n_cols=3, figsize=(15, 10)):
    if expiries is None:
        all_exp = fitted_df['expiry'].unique()
        expiries = [all_exp[i] for i in np.linspace(0, len(all_exp)-1, min(9, len(all_exp))).astype(int)]
    
    n_rows = (len(expiries) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(expiries) > 1 else [axes]
    
    for i, exp in enumerate(expiries):
        ax = axes[i]
        df = fitted_df[fitted_df['expiry'] == exp]
        
        k = df['k'].values
        iv_mkt = df['iv_market'].values * 100
        iv_fit = df['iv_fitted'].values * 100
        
        ax.scatter(k, iv_mkt, s=15, c=C['market'], alpha=0.6)
        ax.plot(k, iv_fit, '-', c=C['ssvi'], lw=1.5)
        
        rmse = np.sqrt(np.mean((iv_fit - iv_mkt)**2))
        ax.set_title(f"{exp}\nT={df['T'].iloc[0]:.3f}y, RMSE={rmse:.2f}%", fontsize=9)
        ax.axvline(0, color='gray', ls='--', alpha=0.3, lw=0.5)
        
        if i % n_cols == 0: ax.set_ylabel('IV (%)')
        if i >= len(expiries) - n_cols: ax.set_xlabel('k')
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Volatility Smiles: Market vs Fitted', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# --- ATM Term Structure ---

def plot_atm_term_structure(T, theta, theta_interp=None, figsize=(10, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].scatter(T, theta, s=50, c=C['market'], label='Market', zorder=3)
    if theta_interp:
        T_s = np.linspace(T.min(), T.max(), 100)
        axes[0].plot(T_s, theta_interp(T_s), '-', c=C['ssvi'], lw=2, label='Interp', zorder=2)
    axes[0].set_xlabel('T (years)')
    axes[0].set_ylabel('ATM Total Variance (θ)')
    axes[0].set_title('ATM Total Variance Term Structure')
    axes[0].legend()
    
    atm_iv = np.sqrt(theta / T) * 100
    axes[1].scatter(T, atm_iv, s=50, c=C['market'], label='Market', zorder=3)
    if theta_interp:
        axes[1].plot(T_s, np.sqrt(theta_interp(T_s) / T_s) * 100, '-', c=C['ssvi'], lw=2, label='Interp', zorder=2)
    axes[1].set_xlabel('T (years)')
    axes[1].set_ylabel('ATM IV (%)')
    axes[1].set_title('ATM Volatility Term Structure')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


# --- 3D Surfaces ---

def plot_3d_surface(fitted_df, plot_type='fitted', figsize=(12, 8), elev=25, azim=-45):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    T = fitted_df['T'].values
    k = fitted_df['k'].values
    
    if plot_type == 'market':
        z = fitted_df['iv_market'].values * 100
        title = 'Market Volatility Surface'
    elif plot_type == 'error':
        z = fitted_df['iv_error'].values * 100
        title = 'Fitting Error Surface'
    else:
        z = fitted_df['iv_fitted'].values * 100
        title = 'Fitted Volatility Surface'
    
    cmap = cm.RdBu_r if plot_type == 'error' else cm.viridis
    scatter = ax.scatter(T, k, z, c=z, cmap=cmap, s=5, alpha=0.6)
    
    ax.set_xlabel('T')
    ax.set_ylabel('k')
    ax.set_zlabel('IV (%)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='IV (%)')
    
    plt.tight_layout()
    return fig


def plot_3d_wireframe(smooth_df, figsize=(12, 8), elev=25, azim=-45):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    T_u = np.sort(smooth_df['T'].unique())
    k_u = np.sort(smooth_df['k'].unique())
    T_g, k_g = np.meshgrid(T_u, k_u)
    iv_g = smooth_df.pivot(index='k', columns='T', values='iv').values * 100
    
    surf = ax.plot_surface(T_g, k_g, iv_g, cmap=cm.viridis, alpha=0.8, lw=0.5, edgecolor='gray')
    
    ax.set_xlabel('T')
    ax.set_ylabel('k')
    ax.set_zlabel('IV (%)')
    ax.set_title('SSVI Volatility Surface')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='IV (%)')
    
    plt.tight_layout()
    return fig


def plot_3d_side_by_side(fitted_df, figsize=(16, 7), elev=25, azim=-45):
    fig = plt.figure(figsize=figsize)
    
    T = fitted_df['T'].values
    k = fitted_df['k'].values
    iv_mkt = fitted_df['iv_market'].values * 100
    iv_fit = fitted_df['iv_fitted'].values * 100
    
    ax1 = fig.add_subplot(121, projection='3d')
    s1 = ax1.scatter(T, k, iv_mkt, c=iv_mkt, cmap=cm.viridis, s=5, alpha=0.6)
    ax1.set_xlabel('T'); ax1.set_ylabel('k'); ax1.set_zlabel('IV (%)')
    ax1.set_title('Market Surface')
    ax1.view_init(elev=elev, azim=azim)
    
    ax2 = fig.add_subplot(122, projection='3d')
    s2 = ax2.scatter(T, k, iv_fit, c=iv_fit, cmap=cm.viridis, s=5, alpha=0.6)
    ax2.set_xlabel('T'); ax2.set_ylabel('k'); ax2.set_zlabel('IV (%)')
    ax2.set_title('Fitted Surface (SSVI)')
    ax2.view_init(elev=elev, azim=azim)
    
    z_min, z_max = min(iv_mkt.min(), iv_fit.min()), max(iv_mkt.max(), iv_fit.max())
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)
    
    fig.colorbar(s1, ax=[ax1, ax2], shrink=0.5, aspect=15, label='IV (%)')
    plt.suptitle('Market vs Fitted Volatility Surface', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_3d_overlay(fitted_df, figsize=(12, 9), elev=25, azim=-45):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    T = fitted_df['T'].values
    k = fitted_df['k'].values
    iv_mkt = fitted_df['iv_market'].values * 100
    iv_fit = fitted_df['iv_fitted'].values * 100
    
    ax.scatter(T, k, iv_mkt, c=C['market'], s=15, alpha=0.6, label='Market')
    ax.scatter(T, k, iv_fit, c=C['ssvi'], s=10, alpha=0.4, label='SSVI Fitted')
    
    ax.set_xlabel('T'); ax.set_ylabel('k'); ax.set_zlabel('IV (%)')
    ax.set_title('Market vs SSVI Fitted Surface (Overlay)')
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_3d_overlay_wireframe(fitted_df, ssvi_result, figsize=(12, 9), elev=25, azim=-45):
    from ssvi_calibration import generate_smooth_surface
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    T = fitted_df['T'].values
    k = fitted_df['k'].values
    iv_mkt = fitted_df['iv_market'].values * 100
    
    ax.scatter(T, k, iv_mkt, c=C['market'], s=12, alpha=0.5, label='Market')
    
    smooth_df = generate_smooth_surface(ssvi_result)
    T_u = np.sort(smooth_df['T'].unique())
    k_u = np.sort(smooth_df['k'].unique())
    T_g, k_g = np.meshgrid(T_u, k_u)
    iv_g = smooth_df.pivot(index='k', columns='T', values='iv').values * 100
    
    ax.plot_wireframe(T_g, k_g, iv_g, color=C['ssvi'], alpha=0.4, lw=0.5, label='SSVI Surface')
    
    ax.set_xlabel('T'); ax.set_ylabel('k'); ax.set_zlabel('IV (%)')
    ax.set_title('Market Data vs SSVI Fitted Surface')
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig


# --- Error Analysis ---

def plot_error_analysis(fitted_df, figsize=(14, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    iv_err = fitted_df['iv_error'].values * 100
    k = fitted_df['k'].values
    T = fitted_df['T'].values
    
    # Histogram
    axes[0,0].hist(iv_err, bins=50, color=C['error'], alpha=0.7, edgecolor='white')
    axes[0,0].axvline(0, color='black', lw=1)
    axes[0,0].axvline(np.mean(iv_err), color='red', ls='--', label=f'Mean: {np.mean(iv_err):.3f}%')
    axes[0,0].set_xlabel('IV Error (%)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Error Distribution')
    axes[0,0].legend()
    
    # Error vs moneyness
    axes[0,1].scatter(k, iv_err, s=5, alpha=0.5, c=C['error'])
    axes[0,1].axhline(0, color='black', lw=0.5)
    idx = np.argsort(k)
    roll = pd.Series(iv_err[idx]).rolling(max(len(k)//20, 10), center=True).mean()
    axes[0,1].plot(k[idx], roll, c='black', lw=2, label='Rolling mean')
    axes[0,1].set_xlabel('k')
    axes[0,1].set_ylabel('IV Error (%)')
    axes[0,1].set_title('Error vs Moneyness')
    axes[0,1].legend()
    
    # Error vs maturity
    axes[1,0].scatter(T, iv_err, s=5, alpha=0.5, c=C['error'])
    axes[1,0].axhline(0, color='black', lw=0.5)
    axes[1,0].set_xlabel('T')
    axes[1,0].set_ylabel('IV Error (%)')
    axes[1,0].set_title('Error vs Maturity')
    
    # RMSE by expiry
    expiries = fitted_df['expiry'].unique()
    rmse_list, T_list = [], []
    for exp in expiries:
        mask = fitted_df['expiry'] == exp
        rmse_list.append(np.sqrt(np.mean(fitted_df.loc[mask, 'iv_error']**2)) * 100)
        T_list.append(fitted_df.loc[mask, 'T'].iloc[0])
    
    axes[1,1].bar(range(len(expiries)), rmse_list, color=C['error'], alpha=0.7)
    axes[1,1].set_xticks(range(len(expiries)))
    axes[1,1].set_xticklabels([f'{t:.2f}' for t in T_list], rotation=45, ha='right')
    axes[1,1].set_xlabel('T')
    axes[1,1].set_ylabel('RMSE (%)')
    axes[1,1].set_title('RMSE by Expiry')
    
    plt.suptitle('Fitting Error Analysis', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# --- SVI vs SSVI Comparison ---

def plot_svi_vs_ssvi(data, svi_results, ssvi_result, expiries=None, figsize=(15, 10)):
    from svi_calibration import svi_raw, SVIParams
    from ssvi_calibration import ssvi_w
    
    if expiries is None:
        all_exp = list(data.expiries)
        expiries = [all_exp[i] for i in np.linspace(0, len(all_exp)-1, min(6, len(all_exp))).astype(int)]
    
    n_cols = 3
    n_rows = (len(expiries) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, exp in enumerate(expiries):
        ax = axes[i]
        df = data.slices[exp]
        k = df['k'].values
        w_mkt = df['w'].values
        T = df['T'].iloc[0]
        
        iv_mkt = np.sqrt(w_mkt / T) * 100
        
        # SVI
        if exp in svi_results:
            w_svi = svi_raw(k, svi_results[exp].params)
            iv_svi = np.sqrt(np.maximum(w_svi, 0) / T) * 100
        else:
            iv_svi = None
        
        # SSVI
        theta = float(ssvi_result.theta_interp(T))
        w_ssvi = ssvi_w(k, theta, ssvi_result.params)
        iv_ssvi = np.sqrt(np.maximum(w_ssvi, 0) / T) * 100
        
        ax.scatter(k, iv_mkt, s=20, c=C['market'], alpha=0.6, label='Market')
        if iv_svi is not None:
            ax.plot(k, iv_svi, '-', c=C['svi'], lw=1.5, label='SVI')
        ax.plot(k, iv_ssvi, '--', c=C['ssvi'], lw=1.5, label='SSVI')
        
        ax.axvline(0, color='gray', ls='--', alpha=0.3, lw=0.5)
        ax.set_title(f'{exp} (T={T:.3f}y)', fontsize=9)
        
        if i == 0: ax.legend(loc='upper right', fontsize=8)
        if i % n_cols == 0: ax.set_ylabel('IV (%)')
        if i >= len(expiries) - n_cols: ax.set_xlabel('k')
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('SVI vs SSVI Comparison', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# --- Data Loading ---

def find_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return max(files)


def load_svi_results(params_csv, fitted_csv):
    return pd.read_csv(params_csv), pd.read_csv(fitted_csv)


def load_ssvi_results(params_csv, atm_csv, fitted_csv):
    params_df = pd.read_csv(params_csv)
    atm_df = pd.read_csv(atm_csv)
    fitted_df = pd.read_csv(fitted_csv)
    
    class SSVIParams:
        def __init__(self, rho, eta, gamma):
            self.rho, self.eta, self.gamma = rho, eta, gamma
    
    params = SSVIParams(params_df['rho'].iloc[0], params_df['eta'].iloc[0], params_df['gamma'].iloc[0])
    theta_interp = interp1d(atm_df['T'].values, atm_df['theta'].values, kind='linear', fill_value='extrapolate')
    
    class Result: pass
    result = Result()
    result.params = params
    result.theta_interp = theta_interp
    result.T_values = atm_df['T'].values
    result.theta_values = atm_df['theta'].values
    result.rmse = params_df['rmse'].iloc[0]
    
    return result, fitted_df


def load_surface_data(csv_path):
    df = pd.read_csv(csv_path)
    
    class Data: pass
    data = Data()
    data.ticker = 'SPY'
    data.spot = df['spot'].iloc[0]
    data.fetch_time = pd.to_datetime(df['fetch_time'].iloc[0])
    data.expiries = sorted(df['expiry'].unique())
    data.slices = {exp: df[df['expiry'] == exp].copy() for exp in data.expiries}
    data.forwards = {exp: df[df['expiry'] == exp]['F'].iloc[0] for exp in data.expiries}
    return data


# --- Main ---

def generate_all_plots(data_csv, svi_params_csv, svi_fitted_csv, 
                       ssvi_params_csv, ssvi_atm_csv, ssvi_fitted_csv, output_dir='.'):
    print("Loading results...")
    
    data = load_surface_data(data_csv)
    svi_params_df, svi_fitted_df = load_svi_results(svi_params_csv, svi_fitted_csv)
    ssvi_result, ssvi_fitted_df = load_ssvi_results(ssvi_params_csv, ssvi_atm_csv, ssvi_fitted_csv)
    
    print(f"  Raw data: {len(data.expiries)} expiries")
    print(f"  SVI: {len(svi_params_df)} fits")
    print(f"  SSVI: ρ={ssvi_result.params.rho:.4f}, η={ssvi_result.params.eta:.4f}, γ={ssvi_result.params.gamma:.4f}")
    
    # Reconstruct SVI results dict
    from svi_calibration import SVIParams, SVIResult
    svi_results = {}
    for _, row in svi_params_df.iterrows():
        params = SVIParams(row['a'], row['b'], row['rho'], row['m'], row['sigma'])
        svi_results[row['expiry']] = SVIResult(params, row['expiry'], row['T'], row['rmse'], 
                                                row['max_error'], row['n_points'], row['is_arbitrage_free'], row['success'], '')
    
    timestamp = data.fetch_time.strftime('%Y%m%d_%H%M%S')
    saved = []
    
    print("\nGenerating plots...")
    
    plots = [
        ('smiles', lambda: plot_multiple_smiles(ssvi_fitted_df)),
        ('atm_term_structure', lambda: plot_atm_term_structure(ssvi_result.T_values, ssvi_result.theta_values, ssvi_result.theta_interp)),
        ('3d_market', lambda: plot_3d_surface(ssvi_fitted_df, 'market')),
        ('3d_fitted', lambda: plot_3d_surface(ssvi_fitted_df, 'fitted')),
        ('3d_smooth', lambda: plot_3d_wireframe(generate_smooth_surface(ssvi_result))),
        ('error_analysis', lambda: plot_error_analysis(ssvi_fitted_df)),
        ('svi_vs_ssvi', lambda: plot_svi_vs_ssvi(data, svi_results, ssvi_result)),
        ('3d_side_by_side', lambda: plot_3d_side_by_side(ssvi_fitted_df)),
        ('3d_overlay', lambda: plot_3d_overlay(ssvi_fitted_df)),
        ('3d_overlay_wireframe', lambda: plot_3d_overlay_wireframe(ssvi_fitted_df, ssvi_result)),
    ]
    
    for name, plot_fn in plots:
        from ssvi_calibration import generate_smooth_surface
        fig = plot_fn()
        fname = f'{output_dir}/plot_{name}_{timestamp}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(fname)
        print(f"  ✓ {fname}")
    
    return saved


if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("VOLATILITY SURFACE VISUALIZATION")
    print("="*60)
    
    if len(sys.argv) == 7:
        files = sys.argv[1:7]
    else:
        print("\nAuto-detecting CSV files...")
        try:
            files = [
                find_latest("spy_vol_surface_*.csv"),
                find_latest("svi_params_*.csv"),
                find_latest("svi_fitted_*.csv"),
                find_latest("ssvi_params_*.csv"),
                find_latest("ssvi_atm_term_structure_*.csv"),
                find_latest("ssvi_fitted_*.csv")
            ]
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nRun data_pipeline.py, svi_calibration.py, ssvi_calibration.py first.")
            sys.exit(1)
        
        for f in files:
            print(f"  {f}")
    
    print()
    saved = generate_all_plots(*files)
    
    print("\n" + "="*60)
    print(f"Done! Saved {len(saved)} plots.")
    print("="*60)
