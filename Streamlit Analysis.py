import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.dates import DateFormatter
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import streamlit as st
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Configuration Matplotlib ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

# --- Configuration Streamlit ---
st.set_page_config(page_title="Financial Time Series Analysis", layout="wide")
st.title("📈 Financial time series analysis")

# --- Fonctions utilitaires ---

def get_asset_names():
    """Liste tous les fichiers .xlsx du dossier courant et extrait les noms (sans extension)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_files = glob.glob(os.path.join(script_dir, '*.xlsx'))
    asset_names = [os.path.splitext(os.path.basename(f))[0] for f in excel_files if not os.path.basename(f).startswith('~$')]
    return sorted(asset_names)

def load_data(actif):
    """Charge les données d'un actif depuis le fichier Excel"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, actif + '.xlsx')
    if not os.path.exists(file_path):
        return None
    df = pd.read_excel(file_path, engine='openpyxl', decimal=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    return df

def resample_data(df, freq):
    """Resample data according to frequency"""
    freq_map = {
        'Daily': None,
        'Weekly': 'W',
        'Monthly': 'ME',
        'Yearly': 'YE',
    }
    if freq_map[freq]:
        df = df.set_index('Date').resample(freq_map[freq]).last().reset_index()
    return df

def compute_returns(df, return_type):
    """Compute returns according to selected type"""
    if return_type == "Simple":
        returns = df['Close'].pct_change()
    else:
        returns = np.log(df['Close'] / df['Close'].shift(1))
    return returns

# --- Sidebar : Paramètres ---
asset_names = get_asset_names()

with st.sidebar:
    st.header("⚙️ Settings")
    actif = st.selectbox("Asset", asset_names, index=asset_names.index("SPI") if "SPI" in asset_names else 0)
    
    # Charger les données pour obtenir les dates min/max
    df_raw = load_data(actif)
    if df_raw is not None and not df_raw.empty:
        date_min = df_raw['Date'].min().date()
        date_max = df_raw['Date'].max().date()
    else:
        from datetime import date
        date_min = date(2000, 1, 1)
        date_max = date.today()
    
    all_series = st.checkbox("Full series", value=False)
    
    if not all_series:
        date_debut = st.date_input("Start date", value=date_min, min_value=date_min, max_value=date_max)
        date_fin = st.date_input("End date", value=date_max, min_value=date_min, max_value=date_max)
    else:
        date_debut = date_min
        date_fin = date_max
    
    freq = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly", "Yearly"], index=0)
    return_type = st.selectbox("Return type", ["Simple", "Logarithmic"], index=0)

# --- Chargement et filtrage des données ---
df = load_data(actif)
if df is None or df.empty:
    st.error(f"Unable to load data for asset '{actif}'.")
    st.stop()

# Filtrage par dates
if not all_series:
    df = df[(df['Date'] >= pd.Timestamp(date_debut)) & (df['Date'] <= pd.Timestamp(date_fin))]

# Rééchantillonnage
df = resample_data(df, freq)

if df.empty:
    st.warning("No data available for the selected period.")
    st.stop()

# Calcul des rendements
returns = compute_returns(df, return_type)
returns_clean = returns.dropna()

# --- Onglets ---
tab_cours, tab_logprice, tab_return, tab_scatter, tab_moments, tab_qq, tab_stats, tab_cumret, tab_rolling = st.tabs([
    "📈 Price",
    "📊 Log Price",
    "📊 Returns",
    "🔵 Scatter Plot",
    "📈 Rolling Moments",
    "📐 QQ-Plot",
    "📋 Statistics",
    "📈 Comparative Cumulative Return",
    "🔄 Rolling Return",
])

# =====================================================
# Onglet 1 : Cours (prix + drawdown)
# =====================================================
with tab_cours:
    fig, (ax_price, ax_drawdown) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Prix
    ax_price.plot(df['Date'], df['Close'], color='#4F8EF7', linestyle='-')
    ax_price.set_title(f'Stock Price {actif} ({freq})')
    ax_price.set_ylabel('Close')
    ax_price.grid(True)
    
    # Drawdown
    if 'Close' in df.columns and not df['Close'].isnull().all():
        prices = df['Close']
        cummax = prices.cummax()
        drawdown = (prices / cummax) - 1
        ax_drawdown.plot(df['Date'], drawdown, color='crimson', label='Drawdown')
        ax_drawdown.fill_between(df['Date'], drawdown, 0, where=(drawdown < 0), color='crimson', alpha=0.3)
        ax_drawdown.set_ylim(-1, 0)
        ax_drawdown.legend(loc='lower left')
    ax_drawdown.set_title('Drawdown')
    ax_drawdown.set_ylabel('Drawdown')
    ax_drawdown.set_xlabel('Date')
    ax_drawdown.grid(True)
    
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =====================================================
# Onglet 2 : Log Price
# =====================================================
with tab_logprice:
    col1, col2 = st.columns([1, 3])
    with col1:
        show_ma = st.checkbox("Show moving average", value=False, key="logprice_ma_cb")
        ma_window = st.number_input("Moving average (steps)", min_value=1, max_value=10000, value=20, key="logprice_ma_spin")
    
    fig_lp, (ax_logprice, ax_logdelta) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    log_prices = np.log(df['Close'])
    # Use date-based numeric axis (days since start) for a truly straight regression line
    dates_numeric = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds().values / 86400.0
    # Drop NaN values for fitting
    valid_mask = ~np.isnan(log_prices)
    coeffs = np.polyfit(dates_numeric[valid_mask], log_prices[valid_mask], 1)
    trend_line = coeffs[0] * dates_numeric + coeffs[1]
    
    ax_logprice.plot(df['Date'], log_prices, color='#4F8EF7', linestyle='-', linewidth=1.5, label='Log Price')
    ax_logprice.plot(df['Date'], trend_line, color='red', linestyle='--', linewidth=2,
                     label=f'Linear regression (slope: {coeffs[0]:.6f})')
    
    if show_ma:
        log_prices_series = pd.Series(log_prices.values, index=df['Date'])
        ma = log_prices_series.rolling(window=ma_window).mean()
        ax_logprice.plot(df['Date'], ma, color='green', linestyle='-', linewidth=2,
                         label=f'Moving average ({ma_window} steps)')
        
        delta = log_prices_series - ma
        ax_logdelta.plot(df['Date'], delta, color='purple', linewidth=1.5, label='Deviation (Log-Price - MA)')
        ax_logdelta.axhline(0, color='black', linestyle='--', linewidth=1)
        ax_logdelta.fill_between(df['Date'], delta, 0, where=(delta > 0), color='forestgreen', alpha=0.15)
        ax_logdelta.fill_between(df['Date'], delta, 0, where=(delta < 0), color='crimson', alpha=0.15)
        
        delta_std = delta.std()
        ax_logdelta.axhline(2 * delta_std, color='red', linestyle=':', linewidth=1.2, label='± 2σ')
        ax_logdelta.axhline(-2 * delta_std, color='red', linestyle=':', linewidth=1.2)
        ax_logdelta.set_ylabel('Deviation')
        ax_logdelta.grid(True, linestyle=':', alpha=0.6)
        ax_logdelta.legend(loc='lower left', fontsize=9, ncol=2)
    
    ax_logprice.set_title(f'Log-Price {actif} with linear regression ({freq})')
    ax_logprice.set_ylabel('Log(Price)')
    ax_logprice.grid(True)
    ax_logprice.legend()
    ax_logdelta.set_xlabel('Date')
    
    fig_lp.tight_layout()
    st.pyplot(fig_lp)
    plt.close(fig_lp)

# =====================================================
# Onglet 3 : Rendements
# =====================================================
with tab_return:
    fig_ret, (ax_return, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Série temporelle des rendements
    sns.lineplot(x=df['Date'], y=returns, ax=ax_return, color='tab:green', linestyle='-', label='Return')
    std_ret = returns.std(skipna=True)
    ax_return.axhline(2 * std_ret, color='red', linestyle='--', linewidth=1, label='+2 std dev')
    ax_return.axhline(-2 * std_ret, color='red', linestyle='--', linewidth=1, label='-2 std dev')
    ax_return.set_title(f'Returns {actif} ({freq})')
    ax_return.set_xlabel('Date')
    ax_return.set_ylabel(f'{freq} return')
    ax_return.grid(True)
    ax_return.legend()
    
    # Histogramme + ajustement Student-t
    sns.histplot(returns_clean * 100, bins=50, kde=True, ax=ax_hist, color='tab:orange',
                 stat="density", label="Kernel Density Estimate", edgecolor=None, alpha=0.5, shrink=0.95)
    df_fit, loc, scale = stats.t.fit(returns_clean * 100)
    x_hist = pd.Series(returns_clean * 100).sort_values()
    y_hist = stats.t.pdf(x_hist, df_fit, loc, scale)
    ax_hist.plot(x_hist, y_hist, color='blue', linestyle="--", label="Student-t fit")
    ax_hist.legend()
    ax_hist.set_title("Returns histogram (%)")
    ax_hist.grid(True)
    ax_hist.set_xlabel("Return (%)")
    ax_hist.set_ylabel("Density")
    
    x_min, x_max = ax_hist.get_xlim()
    step = 2
    ticks = np.arange(np.floor(x_min), np.ceil(x_max) + step, step)
    ax_hist.set_xticks(ticks)
    
    param_annotation = f"Student-t params\ndf = {df_fit:.2f}\nμ = {loc:.4f}\nσ = {scale:.4f}"
    ax_hist.text(0.98, 0.05, param_annotation, transform=ax_hist.transAxes,
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(facecolor='white', edgecolor='black'))
    
    fig_ret.tight_layout()
    st.pyplot(fig_ret)
    plt.close(fig_ret)

# =====================================================
# Onglet 4 : Nuage de points
# =====================================================
with tab_scatter:
    fig_sc, ax_scatter = plt.subplots(figsize=(10, 8))
    
    returns_t = returns_clean.values[1:]
    returns_tm1 = returns_clean.values[:-1]
    ax_scatter.scatter(returns_tm1, returns_t, alpha=0.5, color='tab:blue')
    ax_scatter.margins(0.1)
    
    x_mean = returns_tm1.mean()
    y_mean = returns_t.mean()
    a = ((returns_tm1 - x_mean) * (returns_t - y_mean)).sum() / ((returns_tm1 - x_mean) ** 2).sum()
    b = y_mean - a * x_mean
    x_line = np.array([returns_tm1.min(), returns_tm1.max()])
    y_line = a * x_line + b
    ax_scatter.plot(x_line, y_line, color='red', lw=2, label=f"Regression line: y = {a:.2f}x + {b:.2g}")
    
    ax_scatter.set_xlabel('Return t-1')
    ax_scatter.set_ylabel('Return t')
    ax_scatter.set_title(f'Scatter plot: Return t vs t-1 ({freq})')
    ax_scatter.legend()
    ax_scatter.grid(True)
    
    fig_sc.tight_layout()
    st.pyplot(fig_sc)
    plt.close(fig_sc)

# =====================================================
# Onglet 5 : Moments glissants
# =====================================================
with tab_moments:
    rolling_window_val = st.number_input("Rolling window size (days)", min_value=5, max_value=10000, value=252, key="rolling_moments_win")
    
    if st.button("Compute rolling moments", key="btn_moments"):
        fig_mom = plt.figure(figsize=(14, 10))
        gs_mom = gridspec.GridSpec(2, 2)
        ax_mean = fig_mom.add_subplot(gs_mom[0, 0])
        ax_vol = fig_mom.add_subplot(gs_mom[0, 1])
        ax_skew = fig_mom.add_subplot(gs_mom[1, 0])
        ax_kurt = fig_mom.add_subplot(gs_mom[1, 1])
        
        dates_mom = df['Date'].iloc[1:]
        rolling_mean = returns_clean.rolling(rolling_window_val).mean()
        rolling_vol = returns_clean.rolling(rolling_window_val).std()
        rolling_skew_vals = returns_clean.rolling(rolling_window_val).apply(skew, raw=True)
        rolling_kurt_vals = returns_clean.rolling(rolling_window_val).apply(kurtosis, raw=True)
        
        ax_mean.plot(dates_mom, rolling_mean, color='black', label=f"Mean (rolling window {rolling_window_val} days)")
        ax_mean.axhline(0, color='gray', lw=0.7)
        ax_mean.set_title('Time-varying mean')
        ax_mean.set_ylabel('Mean')
        ax_mean.legend(loc='upper right')
        
        ax_vol.plot(dates_mom, rolling_vol, color='black', label=f"Volatility (rolling window {rolling_window_val} days)")
        ax_vol.set_title('Time-varying volatility')
        ax_vol.set_ylabel('Volatility')
        ax_vol.legend(loc='upper right')
        
        ax_skew.plot(dates_mom, rolling_skew_vals, color='black', label=f"Skewness (rolling window {rolling_window_val} days)")
        ax_skew.axhline(0, color='gray', lw=0.7)
        ax_skew.set_title('Time-varying skewness')
        ax_skew.set_ylabel('Skewness')
        ax_skew.legend(loc='upper right')
        
        ax_kurt.plot(dates_mom, rolling_kurt_vals, color='black', label=f"Kurtosis (rolling window {rolling_window_val} days)")
        ax_kurt.set_title('Time-varying kurtosis')
        ax_kurt.set_ylabel('Kurtosis')
        ax_kurt.legend(loc='upper right')
        
        for ax in [ax_mean, ax_vol, ax_skew, ax_kurt]:
            ax.set_xlabel('Date')
            ax.grid(True)
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        
        fig_mom.tight_layout()
        st.pyplot(fig_mom)
        plt.close(fig_mom)

# =====================================================
# Onglet 6 : QQ-Plot + ACF + PACF
# =====================================================
with tab_qq:
    fig_qq = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    ax_qq = fig_qq.add_subplot(gs[0, :])
    ax_acf = fig_qq.add_subplot(gs[1, 0])
    ax_pacf = fig_qq.add_subplot(gs[1, 1])
    
    # QQ-Plot
    stats.probplot(returns_clean, dist="norm", plot=ax_qq)
    for line in ax_qq.get_lines():
        if line.get_linestyle() == 'None':
            line.set_color('#1f77b4')
    ax_qq.grid(True)
    ax_qq.set_title('QQ-Plot of returns')
    
    # Test de Jarque-Bera
    jb_stat, jb_pvalue = jarque_bera(returns_clean)
    normality = "Normal" if jb_pvalue > 0.05 else "Non-Normal"
    label_jb = f"JB = {jb_stat:.2f}\n{normality}"
    ax_qq.text(0.98, 0.02, label_jb, transform=ax_qq.transAxes,
               fontsize=12, color='black', ha='right', va='bottom',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # ACF
    plot_acf(returns_clean, ax=ax_acf, lags=10, zero=False, alpha=0.05)
    ax_acf.set_title('ACF of returns')
    
    # PACF
    plot_pacf(returns_clean, ax=ax_pacf, lags=10, zero=False, alpha=0.05, method='ywm')
    ax_pacf.set_title('PACF of returns')
    
    fig_qq.tight_layout()
    st.pyplot(fig_qq)
    plt.close(fig_qq)

# =====================================================
# Onglet 7 : Statistiques
# =====================================================
with tab_stats:
    mean_val = returns_clean.mean()
    median_val = returns_clean.median()
    min_val = returns_clean.min()
    max_val = returns_clean.max()
    std_val = returns_clean.std()
    skewness_val = skew(returns_clean)
    kurt_val = kurtosis(returns_clean)
    var_5 = returns_clean.quantile(0.05)
    cvar_5 = returns_clean[returns_clean <= var_5].mean()
    
    # Maximum Drawdown
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.cummax()
    drawdown_series = (cumulative - running_max) / running_max
    max_drawdown = drawdown_series.min()
    
    plus_3std = 3 * std_val
    minus_3std = -3 * std_val
    
    stats_data = {
        "Statistic": [
            "Mean", "Median", "Minimum", "Maximum",
            "Std Dev", "Skewness", "Kurtosis",
            "VaR 5%", "CVaR 5%", "Maximum Drawdown"
        ],
        "Value": [
            f"{mean_val * 100:.4f} %",
            f"{median_val * 100:.4f} %",
            f"{min_val * 100:.4f} %",
            f"{max_val * 100:.4f} %",
            f"{std_val * 100:.4f} % (±3σ: [{minus_3std * 100:.4f} %, {plus_3std * 100:.4f} %])",
            f"{skewness_val:.4f}",
            f"{kurt_val:.4f}",
            f"{var_5 * 100:.4f} %",
            f"{cvar_5 * 100:.4f} %",
            f"{max_drawdown * 100:.4f} %",
        ]
    }
    
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, width='stretch', hide_index=True)

# =====================================================
# Onglet 8 : Comparative cumulative Return
# =====================================================
with tab_cumret:
    selected_assets = st.multiselect("Select assets to compare", asset_names, default=[], key="cumret_assets")
    
    if selected_assets:
        fig_cr, ax_cumret = plt.subplots(figsize=(14, 6))
        
        for asset in selected_assets:
            df_asset = load_data(asset)
            if df_asset is None:
                st.warning(f"Unable to load {asset}")
                continue
            # Filtrage selon la période choisie
            if not all_series:
                df_asset = df_asset[(df_asset['Date'] >= pd.Timestamp(date_debut)) & (df_asset['Date'] <= pd.Timestamp(date_fin))]
            if 'Close' in df_asset.columns:
                prices_asset = df_asset['Close']
            else:
                prices_asset = df_asset.iloc[:, 1]
            ret_asset = prices_asset.pct_change().dropna()
            cum_asset = (1 + ret_asset).cumprod()
            if not cum_asset.empty:
                final_return = (cum_asset.iloc[-1] - 1) * 100
                label = f"{asset} ({final_return:.2f} %)"
            else:
                label = asset
            ax_cumret.plot(df_asset['Date'].iloc[1:], cum_asset, label=label)
        
        ax_cumret.set_title("Comparative cumulative Return")
        ax_cumret.set_xlabel("Date")
        ax_cumret.set_ylabel("Cumulative Return")
        ax_cumret.legend()
        ax_cumret.grid(True)
        fig_cr.tight_layout()
        st.pyplot(fig_cr)
        plt.close(fig_cr)
    else:
        st.info("Please select at least one asset.")

# =====================================================
# Onglet 9 : Rolling Return
# =====================================================
with tab_rolling:
    window_label = st.selectbox("Rolling window", ["1 year", "3 years", "5 years", "10 years"], index=0, key="rolling_window_sel")
    
    if st.button("Compute rolling return", key="btn_rolling"):
        window_map = {"1 year": 252, "3 years": 252 * 3, "5 years": 252 * 5, "10 years": 252 * 10}
        window = window_map[window_label]
        
        # Recharger et recalculer les rendements pour cet onglet
        df_roll = load_data(actif)
        if df_roll is not None:
            if not all_series:
                df_roll = df_roll[(df_roll['Date'] >= pd.Timestamp(date_debut)) & (df_roll['Date'] <= pd.Timestamp(date_fin))]
            
            returns_roll = compute_returns(df_roll, return_type)
            rolling_return = (1 + returns_roll).rolling(window).apply(np.prod, raw=True) - 1
            
            fig_roll, (ax_rolling, ax_rolling_hist) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Série temporelle du rolling return
            ax_rolling.plot(df_roll['Date'], rolling_return, color='#4F8EF7', label='Rolling Return')
            mean_rr = rolling_return.mean()
            median_rr = rolling_return.median()
            ax_rolling.axhline(mean_rr, color='black', linestyle='-', linewidth=2, label=f'Mean ({mean_rr:.2%})')
            ax_rolling.axhline(median_rr, color='orange', linestyle='--', linewidth=2, label=f'Median ({median_rr:.2%})')
            ax_rolling.set_title(f"Rolling Return {actif} ({window_label})")
            ax_rolling.set_xlabel('Date')
            ax_rolling.set_ylabel('Rolling Return')
            ax_rolling.grid(True)
            ax_rolling.xaxis.set_major_formatter(DateFormatter('%Y'))
            ax_rolling.legend(loc='upper right')
            
            # Histogramme
            n, bins, patches = ax_rolling_hist.hist(rolling_return.dropna() * 100, bins=30, color='#4F8EF7', alpha=0.75, density=True)
            for patch in patches:
                current_width = patch.get_width()
                patch.set_width(current_width * 0.9)
                patch.set_x(patch.get_x() + current_width * 0.05)
            ax_rolling_hist.set_title('Rolling returns distribution')
            ax_rolling_hist.set_xlabel('Rolling return (%)')
            ax_rolling_hist.set_ylabel('Density (%)')
            ax_rolling_hist.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax_rolling_hist.grid(True, linestyle='--')
            
            rolling_return_clean = rolling_return.dropna()
            prob_neg = (rolling_return_clean < 0).mean()
            label_text = f"P(R < 0%) = {prob_neg:.1%}"
            ax_rolling_hist.axvline(0, color='red', linestyle='--', linewidth=2, label=label_text)
            ax_rolling_hist.legend(loc='upper right')
            
            fig_roll.tight_layout()
            st.pyplot(fig_roll)
            plt.close(fig_roll)