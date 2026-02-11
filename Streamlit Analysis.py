import sys
import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera, gaussian_kde
from statsmodels.tsa.stattools import acf, pacf

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

# --- Plotly layout helpers ---
PLOTLY_LAYOUT = dict(
    font=dict(family="Arial", size=14),
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(l=50, r=20, t=60, b=80),
)

def apply_plotly_layout(fig, title=None, xaxis_title=None, yaxis_title=None, height=None, showlegend=True):
    """Apply consistent Plotly layout styling"""
    updates = dict(**PLOTLY_LAYOUT, showlegend=showlegend)
    if title:
        updates['title'] = dict(text=f"<b>{title}</b>", font=dict(size=20, color="#2c3e50"))
    if xaxis_title:
        updates['xaxis_title'] = xaxis_title
    if yaxis_title:
        updates['yaxis_title'] = yaxis_title
    if height:
        updates['height'] = height
    fig.update_layout(**updates)
    # Style subplot titles (annotations)
    fig.update_annotations(font=dict(size=16, color="#2c3e50"))
    return fig

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
    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
        subplot_titles=[f'<b>Stock Price {actif} ({freq})</b>', '<b>Drawdown</b>']
    )
    
    # Prix
    fig_price.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], mode='lines',
                   line=dict(color='#4F8EF7', width=1.5), name='Close'),
        row=1, col=1
    )
    
    # Drawdown
    if 'Close' in df.columns and not df['Close'].isnull().all():
        prices = df['Close']
        cummax = prices.cummax()
        drawdown = (prices / cummax) - 1
        fig_price.add_trace(
            go.Scatter(x=df['Date'], y=drawdown, mode='lines',
                       line=dict(color='crimson', width=1), name='Drawdown',
                       fill='tozeroy', fillcolor='rgba(220,20,60,0.2)'),
            row=2, col=1
        )
    
    fig_price.update_yaxes(title_text='Close', row=1, col=1)
    fig_price.update_yaxes(title_text='Drawdown', range=[-1, 0], row=2, col=1)
    fig_price.update_xaxes(title_text='Date', row=2, col=1)
    apply_plotly_layout(fig_price, height=700)
    st.plotly_chart(fig_price, key="chart_price", use_container_width=True)

# =====================================================
# Onglet 2 : Log Price
# =====================================================
with tab_logprice:
    col1, col2 = st.columns([1, 3])
    with col1:
        show_ma = st.checkbox("Show moving average", value=False, key="logprice_ma_cb")
        ma_window = st.number_input("Moving average (steps)", min_value=1, max_value=10000, value=252, key="logprice_ma_spin")
    
    log_prices = np.log(df['Close'])
    dates_numeric = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds().values / 86400.0
    valid_mask = ~np.isnan(log_prices)
    coeffs = np.polyfit(dates_numeric[valid_mask], log_prices[valid_mask], 1)
    trend_line = coeffs[0] * dates_numeric + coeffs[1]
    
    n_rows = 2 if show_ma else 1
    row_heights = [0.8, 0.2] if show_ma else [1.0]
    subtitles = [f'<b>Log-Price {actif} with linear regression ({freq})</b>']
    if show_ma:
        subtitles.append('<b>Deviation (Log-Price - MA)</b>')
    
    fig_lp = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.08,
        subplot_titles=subtitles
    )
    
    fig_lp.add_trace(
        go.Scatter(x=df['Date'], y=log_prices, mode='lines',
                   line=dict(color='#4F8EF7', width=1.5), name='Log Price'),
        row=1, col=1
    )
    fig_lp.add_trace(
        go.Scatter(x=df['Date'], y=trend_line, mode='lines',
                   line=dict(color='red', width=2, dash='dash'),
                   name=f'Linear regression (slope: {coeffs[0]:.6f})'),
        row=1, col=1
    )
    
    if show_ma:
        log_prices_series = pd.Series(log_prices.values, index=df['Date'])
        ma = log_prices_series.rolling(window=ma_window).mean()
        fig_lp.add_trace(
            go.Scatter(x=df['Date'], y=ma, mode='lines',
                       line=dict(color='green', width=2),
                       name=f'Moving average ({ma_window} steps)'),
            row=1, col=1
        )
        
        delta = log_prices_series - ma
        delta_std = delta.std()
        
        # Positive deviation fill
        delta_pos = delta.where(delta > 0, 0)
        fig_lp.add_trace(
            go.Scatter(x=df['Date'], y=delta_pos, mode='lines',
                       line=dict(color='forestgreen', width=0),
                       fill='tozeroy', fillcolor='rgba(34,139,34,0.15)',
                       name='Positive deviation', showlegend=False),
            row=2, col=1
        )
        # Negative deviation fill
        delta_neg = delta.where(delta < 0, 0)
        fig_lp.add_trace(
            go.Scatter(x=df['Date'], y=delta_neg, mode='lines',
                       line=dict(color='crimson', width=0),
                       fill='tozeroy', fillcolor='rgba(220,20,60,0.15)',
                       name='Negative deviation', showlegend=False),
            row=2, col=1
        )
        # Deviation line
        fig_lp.add_trace(
            go.Scatter(x=df['Date'], y=delta, mode='lines',
                       line=dict(color='purple', width=1.5),
                       name='Deviation (Log-Price - MA)'),
            row=2, col=1
        )
        # +/- 2 sigma lines
        fig_lp.add_hline(y=2 * delta_std, line=dict(color='red', width=1.2, dash='dot'),
                         annotation_text='+ 2σ', row=2, col=1)
        fig_lp.add_hline(y=-2 * delta_std, line=dict(color='red', width=1.2, dash='dot'),
                         annotation_text='- 2σ', row=2, col=1)
        fig_lp.add_hline(y=0, line=dict(color='black', width=1, dash='dash'), row=2, col=1)
        
        fig_lp.update_yaxes(title_text='Deviation', row=2, col=1)
    
    fig_lp.update_yaxes(title_text='Log(Price)', row=1, col=1)
    fig_lp.update_xaxes(title_text='Date', row=n_rows, col=1)
    apply_plotly_layout(fig_lp, height=700 if show_ma else 500)
    st.plotly_chart(fig_lp, key="chart_logprice", use_container_width=True)

# =====================================================
# Onglet 3 : Rendements
# =====================================================
with tab_return:
    fig_ret = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
        subplot_titles=[f'<b>Returns {actif} ({freq})</b>', '<b>Returns histogram (%)</b>']
    )
    
    # Série temporelle des rendements
    std_ret = returns.std(skipna=True)
    fig_ret.add_trace(
        go.Scatter(x=df['Date'], y=returns, mode='lines',
                   line=dict(color='#2ca02c', width=1), name='Return'),
        row=1, col=1
    )
    fig_ret.add_hline(y=2 * std_ret, line=dict(color='red', width=1, dash='dash'),
                      annotation_text='+2σ', row=1, col=1)
    fig_ret.add_hline(y=-2 * std_ret, line=dict(color='red', width=1, dash='dash'),
                      annotation_text='-2σ', row=1, col=1)
    
    # Histogramme
    fig_ret.add_trace(
        go.Histogram(x=returns_clean * 100, nbinsx=50, histnorm='probability density',
                     marker_color='rgba(255,127,14,0.5)', marker_line=dict(width=0),
                     name='Returns distribution'),
        row=2, col=1
    )
    
    # Student-t fit curve
    df_fit, loc, scale = stats.t.fit(returns_clean * 100)
    x_hist = np.linspace((returns_clean * 100).min(), (returns_clean * 100).max(), 300)
    y_hist = stats.t.pdf(x_hist, df_fit, loc, scale)
    fig_ret.add_trace(
        go.Scatter(x=x_hist, y=y_hist, mode='lines',
                   line=dict(color='blue', width=2, dash='dash'),
                   name='Student-t fit'),
        row=2, col=1
    )
    
    # KDE curve
    kde = gaussian_kde(returns_clean * 100)
    x_kde = np.linspace((returns_clean * 100).min(), (returns_clean * 100).max(), 300)
    y_kde = kde(x_kde)
    fig_ret.add_trace(
        go.Scatter(x=x_kde, y=y_kde, mode='lines',
                   line=dict(color='orange', width=2),
                   name='KDE'),
        row=2, col=1
    )
    
    # Annotation Student-t params
    param_annotation = f"Student-t params<br>df = {df_fit:.2f}<br>μ = {loc:.4f}<br>σ = {scale:.4f}"
    fig_ret.add_annotation(
        text=param_annotation, xref='x2 domain', yref='y2 domain',
        x=0.98, y=0.95, showarrow=False,
        font=dict(size=11), align='right',
        bgcolor='white', bordercolor='black', borderwidth=1
    )
    
    fig_ret.update_yaxes(title_text=f'{freq} return', row=1, col=1)
    # Remove the redundant 'Date' title to avoid overlap with subplot title below
    # fig_ret.update_xaxes(title_text='Date', row=1, col=1)
    fig_ret.update_yaxes(title_text='Density', row=2, col=1)
    fig_ret.update_xaxes(title_text='Return (%)', row=2, col=1)
    apply_plotly_layout(fig_ret, height=800)
    st.plotly_chart(fig_ret, key="chart_returns", use_container_width=True)

# =====================================================
# Onglet 4 : Nuage de points
# =====================================================
with tab_scatter:
    returns_t = returns_clean.values[1:]
    returns_tm1 = returns_clean.values[:-1]
    
    # Regression
    x_mean = returns_tm1.mean()
    y_mean = returns_t.mean()
    a = ((returns_tm1 - x_mean) * (returns_t - y_mean)).sum() / ((returns_tm1 - x_mean) ** 2).sum()
    b = y_mean - a * x_mean
    x_line = np.array([returns_tm1.min(), returns_tm1.max()])
    y_line = a * x_line + b
    
    fig_sc = make_subplots(rows=1, cols=1, subplot_titles=[f'<b>Scatter plot: Return t vs t-1 ({freq})</b>'])
    fig_sc.add_trace(
        go.Scatter(x=returns_tm1, y=returns_t, mode='markers',
                   marker=dict(color='#1f77b4', size=5, opacity=0.5),
                   name='Returns'),
        row=1, col=1
    )
    fig_sc.add_trace(
        go.Scatter(x=x_line, y=y_line, mode='lines',
                   line=dict(color='red', width=2),
                   name=f'Regression: y = {a:.2f}x + {b:.2g}'),
        row=1, col=1
    )
    apply_plotly_layout(fig_sc,
                        xaxis_title='Return t-1',
                        yaxis_title='Return t',
                        height=650)
    st.plotly_chart(fig_sc, key="chart_scatter", use_container_width=True)

# =====================================================
# Onglet 5 : Moments glissants
# =====================================================
with tab_moments:
    rolling_window_val = st.number_input("Rolling window size (days)", min_value=5, max_value=10000, value=252, key="rolling_moments_win")
    
    if st.button("Compute rolling moments", key="btn_moments"):
        dates_mom = df['Date'].iloc[1:]
        rolling_mean = returns_clean.rolling(rolling_window_val).mean()
        rolling_vol = returns_clean.rolling(rolling_window_val).std()
        rolling_skew_vals = returns_clean.rolling(rolling_window_val).apply(skew, raw=True)
        rolling_kurt_vals = returns_clean.rolling(rolling_window_val).apply(kurtosis, raw=True)
        
        fig_mom = make_subplots(
            rows=2, cols=2,
            subplot_titles=['<b>Time-varying mean</b>', '<b>Time-varying volatility</b>',
                            '<b>Time-varying skewness</b>', '<b>Time-varying kurtosis</b>'],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        fig_mom.add_trace(
            go.Scatter(x=dates_mom, y=rolling_mean, mode='lines',
                       line=dict(color='black', width=1.5),
                       name=f'Mean ({rolling_window_val}d)'),
            row=1, col=1
        )
        fig_mom.add_hline(y=0, line=dict(color='gray', width=0.7), row=1, col=1)
        
        fig_mom.add_trace(
            go.Scatter(x=dates_mom, y=rolling_vol, mode='lines',
                       line=dict(color='black', width=1.5),
                       name=f'Volatility ({rolling_window_val}d)'),
            row=1, col=2
        )
        
        fig_mom.add_trace(
            go.Scatter(x=dates_mom, y=rolling_skew_vals, mode='lines',
                       line=dict(color='black', width=1.5),
                       name=f'Skewness ({rolling_window_val}d)'),
            row=2, col=1
        )
        fig_mom.add_hline(y=0, line=dict(color='gray', width=0.7), row=2, col=1)
        
        fig_mom.add_trace(
            go.Scatter(x=dates_mom, y=rolling_kurt_vals, mode='lines',
                       line=dict(color='black', width=1.5),
                       name=f'Kurtosis ({rolling_window_val}d)'),
            row=2, col=2
        )
        
        fig_mom.update_yaxes(title_text='Mean', row=1, col=1)
        fig_mom.update_yaxes(title_text='Volatility', row=1, col=2)
        fig_mom.update_yaxes(title_text='Skewness', row=2, col=1)
        fig_mom.update_yaxes(title_text='Kurtosis', row=2, col=2)
        for r in [1, 2]:
            for c in [1, 2]:
                if r == 2: # Only show 'Date' on the bottom row
                    fig_mom.update_xaxes(title_text='Date', row=r, col=c)
        
        apply_plotly_layout(fig_mom, height=750)
        st.plotly_chart(fig_mom, key="chart_moments", use_container_width=True)

# =====================================================
# Onglet 6 : QQ-Plot + ACF + PACF
# =====================================================
with tab_qq:
    # QQ-Plot using Plotly
    sorted_returns = np.sort(returns_clean.values)
    n = len(sorted_returns)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    
    # Regression line for QQ plot
    slope, intercept, _, _, _ = stats.linregress(theoretical_quantiles, sorted_returns)
    qq_line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    qq_line_y = slope * qq_line_x + intercept
    
    fig_qq = make_subplots(
        rows=2, cols=2,
        row_heights=[0.6, 0.4],
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=['<b>QQ-Plot of returns</b>', '<b>ACF of returns</b>', '<b>PACF of returns</b>'],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    fig_qq.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_returns, mode='markers',
                   marker=dict(color='#1f77b4', size=4, opacity=0.6),
                   name='Returns'),
        row=1, col=1
    )
    fig_qq.add_trace(
        go.Scatter(x=qq_line_x, y=qq_line_y, mode='lines',
                   line=dict(color='red', width=2),
                   name='Normal reference'),
        row=1, col=1
    )
    
    # Test de Jarque-Bera
    jb_stat, jb_pvalue = jarque_bera(returns_clean)
    normality = "Normal" if jb_pvalue > 0.05 else "Non-Normal"
    label_jb = f"JB = {jb_stat:.2f}<br>{normality}"
    fig_qq.add_annotation(
        text=label_jb, xref='x domain', yref='y domain',
        x=0.98, y=0.05, showarrow=False,
        font=dict(size=12), align='right',
        bgcolor='white', bordercolor='gray', borderwidth=1
    )
    
    # ACF
    n_lags = 10
    acf_values = acf(returns_clean, nlags=n_lags, fft=True, alpha=0.05)
    acf_vals = acf_values[0][1:]  # skip lag 0
    lags = np.arange(1, n_lags + 1)
    
    # Confidence band (approximate 95%)
    conf_bound = 1.96 / np.sqrt(len(returns_clean))
    
    for i, lag in enumerate(lags):
        fig_qq.add_trace(
            go.Bar(x=[lag], y=[acf_vals[i]], marker_color='#1f77b4',
                   width=0.4, showlegend=False),
            row=2, col=1
        )
    fig_qq.add_hline(y=conf_bound, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
    fig_qq.add_hline(y=-conf_bound, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
    fig_qq.add_hline(y=0, line=dict(color='black', width=0.5), row=2, col=1)
    
    # PACF
    pacf_values = pacf(returns_clean, nlags=n_lags, method='ywm', alpha=0.05)
    pacf_vals = pacf_values[0][1:]
    
    for i, lag in enumerate(lags):
        fig_qq.add_trace(
            go.Bar(x=[lag], y=[pacf_vals[i]], marker_color='#1f77b4',
                   width=0.4, showlegend=False),
            row=2, col=2
        )
    fig_qq.add_hline(y=conf_bound, line=dict(color='red', width=1, dash='dash'), row=2, col=2)
    fig_qq.add_hline(y=-conf_bound, line=dict(color='red', width=1, dash='dash'), row=2, col=2)
    fig_qq.add_hline(y=0, line=dict(color='black', width=0.5), row=2, col=2)
    
    fig_qq.update_xaxes(title_text='Theoretical Quantiles', row=1, col=1)
    fig_qq.update_yaxes(title_text='Sample Quantiles', row=1, col=1)
    fig_qq.update_xaxes(title_text='Lag', dtick=1, row=2, col=1)
    fig_qq.update_xaxes(title_text='Lag', dtick=1, row=2, col=2)
    fig_qq.update_yaxes(title_text='ACF', row=2, col=1)
    fig_qq.update_yaxes(title_text='PACF', row=2, col=2)
    
    apply_plotly_layout(fig_qq, height=800, showlegend=False)
    st.plotly_chart(fig_qq, key="chart_qq", use_container_width=True)

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
        fig_cr = make_subplots(rows=1, cols=1, subplot_titles=['<b>Comparative Cumulative Return</b>'])
        
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
            fig_cr.add_trace(
                go.Scatter(x=df_asset['Date'].iloc[1:], y=cum_asset, mode='lines',
                           line=dict(width=1.5), name=label),
                row=1, col=1
            )
        
        apply_plotly_layout(fig_cr,
                            xaxis_title='Date',
                            yaxis_title='Cumulative Return',
                            height=550)
        st.plotly_chart(fig_cr, key="chart_cumret", use_container_width=True)
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
            
            fig_roll = make_subplots(
                rows=2, cols=1,
                row_heights=[0.65, 0.35],
                vertical_spacing=0.12,
                subplot_titles=[f'<b>Rolling Return {actif} ({window_label})</b>', '<b>Rolling returns distribution</b>']
            )
            
            # Série temporelle du rolling return
            mean_rr = rolling_return.mean()
            median_rr = rolling_return.median()
            
            fig_roll.add_trace(
                go.Scatter(x=df_roll['Date'], y=rolling_return, mode='lines',
                           line=dict(color='#4F8EF7', width=1.5), name='Rolling Return'),
                row=1, col=1
            )
            fig_roll.add_hline(y=mean_rr, line=dict(color='black', width=2),
                               annotation_text=f'Mean ({mean_rr:.2%})',
                               row=1, col=1)
            fig_roll.add_hline(y=median_rr, line=dict(color='orange', width=2, dash='dash'),
                               annotation_text=f'Median ({median_rr:.2%})',
                               row=1, col=1)
            
            # Histogramme
            rolling_return_clean = rolling_return.dropna()
            fig_roll.add_trace(
                go.Histogram(x=rolling_return_clean * 100, nbinsx=30,
                             histnorm='probability density',
                             marker_color='rgba(79,142,247,0.75)',
                             name='Distribution'),
                row=2, col=1
            )
            
            prob_neg = (rolling_return_clean < 0).mean()
            fig_roll.add_vline(x=0, line=dict(color='red', width=2, dash='dash'),
                               annotation_text=f'P(R < 0%) = {prob_neg:.1%}',
                               row=2, col=1)
            
            fig_roll.update_yaxes(title_text='Rolling Return', row=1, col=1)
            # Remove redundant 'Date' title
            # fig_roll.update_xaxes(title_text='Date', row=1, col=1)
            fig_roll.update_yaxes(title_text='Density', row=2, col=1)
            fig_roll.update_xaxes(title_text='Rolling return (%)', row=2, col=1)
            
            apply_plotly_layout(fig_roll, height=800)
            st.plotly_chart(fig_roll, key="chart_rolling", use_container_width=True)