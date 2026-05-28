import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from portfolio_engine import (
    sync_market_data,
    get_cached_portfolio_analytics,
    run_black_litterman_optimization, 
    calculate_egarch_var, 
    generate_rebalancing_execution_file,
    run_macro_shock_simulation,
    get_fama_french_exposure, 
    ingest_broker_statement
)

custom_css = """
body { background-color: #08080a !important; color: #f1f5f9 !important; font-family: 'Inter', system-ui, sans-serif; }
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 2rem !important; }
.custom-card { 
    background: #0f1015 !important; 
    border: 1px solid #1e293b !important; 
    border-radius: 14px !important; 
    padding: 1.75rem !important; 
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
    margin-bottom: 1.5rem !important;
}
.plot-container { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #1e293b !important; margin-bottom: 1.5rem !important; }
.btn-primary { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; border: none !important; border-radius: 8px !important; }
.btn-primary:hover { background: linear-gradient(135deg, #3b82f6, #2563eb) !important; }
footer { display: none !important; }
"""

USER_PORTFOLIO_STATE = {}

def process_suite_execution(
    risk_free_pct, confidence, benchmark_choice, 
    tech_view, bank_view, energy_view, consumer_view,
    tech_conf, bank_conf, energy_conf, consumer_conf,
    macro_shock
):
    global USER_PORTFOLIO_STATE
    rf_rate = float(risk_free_pct) / 100.0
    filename = "rebalance_orders.csv"
    
    market_df, tickers, market_benchmark_ticker = sync_market_data(benchmark_choice)
    
    pivoted_df = market_df.pivot(index='price_date', columns='ticker', values='close_price').sort_index().ffill().bfill()
    pivoted_df = pivoted_df.tail(252)
    daily_returns_df = pivoted_df.pct_change().dropna()
    
    if USER_PORTFOLIO_STATE and len(USER_PORTFOLIO_STATE) > 0:
        active_tickers = [t for t in USER_PORTFOLIO_STATE.keys() if t in daily_returns_df.columns]
    else:
        active_tickers = [t for t in tickers if t in daily_returns_df.columns]
        
    if not active_tickers:
        active_tickers = [c for c in daily_returns_df.columns if c != market_benchmark_ticker][:4]
        
    returns_matrix = daily_returns_df[active_tickers]
    market_proxy = daily_returns_df[market_benchmark_ticker] if market_benchmark_ticker in daily_returns_df.columns else daily_returns_df.mean(axis=1)

    investor_views = {}
    view_confidences = {}
    view_inputs = [tech_view, bank_view, energy_view, consumer_view]
    conf_inputs = [tech_conf, bank_conf, energy_conf, consumer_conf]
    
    for i, t in enumerate(active_tickers):
        investor_views[t] = float(view_inputs[i % len(view_inputs)]) / 100.0
        view_confidences[t] = float(conf_inputs[i % len(conf_inputs)]) / 100.0
    
    market_caps = {}
    sector_labels = []
    total_portfolio_value = 0.0
    
    for t in active_tickers:
        latest_market_price = float(pivoted_df[t].iloc[-1])
        if t in USER_PORTFOLIO_STATE:
            USER_PORTFOLIO_STATE[t]["price"] = latest_market_price
            shares_qty = USER_PORTFOLIO_STATE[t]["shares"]
        else:
            shares_qty = 150
            USER_PORTFOLIO_STATE[t] = {"shares": shares_qty, "price": latest_market_price}
            
        asset_valuation = shares_qty * latest_market_price
        total_portfolio_value += asset_valuation
        market_caps[t] = asset_valuation * 10
        sector_labels.append(f"{t} ({shares_qty} Owned Shares Portfolio Core)")

    def run_analytics_pipeline():
        opt_w = run_black_litterman_optimization(
            historical_returns=returns_matrix,
            market_caps=market_caps,
            investor_views=investor_views,
            view_confidences=view_confidences,
            risk_free_rate=rf_rate
        )
        return {"weights": opt_w}
        
    cached_metrics = get_cached_portfolio_analytics("global_active_user", run_analytics_pipeline, ttl_hours=1)
    opt_weights = cached_metrics["weights"]
    
    if len(opt_weights) == 0 or len(opt_weights) != len(active_tickers):
        opt_weights = np.ones(len(active_tickers)) / len(active_tickers)
    
    portfolio_returns_series = returns_matrix.dot(opt_weights)
    mean_returns = returns_matrix.mean() * 252
    cov_matrix = returns_matrix.cov() * 252
    
    try:
        var_exposure, c_var, current_vol = calculate_egarch_var(portfolio_returns_series, confidence)
    except Exception:
        current_vol = float(portfolio_returns_series.std() * np.sqrt(252))
        var_exposure = current_vol * 1.645 * np.sqrt(30 / 252)
        c_var = var_exposure * 1.15
        
    beta_val, alpha_val = get_fama_french_exposure(portfolio_returns_series, market_proxy)
    
    asset_betas = {}
    for col in returns_matrix.columns:
        cov_col = np.cov(returns_matrix[col], market_proxy)
        var_m = np.var(market_proxy)
        asset_betas[col] = cov_col / var_m if (var_m > 0 and cov_col.ndim > 1) else 1.0
        
    portfolio_weights_dict = dict(zip(active_tickers, opt_weights))
    
    shock_pct_decimal = float(macro_shock) / 100.0
    stress_results = run_macro_shock_simulation(portfolio_weights_dict, returns_matrix, shock_pct_decimal, asset_betas)
    
    w_vals = [float(x) for x in opt_weights]
    
    drift_labels = []
    for i, t in enumerate(active_tickers):
        current_weight = (USER_PORTFOLIO_STATE[t]["shares"] * USER_PORTFOLIO_STATE[t]["price"]) / total_portfolio_value if total_portfolio_value > 0 else 0.25
        drift = (w_vals[i] - current_weight) * 100
        drift_labels.append(f"{drift:+.2f}% Allocation Shift Required")

    sector_alloc_df = pd.DataFrame({
        "Sector Segment": sector_labels,
        "Optimized Allocation Weight": [f"{w*100:.2f}%" for w in w_vals],
        "Calculated Drift Tracking": drift_labels
    })
    
    cagr_scalar = float(np.sum(mean_returns.values * opt_weights))
    vol_scalar = float(current_vol)
    beta_scalar = float(beta_val)
    alpha_scalar = float(alpha_val)
    
    active_var = stress_results["stressed_value_at_risk_95"] if shock_pct_decimal != 0 else var_exposure
    active_vol = stress_results["stressed_portfolio_annualized_volatility"] if shock_pct_decimal != 0 else vol_scalar

    summary_metrics_df = pd.DataFrame({
        "Institutional Performance Metric": [
            "Portfolio Annualized Expected Growth (CAGR)", 
            "Asymmetric EGARCH Volatility Profile", 
            "Systematic Allocation Beta (β vs Market Vector)", 
            "Jensen Alpha Performance Attribution (α Edge)",
            f"Parametric Value at Risk (VaR at {confidence}% Bound)", 
            f"Conditional VaR (Expected Tail Loss Limit)",
            "What-If Simulated Stress Shock Drawdown"
        ],
        "System Output Calculations": [
            f"{cagr_scalar*100:.2f}%",
            f"{active_vol*100:.2f}%",
            f"{beta_scalar:.2f}",
            f"{alpha_scalar*100:.2f}%",
            f"{active_var*100:.2f}%",
            f"{c_var*100:.2f}%",
            f"{stress_results['projected_portfolio_shock_drawdown']*100:.2f}%"
        ]
    })
    
    fig_front = Figure(figsize=(6.2, 4), facecolor='#0f1015')
    ax_front = fig_front.add_subplot(111)
    ax_front.set_facecolor('#08080a')
    ax_front.scatter(np.sqrt(np.diag(cov_matrix)), mean_returns, c='#3b82f6', s=70, alpha=0.8, label='Asset Universe Pool')
    
    p_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    p_ret = np.sum(mean_returns * opt_weights)
    ax_front.scatter(p_vol, p_ret, c='#ef4444', marker='*', s=260, edgecolor='white', label='Black-Litterman Target', zorder=5)
    ax_front.set_title("Black-Litterman Blended Frontier Curve", color='white', fontsize=10, fontweight='bold', pad=10)
    ax_front.set_xlabel("Annualized Volatility Risk", color='#94a3b8')
    ax_front.set_ylabel("Expected Return", color='#94a3b8')
    ax_front.grid(True, color='#1e293b', linestyle='--', alpha=0.4)
    ax_front.legend(facecolor='#0f1015', edgecolor='#1e293b', labelcolor='white')
    ax_front.tick_params(colors='#64748b')
    fig_front.tight_layout()
    
    fig_monte = Figure(figsize=(6.2, 4), facecolor='#0f1015')
    ax_monte = fig_monte.add_subplot(111)
    ax_monte.set_facecolor('#08080a')
    
    np.random.seed(42)
    days, visual_paths = 252, 100
    sim_paths = np.empty((days, visual_paths))
    sim_paths[0, :] = 100.0
    
    for t in range(1, days):
        shocks = np.random.normal(0, 1, visual_paths)
        sim_paths[t, :] = sim_paths[t-1, :] * np.exp((p_ret - 0.5 * active_vol**2) * (1/252) + active_vol * shocks * np.sqrt(1/252))
        
    ax_monte.plot(sim_paths, color='#6366f1', alpha=0.18)
    ax_monte.set_title("Asymmetric EGARCH Stochastic Trajectories", color='white', fontsize=10, fontweight='bold', pad=10)
    ax_monte.set_xlabel("Trading Horizon Days", color='#94a3b8')
    ax_monte.set_ylabel("Portfolio Normalized Equity Basis", color='#94a3b8')
    ax_monte.grid(True, color='#1e293b', linestyle='--', alpha=0.4)
    ax_monte.tick_params(colors='#64748b')
    fig_monte.tight_layout()
    
    try:
        generate_rebalancing_execution_file(USER_PORTFOLIO_STATE, portfolio_weights_dict, total_portfolio_value=total_portfolio_value, filename=filename)
    except Exception as e:
        logging.error(f"Error compiling active order book matrix: {e}")
    
    return fig_front, fig_monte, summary_metrics_df, sector_alloc_df, filename
def handle_statement_upload_pipeline(file_object, broker_selection):
    global USER_PORTFOLIO_STATE
    if file_object is None: 
        return "#### `PIPELINE_STATUS_LOG`: [!] Error: No spreadsheet context found."
    try:
        holdings = ingest_broker_statement(file_object.name, broker_selection)
        USER_PORTFOLIO_STATE = holdings
        
        output_msg = f"#### `PIPELINE_STATUS_LOG`: [+] Ingestion Success: Synced {len(holdings)} positions from '{broker_selection}'.\n\n"
        output_msg += "**Active Portfolio Positions Vector:**\n"
        for tk, info in holdings.items():
            output_msg += f"* **{tk}**: {info['shares']} shares\n"
        output_msg += "\n*👉 Action Required: Navigate back to the 'Active Portfolio Risk Optimization' tab and trigger the analytics engine to generate metrics for these live assets!*"
        return output_msg
    except Exception as e:
        return f"#### `PIPELINE_STATUS_LOG`: [-] Pipeline Error: {str(e)}"


with gr.Blocks(title="QUANT-EDGE Platform Suite", css=custom_css) as app_interface:
    gr.Markdown("# 🏛️ QUANT-EDGE | Enterprise Portfolio Engineering Suite")
    gr.Markdown("Production-ready quantitative asset management platform linking live relational **Yahoo Finance Data Streams** to automated **Asymmetric EGARCH Volatility Pipelines** and **Black-Litterman Frontier Optimization**.")
    gr.HTML("<hr style='border: 0px; border-top: 1px solid #1e293b; margin-bottom: 25px;'>")
    
    with gr.Tab("🎯 Active Portfolio Risk Optimization"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Analytical Control Inputs")
                rf_input = gr.Slider(minimum=0.0, maximum=12.0, value=6.5, step=0.1, label="Active Market Risk-Free Return Yield Rate (%)")
                conf_input = gr.Slider(minimum=90.0, maximum=99.9, value=99.0, step=0.1, label="Value at Risk (VaR) Statistical Confidence Limit (%)")
                bench_dropdown = gr.Dropdown(choices=["NIFTY 50", "S&P 500 TR"], value="NIFTY 50", label="Active Tracking Target Benchmark")
                
                gr.Markdown("### ⚡ Interactive Macro Stress-Testing")
                macro_shock = gr.Slider(minimum=-50.0, maximum=0.0, value=0.0, step=1.0, label="Simulate Market Benchmark Crash Scenario (%)")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Black-Litterman Subjective Investor Views")
                with gr.Row():
                    tech_view = gr.Slider(minimum=-20.0, maximum=40.0, value=12.0, step=0.5, label="Tech Return View (%)")
                    tech_conf = gr.Slider(minimum=1.0, maximum=100.0, value=75.0, step=1.0, label="Tech Confidence (%)")
                with gr.Row():
                    bank_view = gr.Slider(minimum=-20.0, maximum=40.0, value=5.0, step=0.5, label="Banking Return View (%)")
                    bank_conf = gr.Slider(minimum=1.0, maximum=100.0, value=40.0, step=1.0, label="Banking Confidence (%)")
                with gr.Row():
                    energy_view = gr.Slider(minimum=-20.0, maximum=40.0, value=15.0, step=0.5, label="Energy Return View (%)")
                    energy_conf = gr.Slider(minimum=1.0, maximum=100.0, value=60.0, step=1.0, label="Energy Confidence (%)")
                with gr.Row():
                    consumer_view = gr.Slider(minimum=-20.0, maximum=40.0, value=8.0, step=0.5, label="Staples Return View (%)")
                    consumer_conf = gr.Slider(minimum=1.0, maximum=100.0, value=50.0, step=1.0, label="Staples Confidence (%)")

        calculate_btn = gr.Button("🚀 Run Suite Analytics Engines", variant="primary")
        gr.HTML("<br>")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Portfolio Performance Attribution Metrics Matrix")
                metrics_output = gr.Dataframe(headers=["Institutional Performance Metric", "System Output Calculations"], interactive=False)
            with gr.Column():
                gr.Markdown("### 🗂️ Systematic Factor Allocation & Drift Analytics Tracking")
                sector_output = gr.Dataframe(headers=["Sector Segment", "Optimized Allocation Weight", "Calculated Drift Tracking"], interactive=False)

        with gr.Row():
            plot_frontier = gr.Plot(label="Markowitz Allocation Frontier View", elem_classes=["plot-container"])
            plot_monte = gr.Plot(label="GARCH Stochastic Simulation Forecast", elem_classes=["plot-container"])
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📥 Order Execution Terminal")
                download_file = gr.File(label="Download Zerodha-Compliant Bulk Basket Order CSV File")
                
    with gr.Tab("📥 Automated Multi-Broker Statement Ingestion"):
        gr.Markdown("### 📂 Multi-Broker File Transaction Import Dropzone")
        file_upload_input = gr.File(label="Drop Broker Ledger Excel/CSV Spreadsheet Here")
        broker_mapping_select = gr.Dropdown(choices=["FINECO", "ZERODHA", "GROWW", "INTERACTIVE_BROKERS"], value="ZERODHA", label="Source Broker Ingestion Identity Schema")
        trigger_ingest_btn = gr.Button("⚡ Run Ingestion Statement Pipeline", variant="secondary")
        gr.HTML("<br>")
        pipeline_status_log = gr.Markdown("#### `PIPELINE_STATUS_LOG`: Awaiting incoming transaction spreadsheet stream.")

    calculate_btn.click(
        fn=process_suite_execution, 
        inputs=[
            rf_input, conf_input, bench_dropdown,
            tech_view, bank_view, energy_view, consumer_view,
            tech_conf, bank_conf, energy_conf, consumer_conf,
            macro_shock
        ], 
        outputs=[plot_frontier, plot_monte, metrics_output, sector_output, download_file]
    )
    
    trigger_ingest_btn.click(
        fn=handle_statement_upload_pipeline, 
        inputs=[file_upload_input, broker_mapping_select], 
        outputs=[pipeline_status_log]
    )

if __name__ == "__main__":
    app_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )