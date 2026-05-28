CREATE TABLE SYSTEM_ASSET_MASTER (
    asset_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker_symbol VARCHAR2(15) NOT NULL UNIQUE,
    asset_name VARCHAR2(100) NOT NULL,
    asset_class VARCHAR2(30) CHECK (asset_class IN ('EQUITY', 'ETF', 'MUTUAL_FUND', 'BOND', 'CRYPTO')),
    sector_segment VARCHAR2(50) NOT NULL,
    macro_benchmark VARCHAR2(15) DEFAULT 'NIFTY50',
    CONSTRAINT uq_ticker UNIQUE (ticker_symbol)
);

-- 2. UNIVERSAL TRANSACTION LEDGER MULTI-BROKER/MULTI-ASSET INTEGRATION
CREATE TABLE TRANSACTION_LEDGER (
    tx_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    broker_source VARCHAR2(30) CHECK (broker_source IN ('FINECO', 'ZERODHA', 'GROWW', 'INTERACTIVE_BROKERS')),
    trade_date TIMESTAMP NOT NULL,
    ticker_symbol VARCHAR2(15) NOT NULL,
    tx_type VARCHAR2(10) CHECK (tx_type IN ('BUY', 'SELL', 'DIVIDEND_REINVEST')),
    quantity NUMBER(18,4) NOT NULL,
    execution_price NUMBER(14,4) NOT NULL,
    execution_currency VARCHAR2(5) DEFAULT 'INR',
    fx_rate_to_base NUMBER(14,6) DEFAULT 1.000000,
    transaction_fees NUMBER(10,2) DEFAULT 0.00,
    record_version NUMBER DEFAULT 1 NOT NULL, -- HISTORICAL VERSIONING LAYER
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_ledger_ticker FOREIGN KEY (ticker_symbol) REFERENCES SYSTEM_ASSET_MASTER(ticker_symbol)
);

-- 3. CENTRALIZED CASH FLOW RECONCILIATION GENERAL LEDGER
CREATE TABLE PORTFOLIO_CASH_FLOWS (
    flow_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    broker_source VARCHAR2(30) NOT NULL,
    flow_date TIMESTAMP NOT NULL,
    amount_base_curr NUMBER(18,2) NOT NULL, -- Positive for deposits, Negative for withdrawals
    flow_type VARCHAR2(20) CHECK (flow_type IN ('DEPOSIT', 'WITHDRAWAL', 'DIVIDEND_CASH')),
    cleared_timestamp TIMESTAMP
);

-- 4. CACHED TIMESERIES HISTORICAL MARKET DATA POOL
CREATE TABLE MARKET_PRICE_FEED (
    feed_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker_symbol VARCHAR2(15) NOT NULL,
    price_date DATE NOT NULL,
    open_price NUMBER(14,4) NOT NULL,
    high_price NUMBER(14,4) NOT NULL,
    low_price NUMBER(14,4) NOT NULL,
    close_price NUMBER(14,4) NOT NULL,
    adjusted_close NUMBER(14,4) NOT NULL,
    trading_volume NUMBER(20) NOT NULL,
    fama_french_smb NUMBER(10,6), -- COMPUTE FACTORS ONBOARDING POOL
    fama_french_hml NUMBER(10,6),
    CONSTRAINT fk_feed_ticker FOREIGN KEY (ticker_symbol) REFERENCES SYSTEM_ASSET_MASTER(ticker_symbol),
    CONSTRAINT uq_ticker_date UNIQUE (ticker_symbol, price_date)
);

-- 5. CACHED TIME-SERIES FOREIGN EXCHANGE (FX) ENGINE DATA POOL
CREATE TABLE FX_CONVERSION_RATES (
    rate_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    currency_pair VARCHAR2(10) NOT NULL, -- e.g., 'USD/EUR', 'USD/INR'
    rate_date DATE NOT NULL,
    conversion_rate NUMBER(14,6) NOT NULL,
    CONSTRAINT uq_pair_date UNIQUE (currency_pair, rate_date)
);

-- 6. PORTFOLIO DAILY INCREMENTAL METRIC SNAPSHOTS
CREATE TABLE PORTFOLIO_DAILY_SNAPSHOTS (
    snapshot_date DATE PRIMARY KEY,
    total_market_value NUMBER(18,2) NOT NULL,
    total_asset_value NUMBER(18,2) NOT NULL,
    cash_balance_value NUMBER(18,2) NOT NULL,
    daily_twr_return NUMBER(10,6) NOT NULL,
    daily_benchmark_return NUMBER(10,6) NOT NULL,
    rolling_sharpe_180d NUMBER(8,4),
    rolling_vol_180d NUMBER(8,4),
    rolling_beta_180d NUMBER(8,4)
);

-- ============================================================================
-- PERFORMANCE CRITICAL COMPOSITE INDEXING LOOKUPS (SECTION 5 OPTIMIZATION)
-- ============================================================================
CREATE INDEX idx_ledger_perf ON TRANSACTION_LEDGER(trade_date, ticker_symbol);
CREATE INDEX idx_prices_perf ON MARKET_PRICE_FEED(price_date, ticker_symbol);
CREATE INDEX idx_cash_perf ON PORTFOLIO_CASH_FLOWS(flow_date);