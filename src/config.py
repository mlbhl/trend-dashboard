"""Configuration and default settings for the trend dashboard."""

# Preset ticker sets
TICKER_PRESETS = {
    "Alpha (Default)": [
        # US industry and theme (8)
        'SMH', 'IGV', 'XAR', 'XBI', 'XME', 'XOP', 'PAVE', '487230.KS',
        # US style (6)
        'VLUE', 'SIZE', 'SCHD', 'USMV', 'MTUM', 'QUAL',
        # Global industry and theme (6)
        'GDX', '372330.KS', 'BOTZ', 'SKYY', 'ICLN', 'AIQ', 
    ],
    "US Sector": [
        # SPDR sector ETFs
        'XLK', 'XLC', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',
    ],
    "US Factor": [
        # iShares(and Schwab) factor ETFs
        'MTUM', 'QUAL', 'VLUE', 'SIZE', 'USMV', 'SCHD',
    ],
    "Global Multi-Asset": [
        # Equities
        'SPY', 'QQQ', 'MGK', 'MGV', 'EFA', 'EEM', '069500.KS', 'MCHI',
        # Bonds
        'BND', 'BNDX', 'LQD', 'USHY', 'SHV', 'TLT',
        # Alternatives
        'GLD', 'GSG', 'IYR',
    ],
    "Top Trend": [
        # Korea
        '091160.KS', '305720.KS', '385510.KS', '445290.KS', '0080G0.KS', '228790.KS', '0115D0.KS', '487240.KS', '266420.KS', '462900.KS',
        # Global
        '390390.KS', '487230.KS', '485540.KS', '0038A0.KS', '0065G0.KS', '478150.KS', '442320.KS', '0132H0.KS',
    ],
}

# Ticker descriptions for display
TICKER_DESCRIPTIONS = {
    'SMH': 'Semiconductors',
    'IGV': 'Software',
    'XAR': 'Aerospace & Defense',
    'XBI': 'Biotech',
    'XME': 'Metals & Mining',
    'XOP': 'Oil & Gas E&P',
    'PAVE': 'Infrastructure',
    'ARKK': 'Innovation',
    'MGK': 'Mega Cap Growth',
    'MGV': 'Mega Cap Value',
    'IWM': 'Small Cap',
    'SCHD': 'Dividend',
    'USMV': 'Min Volatility',
    'MTUM': 'Momentum',
    'QUAL': 'Quality',
    'GDX': 'Gold Miners',
    'URA': 'Uranium',
    'IXN': 'Global Tech',
    '372330.KS': 'China HSI Tech',
    '283580.KS': 'China CSI 300',
    '487230.KS': 'AI Power-Infra',
    'IGF': 'Global Infrastructure',
    'BOTZ': 'Robotics & AI',
    'SKYY': 'Cloud Computing',
    'ICLN': 'Clean Energy',
    'XLK': 'Technology',
    'XLC': 'Communication Services',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'VLUE': 'Value',
    'SIZE': 'Size',
    '091160.KS': 'Korea Semis',
    '305720.KS': 'Korea Battery',
    '385510.KS': 'Korea Renewable',
    '445290.KS': 'Korea Robot',
    '0080G0.KS': 'Korea Defense',
    '228790.KS': 'Korea Cosmetics',
    '0115D0.KS': 'Korea Ships',
    '487240.KS': 'Korea AI Power-Infra',
    '266420.KS': 'Korea Health',
    '462900.KS': 'Korea Bio',
    '390390.KS': 'US Semis',
    '485540.KS': 'US AI Tech',
    '0038A0.KS': 'US Humanoid',
    '0065G0.KS': 'China Tech',
    '478150.KS': 'Global Defense',
    '442320.KS': 'Global Nuclear',
    '0132H0.KS': 'US Nuclear',
    'AIQ': 'AI & Technology',
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'EFA': 'Developed ex-US',
    'EEM': 'Emerging Markets',
    'MCHI': 'China',
    '069500.KS': 'KOSPI 200',
    'IYR': 'US Real Estate',
    'GLD': 'Gold',
    'GSG': 'Commodities',
    'TLT': 'Long Treasury (20Y+)',
    'SHV': 'T-Bills (0-1Y)',
    'BND': 'US Aggregate Bond',
    'BNDX': 'Intl Bond',
    'LQD': 'IG Corporate Bond',
    'USHY': 'High Yield Bond',
}

# Annualisation: prices are resampled to every weekday ('B', exchange holidays forward-filled),
# so a year has ~261 observations. Every annualised number (CAGR, vol, Sharpe) uses this.
PERIODS_PER_YEAR = 261

# Default parameters
DEFAULT_START_DATE = '2000-01-01'
DEFAULT_BACKTEST_START_DATE = '2015-01-01'
DEFAULT_SHORT_WINDOW = 1
DEFAULT_MID_WINDOW = 3
DEFAULT_LONG_WINDOW = 11
DEFAULT_SHORT_WEIGHT = 0.10
DEFAULT_MID_WEIGHT = 0.10
DEFAULT_LONG_WEIGHT = 0.80
DEFAULT_TOP_K = 5
DEFAULT_N_QUANTILES = 5
DEFAULT_THRESH = 10
DEFAULT_WEIGHT_MODE = "custom"        # signal-window weights: "custom" (sliders) or "equal" (1/3 each)
DEFAULT_STRATEGY_MODE = "single"      # "single" or "core-satellite"
DEFAULT_CORE_WEIGHT_PCT = 70          # core allocation in core-satellite mode, %
DEFAULT_STRATEGY_TYPE = "Top-K"       # "Top-K" or "Quantile"
DEFAULT_WEIGHT_METHOD = "equal"       # see WEIGHT_METHODS
DEFAULT_TCOST_PCT = 0.0               # one-way transaction cost, % (sidebar input)
DEFAULT_BM_TYPE = "equal_weight"      # "equal_weight" or "custom"

# Implementation lag: business days from the month-end signal (close) to the rebalance (close).
#   1 = rebalance at the next business day's close, i.e. the first business day of the next month
#   0 = rebalance at the signal day's close (no lag)
# Applies everywhere (dashboard, optimizer, walk-forward, report). Business days are weekdays:
# exchange holidays are not skipped, so a holiday rebalance trades at the forward-filled prior close.
IMPLEMENTATION_LAG = 1

# Weight method options
WEIGHT_METHODS = {
    "Equal Weight": "equal",
    "Inverse Volatility": "inverse_vol",
    "Rank Weight": "rank",
}

# Default preset
DEFAULT_PRESET = "Alpha (Default)"
