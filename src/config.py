"""Configuration and default settings for the trend dashboard."""

# Default ticker list for alpha strategies
ALPHA_LIST = [
    # US industry and theme (8)
    'SMH', 'IGV', 'XAR', 'XBI', 'XME', 'XOP', 'PAVE', 'ARKK',
    # US style (7)
    'MGK', 'MGV', 'IWM', 'SCHD', 'USMV', 'MTUM', 'QUAL',
    # Global sector and theme (9)
    'GDX', 'URA', 'IXN', '372330.KS', '283580.KS', 'IGF', 'BOTZ', 'SKYY', 'ICLN',
]

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
    '372330.KS': 'Korea ESG (KODEX)',
    '283580.KS': 'Korea Dividend (KODEX)',
    'IGF': 'Global Infrastructure',
    'BOTZ': 'Robotics & AI',
    'SKYY': 'Cloud Computing',
    'ICLN': 'Clean Energy',
}

# Default parameters
DEFAULT_START_DATE = '2000-01-01'
DEFAULT_BACKTEST_START_DATE = '2015-01-01'
DEFAULT_SHORT_WINDOW = 4
DEFAULT_MID_WINDOW = 7
DEFAULT_LONG_WINDOW = 11
DEFAULT_SHORT_WEIGHT = 0.05
DEFAULT_MID_WEIGHT = 0.05
DEFAULT_LONG_WEIGHT = 0.90
DEFAULT_TOP_K = 5
DEFAULT_N_QUANTILES = 5
DEFAULT_THRESH = 10
DEFAULT_CAPITAL = 1000
DEFAULT_TCOST = 0.002

# Weight method options
WEIGHT_METHODS = {
    "Equal Weight": "equal",
    "Inverse Volatility": "inverse_vol",
    "Rank Weight": "rank",
}
