"""Trend Dashboard - Dash Version

Momentum Strategy Analysis App using Dash framework.
This is the main entry point for the Dash application.
"""

import dash
import dash_bootstrap_components as dbc
from dash import DiskcacheManager
import diskcache

from dash_components.layout import create_layout
from dash_components.callbacks import register_callbacks

# Setup diskcache for long callbacks
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
    title="Trend Dashboard",
    update_title="Loading...",
)

# Create layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

# Expose server for gunicorn
server = app.server

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
