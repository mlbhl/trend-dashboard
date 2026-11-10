"""Layout definition for Trend Dashboard Dash app: create_layout() assembles sidebar and results."""

import dash_bootstrap_components as dbc
from dash import dcc

from .sidebar_layout import create_sidebar
from .results_layout import create_main_content


def create_layout():
    """Create the complete app layout."""
    return dbc.Container(
        [
            # Stores for data
            dcc.Store(id="dataset-store"),
            dcc.Store(id="bm-data-store"),
            dcc.Store(id="params-store"),
            dcc.Store(id="nav-store"),
            dcc.Store(id="signal-store"),
            dcc.Store(id="weights-store"),
            dcc.Store(id="walk-forward-store"),
            dcc.Store(id="optimized-params-store"),

            # Download components
            dcc.Download(id="download-nav"),
            dcc.Download(id="download-dd"),
            dcc.Download(id="download-annual"),
            dcc.Download(id="download-monthly"),
            dcc.Download(id="download-holdings"),

            # Main layout
            dbc.Row(
                [create_sidebar(), create_main_content()],
                className="g-0",
            ),
        ],
        fluid=True,
        className="p-0",
    )
