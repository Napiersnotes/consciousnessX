#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consciousness Dashboard
Real-time visualization and monitoring for consciousness simulations
Production-ready with Dash/Plotly and comprehensive visualization tools
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import logging
import threading
import time
import json
import pickle
import base64
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import warnings
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessMetrics:
    """Container for consciousness metrics"""
    
    def __init__(self):
        self.metrics = {
            'phi': deque(maxlen=1000),  # Integrated Information
            'coherence': deque(maxlen=1000),  # Quantum coherence
            'collapse_rate': deque(maxlen=1000),  # Collapse rate (Hz)
            'consciousness_level': deque(maxlen=1000),  # Consciousness level
            'self_reference': deque(maxlen=1000),  # Self-reference score
            'complexity': deque(maxlen=1000),  # Complexity measure
            'temporal_stability': deque(maxlen=1000),  # Temporal stability
            'emergence_probability': deque(maxlen=1000)  # Emergence probability
        }
        
        self.timestamps = deque(maxlen=1000)
        self.alerts = []
        
        # Thresholds for alerts
        self.thresholds = {
            'phi_critical': 0.6,
            'phi_warning': 0.3,
            'coherence_min': 0.1,
            'collapse_rate_max': 1000.0,  # Hz
            'emergence_probability_warning': 0.7
        }
    
    def add_measurement(self, 
                       timestamp: float,
                       phi: float,
                       coherence: float,
                       collapse_rate: float,
                       consciousness_level: str,
                       **kwargs):
        """Add a new measurement"""
        self.timestamps.append(timestamp)
        
        self.metrics['phi'].append(phi)
        self.metrics['coherence'].append(coherence)
        self.metrics['collapse_rate'].append(collapse_rate)
        
        # Convert consciousness level to numeric for plotting
        level_map = {
            'Pre-conscious': 0.0,
            'Proto-conscious': 0.33,
            'Emergent consciousness': 0.66,
            'Full consciousness': 1.0
        }
        numeric_level = level_map.get(consciousness_level, 0.0)
        self.metrics['consciousness_level'].append(numeric_level)
        
        # Add optional metrics
        for key in ['self_reference', 'complexity', 'temporal_stability', 'emergence_probability']:
            if key in kwargs:
                self.metrics[key].append(kwargs[key])
        
        # Check for alerts
        self._check_alerts(timestamp, phi, coherence, collapse_rate, consciousness_level)
    
    def _check_alerts(self, 
                     timestamp: float,
                     phi: float,
                     coherence: float,
                     collapse_rate: float,
                     consciousness_level: str):
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        # Check Φ thresholds
        if phi >= self.thresholds['phi_critical']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'CRITICAL',
                'message': f'Φ exceeded critical threshold: {phi:.3f} >= {self.thresholds["phi_critical"]}',
                'metric': 'phi',
                'value': phi
            })
        elif phi >= self.thresholds['phi_warning']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'WARNING',
                'message': f'Φ exceeded warning threshold: {phi:.3f} >= {self.thresholds["phi_warning"]}',
                'metric': 'phi',
                'value': phi
            })
        
        # Check coherence
        if coherence < self.thresholds['coherence_min']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'WARNING',
                'message': f'Coherence below minimum: {coherence:.3f} < {self.thresholds["coherence_min"]}',
                'metric': 'coherence',
                'value': coherence
            })
        
        # Check collapse rate
        if collapse_rate > self.thresholds['collapse_rate_max']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'WARNING',
                'message': f'Collapse rate too high: {collapse_rate:.1f} Hz > {self.thresholds["collapse_rate_max"]} Hz',
                'metric': 'collapse_rate',
                'value': collapse_rate
            })
        
        # Check consciousness level
        if consciousness_level == 'Full consciousness':
            alerts.append({
                'timestamp': timestamp,
                'level': 'INFO',
                'message': 'Full consciousness detected',
                'metric': 'consciousness_level',
                'value': consciousness_level
            })
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        return alerts
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame"""
        data = {
            'timestamp': list(self.timestamps),
            'phi': list(self.metrics['phi']),
            'coherence': list(self.metrics['coherence']),
            'collapse_rate': list(self.metrics['collapse_rate']),
            'consciousness_level': list(self.metrics['consciousness_level'])
        }
        
        # Add optional metrics if available
        for key in ['self_reference', 'complexity', 'temporal_stability', 'emergence_probability']:
            if len(self.metrics[key]) == len(self.timestamps):
                data[key] = list(self.metrics[key])
        
        return pd.DataFrame(data)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.timestamps:
            return {}
        
        df = self.get_dataframe()
        
        stats = {
            'current': {
                'phi': self.metrics['phi'][-1] if self.metrics['phi'] else 0.0,
                'coherence': self.metrics['coherence'][-1] if self.metrics['coherence'] else 0.0,
                'collapse_rate': self.metrics['collapse_rate'][-1] if self.metrics['collapse_rate'] else 0.0,
                'consciousness_level': self.metrics['consciousness_level'][-1] if self.metrics['consciousness_level'] else 0.0
            },
            'stats': {
                'phi_mean': df['phi'].mean() if 'phi' in df.columns else 0.0,
                'phi_std': df['phi'].std() if 'phi' in df.columns else 0.0,
                'coherence_mean': df['coherence'].mean() if 'coherence' in df.columns else 0.0,
                'collapse_rate_mean': df['collapse_rate'].mean() if 'collapse_rate' in df.columns else 0.0
            },
            'alerts': {
                'total': len(self.alerts),
                'critical': sum(1 for a in self.alerts if a['level'] == 'CRITICAL'),
                'warning': sum(1 for a in self.alerts if a['level'] == 'WARNING'),
                'info': sum(1 for a in self.alerts if a['level'] == 'INFO')
            },
            'timestamps': {
                'first': self.timestamps[0] if self.timestamps else 0.0,
                'last': self.timestamps[-1] if self.timestamps else 0.0,
                'duration': (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) >= 2 else 0.0
            }
        }
        
        return stats

class ConsciousnessDashboard:
    """
    Main dashboard class for consciousness monitoring
    """
    
    def __init__(self, 
                 title: str = "ConsciousnessX Dashboard",
                 host: str = "127.0.0.1",
                 port: int = 8050,
                 debug: bool = False):
        
        self.title = title
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize metrics container
        self.metrics = ConsciousnessMetrics()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title=title
        )
        
        # Layout
        self.app.layout = self._create_layout()
        
        # Callbacks
        self._register_callbacks()
        
        # Data simulation thread (for demo)
        self.simulation_thread = None
        self.simulation_running = False
        
        logger.info(f"Consciousness dashboard initialized at http://{host}:{port}")
    
    def _create_layout(self) -> html.Div:
        """Create dashboard layout"""
        return html.Div([
            # Header
            dbc.Navbar(
                dbc.Container([
                    html.A(
                        dbc.Row([
                            dbc.Col(html.Img(src="/assets/logo.png", height="30px")),
                            dbc.Col(dbc.NavbarBrand(self.title, className="ms-2")),
                        ], align="center", className="g-0"),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    
                    dbc.NavbarToggler(id="navbar-toggler"),
                    
                    dbc.Collapse(
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                            dbc.NavItem(dbc.NavLink("Metrics", href="#")),
                            dbc.NavItem(dbc.NavLink("Alerts", href="#")),
                            dbc.NavItem(dbc.NavLink("Settings", href="#")),
                            dbc.NavItem(dbc.NavLink("Documentation", href="#")),
                        ], className="ms-auto", navbar=True),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ]),
                color="dark",
                dark=True,
                sticky="top",
            ),
            
            # Main content
            dbc.Container([
                dbc.Row([
                    # Left column: Summary cards
                    dbc.Col([
                        self._create_summary_cards(),
                        self._create_alert_panel(),
                    ], width=3),
                    
                    # Right column: Charts
                    dbc.Col([
                        self._create_charts_tabs(),
                    ], width=9),
                ]),
                
                # Control panel
                dbc.Row([
                    dbc.Col([
                        self._create_control_panel(),
                    ], width=12),
                ], className="mt-4"),
                
                # Footer
                dbc.Row([
                    dbc.Col([
                        html.Hr(),
                        html.P(
                            "ConsciousnessX Dashboard v1.0 | "
                            "Penrose-Orch-OR Simulation Framework | "
                            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            className="text-center text-muted"
                        ),
                    ], width=12),
                ], className="mt-4"),
            ], fluid=True, className="mt-3"),
            
            # Hidden div for storing data
            dcc.Store(id='metrics-store'),
            dcc.Store(id='alerts-store'),
            
            # Interval for updates
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            ),
        ])
    
    def _create_summary_cards(self) -> dbc.Card:
        """Create summary cards"""
        return dbc.Card([
            dbc.CardHeader("Consciousness Status", className="text-center"),
            dbc.CardBody([
                # Φ card
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Integrated Information (Φ)", className="card-title"),
                        html.H2("0.000", id="phi-value", className="card-text text-center"),
                        html.P("Pre-conscious", id="phi-status", className="card-text text-center"),
                    ]),
                ], className="mb-3", color="secondary", inverse=True),
                
                # Coherence card
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Quantum Coherence", className="card-title"),
                        html.H2("0.000", id="coherence-value", className="card-text text-center"),
                        html.P("Low", id="coherence-status", className="card-text text-center"),
                    ]),
                ], className="mb-3", color="info", inverse=True),
                
                # Collapse rate card
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Collapse Rate", className="card-title"),
                        html.H2("0.0", id="collapse-rate-value", className="card-text text-center"),
                        html.P("Hz", className="card-text text-center"),
                    ]),
                ], className="mb-3", color="warning", inverse=True),
                
                # Consciousness level card
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Consciousness Level", className="card-title"),
                        html.H2("N/A", id="consciousness-level-value", className="card-text text-center"),
                        html.P("Status", id="consciousness-level-status", className="card-text text-center"),
                    ]),
                ], color="success", inverse=True),
            ]),
        ])
    
    def _create_alert_panel(self) -> dbc.Card:
        """Create alert panel"""
        return dbc.Card([
            dbc.CardHeader("Recent Alerts", className="text-center"),
            dbc.CardBody([
                html.Div(id="alerts-list", children=[
                    html.P("No alerts", className="text-muted text-center"),
                ], style={"maxHeight": "300px", "overflowY": "auto"}),
            ]),
        ], className="mt-3")
    
    def _create_charts_tabs(self) -> dbc.Card:
        """Create charts in tabs"""
        return dbc.Card([
            dbc.CardHeader(
                dbc.Tabs([
                    dbc.Tab(label="Φ & Coherence", tab_id="tab-phi"),
                    dbc.Tab(label="Collapse Patterns", tab_id="tab-collapse"),
                    dbc.Tab(label="Consciousness Evolution", tab_id="tab-evolution"),
                    dbc.Tab(label="3D Visualization", tab_id="tab-3d"),
                ], id="charts-tabs", active_tab="tab-phi"),
            ),
            dbc.CardBody([
                html.Div(id="charts-content"),
            ]),
        ])
    
    def _create_control_panel(self) -> dbc.Card:
        """Create control panel"""
        return dbc.Card([
            dbc.CardHeader("Simulation Controls", className="text-center"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Start Simulation", 
                                  id="start-button", 
                                  color="success", 
                                  className="w-100"),
                    ], width=3),
                    dbc.Col([
                        dbc.Button("Pause Simulation", 
                                  id="pause-button", 
                                  color="warning", 
                                  className="w-100"),
                    ], width=3),
                    dbc.Col([
                        dbc.Button("Reset Simulation", 
                                  id="reset-button", 
                                  color="danger", 
                                  className="w-100"),
                    ], width=3),
                    dbc.Col([
                        dbc.Button("Export Data", 
                                  id="export-button", 
                                  color="info", 
                                  className="w-100"),
                    ], width=3),
                ]),
                
                html.Hr(),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Simulation Speed"),
                        dcc.Slider(
                            id="speed-slider",
                            min=0.1,
                            max=10.0,
                            step=0.1,
                            value=1.0,
                            marks={i: f"{i}x" for i in [0.1, 1.0, 5.0, 10.0]},
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Alert Thresholds"),
                        dbc.Checklist(
                            options=[
                                {"label": "Enable Critical Alerts", "value": "critical"},
                                {"label": "Enable Warning Alerts", "value": "warning"},
                                {"label": "Enable Info Alerts", "value": "info"},
                            ],
                            value=["critical", "warning", "info"],
                            id="alerts-checklist",
                            inline=True,
                        ),
                    ], width=6),
                ]),
            ]),
        ])
    
    def _register_callbacks(self):
        """Register Dash callbacks"""
        
        # Update summary cards
        @self.app.callback(
            [Output('phi-value', 'children'),
             Output('phi-status', 'children'),
             Output('coherence-value', 'children'),
             Output('coherence-status', 'children'),
             Output('collapse-rate-value', 'children'),
             Output('consciousness-level-value', 'children'),
             Output('consciousness-level-status', 'children'),
             Output('metrics-store', 'data')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_summary(n_intervals):
            # Get latest metrics
            stats = self.metrics.get_summary_stats()
            
            if not stats:
                return ["0.000", "No data", "0.000", "No data", "0.0", "N/A", "No data", {}]
            
            current = stats['current']
            
            # Format Φ
            phi_value = f"{current['phi']:.3f}"
            if current['phi'] < 0.1:
                phi_status = "Pre-conscious"
            elif current['phi'] < 0.3:
                phi_status = "Proto-conscious"
            elif current['phi'] < 0.6:
                phi_status = "Emergent consciousness"
            else:
                phi_status = "Full consciousness"
            
            # Format coherence
            coherence_value = f"{current['coherence']:.3f}"
            if current['coherence'] < 0.2:
                coherence_status = "Low"
            elif current['coherence'] < 0.5:
                coherence_status = "Medium"
            else:
                coherence_status = "High"
            
            # Format collapse rate
            collapse_rate_value = f"{current['collapse_rate']:.1f}"
     
