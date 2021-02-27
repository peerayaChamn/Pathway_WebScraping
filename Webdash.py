#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:30:39 2021

@author: peeraya
"""

import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.0)
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.

from pandas.io import gbq
from boto.s3.connection import S3Connection
import os
import pandas_gbq

#s3 = S3Connection(os.environ['S3_KEY'])
#df = gbq.read_gbq("select * from `parabolic-hook-303116.Jobs.Daily_scraping`",project_id = "parabolic-hook-303116")

df = pd.read_csv("aaaa.csv")
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Div([

        html.Br(),
        html.Label([''], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='country',
                     options=[{'label': x, 'value': x} for x in
                              df.sort_values('Country_name')['Country_name'].unique()],
                     multi=False,
                     disabled=False,
                     clearable=True,
                     searchable=True,
                     placeholder="Country",
                     className='form-dropdown',
                     style={'width': "50%"},
                     persistence='string',
                     persistence_type='memory'),

        dcc.Dropdown(id='job',
                     options=[{'label': x, 'value': x} for x in df.sort_values('Job_type')['Job_type'].unique()],
                     multi=False,
                     clearable=False,
                     placeholder="Certificate",
                     persistence='string',
                     style={'width': "50%"},
                     persistence_type='session'),

    ], className='three columns'),

    html.Div([
        dcc.Graph(id='our_graph')
    ], className='nine columns'),

])


@app.callback(
    dash.dependencies.Output('our_graph', 'figure'),
    [dash.dependencies.Input('country', 'value'),
     dash.dependencies.Input('job', 'value')])
def build_graph(first, second):
    dff = df[(df['Country_name'] == first) &
             (df['Job_type'] == second)]
    # print(dff[:5])
    fig = px.line(dff, x="Posted_date", y="Total", height=600)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)