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

# from pandas.io import gbq

# import pandas_gbq

# df = gbq.read_gbq("select * from `parabolic-hook-303116.Jobs.Daily_scraping`",project_id = "parabolic-hook-303116")
#df = gbq.read_gbq("select * from `parabolic-hook-303116.Jobs.Daily_scraping`", project_id="parabolic-hook-303116")
df = pd.read_csv("scraping_data.csv")
df = df.sort_values('Posted_date')
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1('Jobs Information'),
 html.Div([
    html.Img(src='/assets/img.png')
]),
        html.Br(),
        html.Label([''], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='country',
                     options=[{'label': x, 'value': x} for x in df.Country_name.unique()],
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
                     options=[{'label': x, 'value': x} for x in df.Job_type.unique()],
                     value=['Accounting', 'Project Management'],
                     multi=True,
                     clearable=True,
                     placeholder="Certificate",
                     persistence='string',
                     style={'width': "50%"},
                     persistence_type='session'),

    ], className='three_columns'),

    html.Div([
        dcc.Graph(id='our_graph')
    ], className='nine_columns'),

])


@app.callback(
    dash.dependencies.Output('our_graph', 'figure'),
    [dash.dependencies.Input('country', 'value'),
     dash.dependencies.Input('job', 'value')])
def build_graph(first, second):
    dff = df[(df['Country_name'] == first) &
             (df['Job_type'].isin(second))]
    # print(dff[:5])

    fig = px.line(dff, x="Posted_date", y="Total", color="Job_type",labels={
                     "Posted_date": "Date",
                     "Total": "Total Entry level jobs",
                     "Job_type": "Certificates"
                 },
                title="Total Job Posting by Day",color_discrete_sequence=["#3A929D", "#FFC328", "#6ABDC8","#B88300",'#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52','#EF553B', '#00CC96'])
    fig.update_layout(plot_bgcolor="white")
    fig.update_traces(line=dict(width=4))
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=False)

