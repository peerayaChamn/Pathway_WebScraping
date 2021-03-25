#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:30:39 2021

@author: peeraya
"""

import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.0)
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

from nltk import word_tokenize

from nltk.corpus import stopwords
import dash
import dash_core_components as dcc
import dash_html_components as html

# from pandas.io import gbq

# import pandas_gbq
# reading from bigquery
# df = gbq.read_gbq("select * from `parabolic-hook-303116.Jobs.Daily_scraping`", project_id="parabolic-hook-303116")
df = pd.read_csv("scraping_data.csv")
df['Posted_date'] = pd.to_datetime(df.Posted_date)
df.sort_values(by='Posted_date')
# df = df.sort_values('Posted_date')
df = df.drop_duplicates(subset=['Job_type', 'Posted_date', 'Country_name'])
df1 = pd.read_csv("all_info.csv")

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

        dcc.Dropdown(id='job',
                     options=[{'label': x, 'value': x} for x in df.Job_type.unique()],
                     value=['Accounting', 'Project Management'],
                     multi=True,
                     clearable=True,
                     placeholder="Certificate",
                     persistence='string',
                     style={'width': "50%"},
                     persistence_type='session'),
        dcc.Dropdown(id='country',
                     options=[{'label': x, 'value': x} for x in df.Country_name.unique()],
                     multi=False,
                     disabled=False,
                     value='Brazil',
                     clearable=True,
                     searchable=True,
                     placeholder="Country",
                     className='form-dropdown',
                     persistence='string',
                     persistence_type='memory'),

        dcc.Dropdown(id='web',
                     options=[{'label': x, 'value': x} for x in df.Website.unique()],
                     multi=False,
                     disabled=False,
                     value='Indeed',
                     clearable=True,
                     searchable=True,
                     placeholder="Country",
                     className='form-dropdown',
                     persistence='string',
                     persistence_type='memory'),


    ], className='drop_down'),

    html.Div([
        dcc.Graph(id='our_graph')
    ], className='search'),

    html.Div([
        dcc.Graph(id='x-time-series')
    ], style={'display': 'inline-block', 'width': '49%', 'margin-top': '30%'}),

    html.Div([
        dcc.Graph(id='map'),
    ], style={'display': 'inline-block', 'width': '49%', 'margin-top': '40%'}),

    html.Hr(),
    html.Div([
        html.H3('Job Summary',id = "key1")
    ]),


    html.Div([dcc.Dropdown(id='job1',
                           options=[{'label': x, 'value': x} for x in df1.job_type.unique()],
                           multi=False,
                           clearable=True,
                           persistence='string',
                           style={'width': '50%'},
                           value = 'accounting',
                           persistence_type='session'),

              dcc.Dropdown(id='country1',
                           options=[{'label': x, 'value': x} for x in df1.Country.unique()],
                           multi=False,
                           clearable=True,
                           value = 'Canada',
                           persistence='string',
                           style={'width': '50%'},
                           persistence_type='session')

              ], className='NLP'),

    html.Div(
        [
            dcc.Input(id="input1", type="text", placeholder=""),
            html.P("Search",id = "key")
        ]
    ),

    html.Div([
        dcc.Graph(id='NLP')
    ], style={'display': 'inline-block', 'width': '49%', 'margin-top': '10%'}),

    html.Div([
        dcc.Graph(id='NLP1')
    ], style={'display': 'inline-block', 'width': '49%', 'margin-top': '10%'}),

    html.Div(
        [
            html.P("Search Terms"),
            html.Div(id="output"),
        ]
    ),

], className='all')


@app.callback(
    dash.dependencies.Output('our_graph', 'figure'),
    [dash.dependencies.Input('country', 'value'),
     dash.dependencies.Input('job', 'value'),dash.dependencies.Input('web', 'value')])
def build_graph(first, second,third):
    dff = df[(df['Website'] == third) & (df['Country_name'] == first) &(df['Job_type'].isin(second))]

    fig = px.line(dff, x="Posted_date", y="Total", color="Job_type", labels={
        "Posted_date": "Date",
        "Total": "Total Entry level jobs",
        "Job_type": "Certificates"
    },
                  title="Total Job Posting by Day",
                  color_discrete_sequence=["#3A929D", "#FFC328", "#6ABDC8", "#B88300", '#AB63FA', '#FFA15A', '#19D3F3',
                                           '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#EF553B', '#00CC96'])
    fig.update_layout(plot_bgcolor="white")
    fig.update_traces(line=dict(width=4))
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('country', 'value'),
     dash.dependencies.Input('job', 'value')])
def build_graph1(first1, second2):
    dff = df[(df['Country_name'] == first1) &
             (df['Job_type'].isin(second2))]
    fig = px.pie(dff, values='Total', names='Job_type', hole=.5, color_discrete_sequence=["#3A929D", "#FFC328", "#6ABDC8", "#B88300", '#AB63FA', '#FFA15A', '#19D3F3','#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#EF553B', '#00CC96'])
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig


@app.callback(
    dash.dependencies.Output('map', 'figure'),
    dash.dependencies.Input('country', 'value'))

def build_graph2(first2):
    mask = df["Country_name"] == first2
    fig = px.bar(df[mask], x="Total", y="Job_type", barmode="group", height=600, color_discrete_sequence=["#3A929D"],labels={
        "Total": "Total Entry level jobs",
        "Job_type": ""
    })
    fig.update_layout(plot_bgcolor="white")
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig


stop_words = stopwords.words('english')


def clean_data(desc):
    desc = word_tokenize(desc)
    desc = [word.lower() for word in desc if word.isalpha() and len(word) > 2]
    desc = [word for word in desc if word not in stop_words]
    return desc


@app.callback(
    dash.dependencies.Output('NLP', 'figure'),
    [dash.dependencies.Input('country1', 'value'),
     dash.dependencies.Input('job1', 'value'),
     dash.dependencies.Input('input1', 'value')
     ])
def build_graph3(country1, job1, value):
    dff = df1[(df1['Country'] == country1) &
              (df1['job_type'] == job1)]
    skills = []
    if(job1=='Public Health'):
        skills = ['communication','teamwork','initiative','interpersonal','analytics','flexibility','health', 'Health']
    elif(job1=='Database'):
        skills = ['problem-solving','communication','Oracle','SQL','PowerShell','security','analytic','database']
    elif(job1=='accounting'):
        skills = ['microsoft','QuickBooks','payable','receivable', '10 key','data entry','bookkeeping','organize','tax','deadline','security']
    skills.append(value)

    tags_df = dff["summary"].apply(clean_data)
    result = tags_df.apply(Counter).sum().items()
    result = sorted(result, key=lambda kv: kv[1], reverse=True)
    result_series = pd.Series({k: v for k, v in result})
    filter_series = result_series.filter(items=skills)
    new = pd.DataFrame(filter_series)
    new = new.reset_index()
    new.columns = ['Words', 'Counts']
    fig = px.bar(new, x="Words", y="Counts", barmode="group",text ="Counts",color_discrete_sequence=["#FFC328"])
    fig.update_traces(textposition='outside')
    fig.update_layout(plot_bgcolor="white")
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig

@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('job1', 'value')])

def build_graph3(job1):
    if(job1 == 'Public Health'):
        text = 'communication, teamwork, initiative, interpersonal, analytics, flexibility, health, Health'
    if(job1=='Database'):
        text = 'Problem-solving,Communication,Oracle,SQL,PowerShell,Security,Analytics,database'
    if(job1=='accounting'):
        text = "['microsoft','QuickBooks','payable','receivable', '10 key','data entry','bookkeeping','organize','tax','deadline','security']"

    return text

@app.callback(
    dash.dependencies.Output('NLP1', 'figure'),
    [dash.dependencies.Input('country1', 'value'),
     dash.dependencies.Input('job1', 'value')])
def build_graph4(country1, job1):
    dff = df1[(df1['Country'] == country1) &
              (df1['job_type'] == job1)]

    text = " ".join(review for review in dff.summary)

    stopwords = set(STOPWORDS)
    stopwords.update(["with","will","from","within","for","which"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white",width=800,height=400).generate(text)

    # Display the generated image:
    fig = px.imshow(wordcloud)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
