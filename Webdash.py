import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.0)
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

from nltk import word_tokenize

from nltk.corpus import stopwords
df = pd.read_csv("scraping_data.csv")
df['Posted_date'] = pd.to_datetime(df.Posted_date)
df.sort_values(by='Posted_date')
df = df.sort_values('Posted_date')
df = df.drop_duplicates(subset=['Job_type', 'Posted_date', 'Country_name'])
df1 = pd.read_csv("all_info.csv")

df2 = pd.read_csv("career.csv")
df2['Posted_date'] = pd.to_datetime(df2.Posted_date)
df2.sort_values(by='Posted_date')
df2 = df2.sort_values('Posted_date')
df2 = df2.drop_duplicates(subset=['Job_type', 'Posted_date', 'Country_name'])

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color":"#f8bc34"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
       html.H2("BYU-Pathway"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Certificate", href="/", active="exact"),
                dbc.NavLink("Skill", href="/page-1", active="exact"),
                dbc.NavLink("Job", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
page_1_layout = html.Div([
html.Div([
        html.H1('Certificate Information'),
        html.Br(),
        html.Label([''], style={'font-weight': 'bold', "text-align": "center"}),

        dcc.Dropdown(id='web',
                     options=[{'label': x, 'value': x} for x in df.Website.unique()],
                     multi=False,
                     disabled=False,
                     value='Indeed',),

        dcc.Dropdown(id='job',
                     options=[{'label': x, 'value': x} for x in df.Job_type.unique()],
                     value=['Accounting', 'Project Management'],
                     multi=True,
                     clearable=True,
                     placeholder="Certificate",
                     persistence='string',
                     persistence_type='session'),
        dcc.Dropdown(id='country', options=[], value=[], placeholder="Country"),

    ], className='drop_down'),

    html.Div([
        dcc.Graph(id='our_graph')
    ], className='search'),

    html.Div([
        dcc.Graph(id='x-time-series')
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='map'),
    ], style={'display': 'inline-block', 'width': '49%'}),

])


page_2_layout = html.Div([
html.Div([
        html.H1('Job Summary',id = "key1")
    ]),


    html.Div([dcc.Dropdown(id='job1',
                           options=[{'label': x, 'value': x} for x in df1.job_type.unique()],
                           multi=False,
                           clearable=True,
                           persistence='string',
                           value='accounting',
                           persistence_type='session'),

              dcc.Dropdown(id='country1',
                           options=[{'label': x, 'value': x} for x in df1.Country.unique()],
                           multi=False,
                           clearable=True,
                           value='USA',
                           persistence='string',
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
    ], style={'margin-left':'20%','margin-top': '15%'}),

    html.Div(
        [
            html.P("Search results", id="search_term"),
            html.Div(id="output"),
        ]
    ),

html.Div(
        [
            html.P("Total Job", id="search_term"),
            html.Div(id="output1"),
        ], style={'margin-left':'75%'}
    ),

    html.Div([
        dcc.Graph(id='NLP1')
    ], style={'margin-top': '15%', 'width': '100%','margin-right':'10%'}),

])

page_3_layout = ([
html.Div([
html.Div([
        html.H1('Jobs Information'),
        html.Br(),
        html.Label([''], style={'font-weight': 'bold', "text-align": "center"}),

        dcc.Dropdown(id='country2',
                     options=[{'label': x, 'value': x} for x in df2.Country_name.unique()],
                     multi=False,
                     disabled=False,
                     value='United Kingdom'),

        dcc.Dropdown(id='job2',
                     options=[{'label': x, 'value': x} for x in df2.Job_type.unique()],
                     value=['Child care worker','Community health worker'],
                     multi=True,
                     clearable=True,
                     placeholder="Career",
                     persistence='string',
                     style={'width': "50%"},
                     persistence_type='session'),
    ], className='drop_down1'),

    html.Div([
        dcc.Graph(id='our_graph1')
    ], className='search'),

    html.Div([
        dcc.Graph(id='x-time-series1')
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='map1'),
    ], style={'display': 'inline-block', 'width': '49%'}),

])
])

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_1_layout
    elif pathname == "/page-1":
        return page_2_layout
    elif pathname == "/page-2":
        return page_3_layout
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback(
    dash.dependencies.Output('country', 'options'),
    dash.dependencies.Output('country', 'value'),
    dash.dependencies.Input('web', 'value'),
)
def set_cities_options(chosen_state):
    dff = df[df.Website == chosen_state]
    counties_of_states = [{'label': c, 'value': c} for c in sorted(dff.Country_name.unique())]
    values_selected = [x['value'] for x in counties_of_states]
    return counties_of_states, values_selected



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
                  title="Total Certificates Posting by Day",
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
    fig = px.pie(dff, values='Total', names='Job_type', hole=.5, color_discrete_sequence=["#3A929D", "#FFC328", "#6ABDC8", "#B88300", '#AB63FA', '#FFA15A', '#19D3F3','#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#EF553B', '#00CC96'],title = "Total Selected Certificates Posted")
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
    }, title = "Total Certificates Posted")
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
        skills = ['microsoft','QuickBooks','payable','receivable', '10 key','data entry','bookkeeping','organize','tax','deadline','security','payroll']
    elif (job1 == 'Marriage Family'):
        skills = ['communication','writing','research','teaching','thinking','teamwork']

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
        text = "Search Term:  communication, teamwork, initiative, interpersonal, analytics, flexibility, health, Health"
    elif(job1=='Database'):
        text = 'Search Term:  Problem-solving,Communication,Oracle,SQL,PowerShell,Security,Analytics,database'
    elif(job1=='accounting'):
        text = 'Search Term:  microsoft,QuickBooks , payable , receivable , 10 key , data entry, bookkeeping ,organize ,tax ,deadline ,security ,payroll'
    elif(job1=='Marriage Family'):
        text = "Search Term:  ['communication','writing','research','teaching','thinking','teamwork']"
    return text

@app.callback(
    dash.dependencies.Output('output1', 'children'),
    [dash.dependencies.Input('job1', 'value'),dash.dependencies.Input('country1', 'value')])

def build_graph3(job1,country1):
    dff = df1[(df1['job_type'] == job1)&(df1['Country'] == country1)]
    count = len(dff)
    return count


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
    wordcloud = WordCloud(stopwords=stopwords, background_color="white",width=1200,height=800).generate(text)

    # Display the generated image:
    fig = px.imshow(wordcloud)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
        width=1000, height=800
    )
    return fig

@app.callback(
    dash.dependencies.Output('our_graph1', 'figure'),
    [dash.dependencies.Input('country2', 'value'),
     dash.dependencies.Input('job2', 'value')])
def build_graph(first, second):
    dff = df2[(df2['Country_name'] == first) &(df2['Job_type'].isin(second))]

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
    dash.dependencies.Output('x-time-series1', 'figure'),
    [dash.dependencies.Input('country2', 'value'),
     dash.dependencies.Input('job2', 'value')])
def build_graph1(first1, second2):
    dff = df2[(df2['Country_name'] == first1) &
             (df2['Job_type'].isin(second2))]
    fig = px.pie(dff, values='Total', names='Job_type', hole=.5, color_discrete_sequence=["#3A929D", "#FFC328", "#6ABDC8", "#B88300", '#AB63FA', '#FFA15A', '#19D3F3','#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#EF553B', '#00CC96'],title = "Total Selected Jobs")
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig


@app.callback(
    dash.dependencies.Output('map1', 'figure'),
    dash.dependencies.Input('country2', 'value'))

def build_graph2(first2):
    mask = df2["Country_name"] == first2
    fig = px.bar(df2[mask], x="Total", y="Job_type", barmode="group", height=600, color_discrete_sequence=["#3A929D"],labels={
        "Total": "Total Entry level jobs",
        "Job_type": ""
    },title = "Total Jobs")
    fig.update_layout(plot_bgcolor="white")
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
