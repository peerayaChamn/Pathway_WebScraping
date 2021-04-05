import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

from nltk import word_tokenize

from nltk.corpus import stopwords

# reading data for certification page sort by date posted and drop duplicates
df = pd.read_csv("scraping_data.csv")
df['Posted_date'] = pd.to_datetime(df.Posted_date)
df = df.sort_values('Posted_date')
df = df.drop_duplicates(subset=['Job_type', 'Posted_date', 'Country_name'])

# reading data for summary page
df1 = pd.read_csv("all_info.csv")

# reading data for job page sort by date posted and drop duplicates
df2 = pd.read_csv("career.csv")
df2['Posted_date'] = pd.to_datetime(df2.Posted_date)
df2.sort_values(by='Posted_date')
df2 = df2.sort_values('Posted_date')
df2 = df2.drop_duplicates(subset=['Job_type', 'Posted_date', 'Country_name'])

# reading data for location page sort by date posted
df3 = pd.read_csv("scraping_data2.csv")
df3['Posted_date'] = pd.to_datetime(df3.Posted_date)
df3.sort_values(by='Posted_date')
df3 = df3.sort_values('Posted_date')

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
server = app.server

################################################################################
# Creating sidebar
################################################################################

sidebar = html.Div(
    [
       html.H2("BYU-Pathway"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Certificate", href="/", active="exact"),
                dbc.NavLink("Skill", href="/page-1", active="exact"),
                dbc.NavLink("Job", href="/page-2", active="exact"),
                dbc.NavLink("Location", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "2rem 1rem","background-color":"#f8bc34"},
)
################################################################################
#  Creating page one layout (Certification)
################################################################################

page_1_layout = html.Div([
html.Div([
        html.H1('Certificate Information'),
        html.Br(),
        # drop down for website
        dcc.Dropdown(id='website_certificate',
                     options=[{'label': x, 'value': x} for x in df.Website.unique()],
                     multi=False,
                     value='Indeed',),

        dcc.Dropdown(id='job_certificate',
                     options=[{'label': x, 'value': x} for x in df.Job_type.unique()],
                     value=['Accounting', 'Project Management'],
                     multi=True,
                     clearable=True,
                     placeholder="Certificate"),

        # country will depends on website chosen
        dcc.Dropdown(id='country_certificate', options=[], value=[], placeholder="Country"),

    ], className='drop_down'),

    # displaying the time-series graph
    html.Div([
        dcc.Graph(id='serie_graph_certificate')
    ], className='search'),

    # Pie chart from selected value
    html.Div([
        dcc.Graph(id='pie_certificate')
    ], style={'display': 'inline-block', 'width': '49%'}),

    # Displaying bar chart for all job type
    html.Div([
        dcc.Graph(id='bar_certificate'),
    ], style={'display': 'inline-block', 'width': '49%'}),

])

################################################################################
#  Creating page two layout (Summary)
################################################################################

page_2_layout = html.Div([
html.Div([
        html.H1('Job Summary',id = "key1")
    ]),

    html.Div([dcc.Dropdown(id='job_summary',
                           options=[{'label': x, 'value': x} for x in df1.job_type.unique()],
                           multi=False,
                           clearable=True,
                           value='accounting',),

              dcc.Dropdown(id='country_summary',
                           options=[{'label': x, 'value': x} for x in df1.Country.unique()],
                           multi=False,
                           clearable=True,
                           value='USA',)
              ], className='NLP'),

# input for search in the bar chart
    html.Div(
        [
            dcc.Input(id="input1", type="text", placeholder=""),
            html.P("Search",id = "key")
        ]
    ),

# displaying bar graph
    html.Div([
        dcc.Graph(id='NLP_bar')
    ], style={'margin-left':'20%','margin-top': '15%'}),

# displaying text used for search terms
    html.Div(
        [
            html.P("Search results", id="search_term"),
            html.Div(id="output"),
        ]
    ),

# displaying total jobs
    html.Div(
        [
            html.P("Total Job", id="Total_job"),
            html.Div(id="output1"),
        ], style={'margin-left':'75%'}
    ),

 # displaying wordcloud
    html.Div([
        dcc.Graph(id='NLP_wordcloud')
    ], style={'margin-top': '15%', 'width': '100%','margin-right':'10%'}),

])

################################################################################
#  Creating page three layout (career)
################################################################################

page_3_layout = ([
html.Div([
html.Div([
        html.H1('Jobs Information'),
        html.Br(),

        dcc.Dropdown(id='job_career', options=[], value=[], multi=True,
                     clearable=True,
                     placeholder="Career",),
        dcc.Dropdown(id='certificate_career',
                 options=[{'label': x, 'value': x} for x in df2.certificate.unique()],),
        dcc.Dropdown(id='country_career',
                     options=[{'label': x, 'value': x} for x in df2.Country_name.unique()],
                     multi=False,
                     clearable=True,
                     placeholder="Country",),
    ], className='drop_down_career'),

    html.Div([
        dcc.Graph(id='time_serie_career')
    ], className='search'),

    html.Div([
        dcc.Graph(id='pie_career')
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='bar_career'),
    ], style={'display': 'inline-block', 'width': '49%'}),

])
])

################################################################################
#  Creating page four layout (location)
################################################################################

page_4_layout = html.Div([
html.Div([
        html.H1('Certificate Information'),
        html.Br(),

        dcc.Dropdown(id='country_location',
                     options=[{'label': x, 'value': x} for x in df3.Country_name.unique()],
                     multi=False,
                     disabled=False,),

        dcc.Dropdown(id='job_location',
                     options=[{'label': x, 'value': x} for x in df3.Job_type.unique()],
                     value=['Accounting', 'Project Management'],
                     multi=True,
                     clearable=True,
                     placeholder="Certificate"),
        dcc.Dropdown(id='city_location',options=[], value=[],
                     multi=False,
                     disabled=False,),

    ], className='drop_down3'),

    html.Div([
        dcc.Graph(id='location_graph')
    ], className='search'),

    html.Div([
        dcc.Graph(id='pie_location')
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='bar_location'),
    ], style={'display': 'inline-block', 'width': '49%'}),

])

################################################################################
#  Content layout
################################################################################

content = html.Div(id="page-content", style={
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
})

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

################################################################################
#  Content layout and link to different page
################################################################################

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):

    if pathname == "/":
        return page_1_layout
    elif pathname == "/page-1":
        return page_2_layout
    elif pathname == "/page-2":
        return page_3_layout
    elif pathname == "/page-3":
        return page_4_layout

################################################################################
#  filter page 1 drop down to correspond to the website
################################################################################

@app.callback(
    dash.dependencies.Output('country_certificate', 'options'),
    dash.dependencies.Output('country_certificate', 'value'),
    dash.dependencies.Input('website_certificate', 'value'),
)
def set_cities_options(chosen_country):
    # create a new df based on the chosen country
    dff = df[df.Website == chosen_country]
    new = [{'label': y, 'value': y} for y in sorted(dff.Country_name.unique())]
    select_value = [x['value'] for x in new]
    return new, select_value

################################################################################
#  Creating graph Page 1 (time-series-graph)
################################################################################
@app.callback(
    dash.dependencies.Output('serie_graph_certificate', 'figure'),
    [dash.dependencies.Input('country_certificate', 'value'),
     dash.dependencies.Input('job_certificate', 'value'),
     dash.dependencies.Input('website_certificate', 'value')])

def build_graph(first, second,third):
    # create a df based on the drop down in the page 1
    dff = df[(df['Website'] == third) & (df['Country_name'] == first) &(df['Job_type'].isin(second))]

    # making graph
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

################################################################################
#  Creating graph Page 1 (Pie Chart)
################################################################################

@app.callback(
    dash.dependencies.Output('pie_certificate', 'figure'),
    [dash.dependencies.Input('country_certificate', 'value'),
     dash.dependencies.Input('job_certificate', 'value')])

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

################################################################################
#  Creating graph Page 1 (Pie Chart)
################################################################################

@app.callback(
    dash.dependencies.Output('bar_certificate', 'figure'),
    dash.dependencies.Input('country_certificate', 'value'))

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


################################################################################
#  Creating graph Page 2 (Bar Chart)
################################################################################
# this stop word is filtering out the most common word in english that is not meaningful
stop_words = stopwords.words('english')

# clean data
def clean_data(word_list):
    word_list = word_tokenize(word_list)
    word_list = [word for word in word_list if word.isalpha()]
    word_list = [word for word in word_list if word not in stop_words]
    return word_list

@app.callback(
    dash.dependencies.Output('NLP_bar', 'figure'),
    [dash.dependencies.Input('country_summary', 'value'),
     dash.dependencies.Input('job_summary', 'value'),
     dash.dependencies.Input('input1', 'value')
     ])
def build_graph3(country1, job1, value):
    dff = df1[(df1['Country'] == country1) &
              (df1['job_type'] == job1)]

    skills = []
    # key word to create graph base on certificate
    if(job1=='Public Health'):
        skills = ['communication','teamwork','initiative','interpersonal','analytics','flexibility','health', 'Health']
    elif(job1=='Database'):
        skills = ['problem-solving','communication','oracle','SQL','powerShell','security','analytic','database']
    elif(job1=='accounting'):
        skills = ['microsoft','QuickBooks','payable','receivable', '10 key','data entry','bookkeeping','organize','tax','deadline','security','Bookkeeper','communicate','Quickbooks','Excel','spreadsheets']
    elif (job1 == 'Marriage Family'):
        skills = ['communicate','writing','research','teaching','thinking','teamwork','counseling']
    # if the input is inserted then it will append the keyword to the list
    skills.append(value)

    # clean the data and create df and count word
    new_df = dff["summary"].apply(clean_data)
    result = new_df.apply(Counter).sum().items()

    result_series = pd.Series({k: v for k, v in result})
    filter_result = result_series.filter(items=skills)
    new_df = pd.DataFrame(filter_result)
    new_df = new_df.reset_index()
    new_df.columns = ['Words', 'Counts']

    # create the graph
    fig = px.bar(new_df, x="Words", y="Counts", barmode="group",text ="Counts",color_discrete_sequence=["#FFC328"])
    fig.update_traces(textposition='outside')
    fig.update_layout(plot_bgcolor="white")
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig

################################################################################
#  Creating graph Page 2 return search result base on chosen certificate
################################################################################

@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('job_summary', 'value')])

def build_graph3(job1):
    if(job1 == 'Public Health'):
        text = "Search Term:  communication, teamwork, initiative, interpersonal, analytics, flexibility, health, Health"
    elif(job1=='Database'):
        text = 'Search Term:  problem-solving,communication,oracle,SQL,powerShell,security,analytics,database'
    elif(job1=='accounting'):
        text = 'Search Term:  microsoft, payable , receivable , 10 key , data entry, bookkeeping ,organize ,tax ,deadline ,security ,communicate,spreadsheets'
    elif(job1=='Marriage Family'):
        text = "Search Term:  ['communication','writing','research','teaching','thinking','teamwork','counseling']"
    return text

################################################################################
#  Creating graph Page 2 return total
################################################################################

@app.callback(
    dash.dependencies.Output('output1', 'children'),
    [dash.dependencies.Input('job_summary', 'value'),dash.dependencies.Input('country_summary', 'value')])

def build_graph3(job1,country1):
    dff = df1[(df1['job_type'] == job1)&(df1['Country'] == country1)]
    count = len(dff)
    return count

################################################################################
#  Creating graph Page 2 (word cloud)
################################################################################
@app.callback(
    dash.dependencies.Output('NLP_wordcloud', 'figure'),
    [dash.dependencies.Input('country_summary', 'value'),
     dash.dependencies.Input('job_summary', 'value')])
def build_graph4(country1, job1):
    dff = df1[(df1['Country'] == country1) &
              (df1['job_type'] == job1)]

    text = " ".join(i for i in dff.summary)

    # Generate a word cloud image stopwrod filter out word that is not nescessary in english
    wordcloud = WordCloud(stopwords=stop_words, background_color="white",width=1200,height=800).generate(text)

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

################################################################################
#  Creating graph Page 3 drop down
################################################################################

@app.callback(
    dash.dependencies.Output('job_career', 'options'),
    dash.dependencies.Output('job_career', 'value'),
    dash.dependencies.Input('certificate_career', 'value'),
)
def set_cities_options(chosen):
    dff = df2[df2.certificate == chosen]
    new = [{'label': y, 'value': y} for y in sorted(dff.Job_type.unique())]
    selected = [x['value'] for x in new]
    return new, selected

################################################################################
#  Creating graph Page 3 (time-series)
################################################################################

@app.callback(
    dash.dependencies.Output('time_serie_career', 'figure'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('job_career', 'value'),
     dash.dependencies.Input('certificate_career', 'value')])

def build_graph(first, second,third):
    dff = df2[(df2['Country_name'] == first) &(df2['Job_type'].isin(second) & (df2['certificate'] == third))]
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

################################################################################
#  Creating graph Page 3 (pie chart)
################################################################################

@app.callback(
    dash.dependencies.Output('pie_career', 'figure'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('job_career', 'value')])
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

################################################################################
#  Creating graph Page 3 (bar chart)
################################################################################

@app.callback(
    dash.dependencies.Output('bar_career', 'figure'),
    dash.dependencies.Input('country_career', 'value'))

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

################################################################################
#  Creating graph Page 4 (time-series)
################################################################################

@app.callback(
    dash.dependencies.Output('location_graph', 'figure'),
    [dash.dependencies.Input('country_location', 'value'),
     dash.dependencies.Input('job_location', 'value'),
     dash.dependencies.Input('city_location', 'value')])

def build_graph(first, second,third):
    dff = df3[(df3['Country_name'] == first) &(df3['Job_type'].isin(second) & (df3['city'] == third))]
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

################################################################################
#  Creating graph Page 4 (pie_chart)
################################################################################

@app.callback(
    dash.dependencies.Output('pie_location', 'figure'),
    [dash.dependencies.Input('country_location', 'value'),
     dash.dependencies.Input('job_location', 'value')])
def build_graph1(first1, second2):
    dff = df3[(df3['Country_name'] == first1) &
             (df3['Job_type'].isin(second2))]
    fig = px.pie(dff, values='Total', names='Job_type', hole=.5, color_discrete_sequence=["#3A929D", "#FFC328", "#6ABDC8", "#B88300", '#AB63FA', '#FFA15A', '#19D3F3','#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#EF553B', '#00CC96'],title = "Total Selected Jobs")
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
    )
    return fig

################################################################################
#  Creating graph Page 4 (bar chart)
################################################################################

@app.callback(
    dash.dependencies.Output('bar_location', 'figure'),
    dash.dependencies.Input('country_location', 'value'))

def build_graph2(first2):
    mask = df3["Country_name"] == first2
    fig = px.bar(df3[mask], x="Total", y="Job_type", barmode="group", height=600, color_discrete_sequence=["#3A929D"],labels={
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

################################################################################
#  Creating graph Page 4 filter drop down
################################################################################

@app.callback(
    dash.dependencies.Output('city_location', 'options'),
    dash.dependencies.Output('city_location', 'value'),
    dash.dependencies.Input('country_location', 'value'),
)
def set_cities_options(chosen_value):
    dff = df3[df3.Country_name == chosen_value]
    new = [{'label': y, 'value': y} for y in sorted(dff.city.unique())]
    selected = [x['value'] for x in new]
    return new, selected


if __name__ == '__main__':
    app.run_server(debug=False)
