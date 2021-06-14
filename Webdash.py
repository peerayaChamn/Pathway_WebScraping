import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly
import plotly.express as px
import collections
from wordcloud import WordCloud, STOPWORDS
import dash_table as dt
from nltk import word_tokenize
import dash_table
from collections import Counter


from nltk.corpus import stopwords

# reading data for summary page
df1 = pd.read_csv("all_info.csv")
df1 = df1.drop_duplicates(subset=['title_web','job_type','salary','location','summary','company','Country','title','website'])
# reading data for job page sort by date posted and drop duplicates
df2 = pd.read_csv("career.csv")
df2['Posted_date'] = pd.to_datetime(df2.Posted_date)
df2.sort_values(by='Posted_date')
df2 = df2.sort_values('Posted_date')
df2 = df2.drop_duplicates(subset=['Job_type', 'Posted_date', 'Country_name'])

df_grouping2 = df2.groupby(['Country_name','Job_type']).mean().round()
df_grouping2 = df_grouping2.reset_index()


df_title = pd.read_csv("title.csv")

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
                dbc.NavLink("Job Analysis", href="/", active="exact"),
                dbc.NavLink("Job Posting", href="/page-1", active="exact"),
                # dbc.NavLink("Job", href="/page-2", active="exact"),
                dbc.NavLink("Title", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "2rem 1rem","background-color":"#f8bc34"},
)


card_content = [
    dbc.CardBody(
        [
            html.H5("Total Job", className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                id = 'output1',className="card-text"
            ),
        ]
    )
]

card_content1 = [
    dbc.CardBody(
        [
            html.H5("Search", className="card-title"),
            dbc.Input(id="input")
            ,
        ]
    )
]

################################################################################
#  Creating page two layout (Summary)
################################################################################

page_2_layout = html.Div([
html.Div([
        html.H1('Job Summary',id = "key1")
    ]),

    dcc.Dropdown(id='job_summary',
                 options=[{'label': x, 'value': x} for x in df1.job_type.unique()],
                 value = 'Community Health'),

    dcc.Dropdown(id='job_title', options=[], value=[], multi=True,
                 clearable=True,
                 placeholder="Career",
                 style={'margin-top': '9%'}),

    dcc.Dropdown(id='country_summary',
                 options=[{'label': x, 'value': x} for x in df1.Country.unique()],
                 multi=False,
                 clearable=True,
                 placeholder="Country",
                 value = 'Canada'),
dcc.Dropdown(
    options=[
        {'label': 'one word', 'value': 'one'},
        {'label': 'two words combination', 'value': 'two'}
    ],
    id='word'
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
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_content1, color="info", inverse=True, style={'position': 'absolute','width': '238px','height': '104px','left': '150px','top': '40px'})),
        ],
        className="mb-4",
    ),
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_content, color="info", inverse=True,style={'position': 'absolute','width': '238px','height': '104px','left': '809px','top': '14px'})),
        ],
        className="mb-4",
    ),

 # displaying wordcloud
    html.Div([
        dcc.Graph(id='NLP_wordcloud')
    ], style={'margin-top': '15%', 'width': '100%','margin-right':'10%'}),

    html.Div(id="table1", style={'position': 'absolute','top': '2100px'}),

    html.Div(id="table2", style={'position': 'absolute','top': '2100px','left':'890px'}),

    html.Div(id="table3", style={'position': 'absolute', 'top': '3000px', 'left': '890px'}),
    html.Div(id="table4", style={'position': 'absolute', 'top': '3000px'}),
])






################################################################################
#  Creating page three layout (career)
################################################################################

page_3_layout = ([
html.Div([
html.Div([
        html.H1('Jobs Information'),
        html.Br(),

    dcc.Dropdown(id='certificate_career',
                 options=[{'label': x, 'value': x} for x in df2.certificate.unique()],
                 value = 'Community Health'),

        dcc.Dropdown(id='job_career', options=[], value=[], multi=True,
                     clearable=True,
                     placeholder="Career",),
        dcc.Dropdown(id='country_career',
                     options=[{'label': x, 'value': x} for x in df2.Country_name.unique()],
                     multi=False,
                     clearable=True,
                     placeholder="Country",
                     value = 'Canada'),
    ], className='drop_down_career'),
    dbc.Button("Learn more about jobs in your area", href='', id='link_page3', color="info"),

    html.Div([
        dcc.Graph(id='time_serie_career')
    ], className='search'),

    html.Div([
        dcc.Graph(id='pie_career')
    ]),

    # html.Div([
    #     dcc.Graph(id='bar_career'),
    # ], style={'display': 'inline-block', 'width': '49%'}),

])
])

################################################################################
#  Creating page three layout (career)
################################################################################

page_4_layout = ([
html.Div([
html.Div([
        html.H1('Jobs Titles'),
        html.Br(),

    dcc.Dropdown(id='certificate_title',
                 options=[{'label': x, 'value': x} for x in df_title.job_type.unique()], ),

        dcc.Dropdown(id='country_title',
                     options=[{'label': x, 'value': x} for x in df_title.Country_name.unique()],
                     multi=False,
                     clearable=True,
                     placeholder="Country",),
    ], className='drop_down_career'),

    dbc.Button("Learn more about jobs in your area", href='', id='link_page3', color="info"),

    html.Div(id="table_title", style={'position': 'absolute','top': '1200px'}),

    html.Div([
        dcc.Graph(id='wordcloud')
    ], style={'margin-top': '15%', 'width': '100%', 'margin-right': '10%'}),

    # html.Div([
    #     dcc.Graph(id='bar_career'),
    # ], style={'display': 'inline-block', 'width': '49%'}),

])
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

    # if pathname == "/":
    #     return page_1_layout
    if pathname == "/":
        return page_2_layout
    elif pathname == "/page-1":
        return page_3_layout
    elif pathname == "/page-3":
         return page_4_layout



################################################################################
#  Creating graph Page 2 drop down
################################################################################

@app.callback(
    dash.dependencies.Output('job_title', 'options'),
    dash.dependencies.Output('job_title', 'value'),
    dash.dependencies.Input('job_summary', 'value'),
)
def set_cities_options(chosen):
    dff = df1[df1.job_type == chosen]
    new = [{'label': y, 'value': y} for y in sorted(dff.title.unique())]
    selected = [x['value'] for x in new]
    return new, selected

################################################################################
#  Creating graph Page 2 (Bar Chart)
################################################################################
# this stop word is filtering out the most common word in english that is not meaningful
stop_words = stopwords.words('english')

@app.callback(
    dash.dependencies.Output('NLP_bar', 'figure'),
    [dash.dependencies.Input('country_summary', 'value'),
     dash.dependencies.Input('job_summary', 'value'),
dash.dependencies.Input('job_title', 'value'),
     dash.dependencies.Input('input', 'value'),
dash.dependencies.Input('word', 'value')
     ])
def build_graph3(country1, job1,job2, value,word):
    dff = df1[(df1['job_type'] == job1)&(df1['Country'] == country1)& (df1['title'].isin(job2))]

    skills = []
    # key word to create graph base on certificate
    if(job1=='Healthcare Administration'):
        skills = ['registers patients','quality contorl','revenue cycle','professional','billing','scheduling','communication','reports','licensed','maintaining','manage','business','ensuring','marketing','critical','excel','microsoft','risk management','advocate' , 'report findings' , 'presentation' , 'powerpoint' , 'logic model' , 'planning model' , 'finance' , 'budget' , 'evaluate' , 'marketing plan' , 'communication skill', 'oral','written' , 'medical terminology' , 'medical coding' , 'leadership' , 'human resources']
    elif(job1=='Database'):
        skills = ['problem-solving','communication','oracle','SQL','powerShell','security','analytic','database']
    elif(job1=='accounting'):
        skills = ['microsoft','QuickBooks','payable','receivable', '10 key','data entry','bookkeeping','organize','tax','deadline','security','Bookkeeper','communicate','Quickbooks','Excel','spreadsheets']
    elif (job1 == 'Marriage Family'):
        skills = ['communicate','writing','research','teaching','thinking','teamwork','counseling']
    elif (job1 == "Occupational safety and health"):
        skills = ['inspect','test','osha basic','environmental compliance', 'evaluate','written reports','active listening','critical thinking','problem solving','reading comprehension','monitoring','systems analysis','quality control','operation monitoring','persuasion','time management', 'reliable','responsible','dependable','persistence' ,'innovation']
    elif(job1== "Community Health"):
        skills = ['communication','business','ensure','assist','support','microsoft','excel' ,'assess' ,'plan' ,'implement' ,'evaluate' ,'communicate' ,'health communication' ,'disease prevention' ,'health promotion' ,'analysis' ,'researc' ,'social media' ,'podcast creation' ,'website creation' ,'poster creation' ,'infographic creation' ,'needs assessment' ,'develop goals' ,'develop objectives' ,'program planning' ,'data collection' ,'data analysis' ,'behavioral change','patient care','behavioral health', 'models', 'theories' ,'health literacy' ,'report findings' ,'presentation' ,'powerpoint' ,'logic model' ,'planning model' ,'finances' ,'budget' ,'research','behavioral','marketing plan' ,'communication channel' ,'persuasive communication' ,'differentiate diseases']
    elif (job1 == 'Web & Computer Programming'):
        skills = ['microsoft','powerbi','tableu','modeling','maintain','technical support','powerhouse team','design','information security','firewall,','encryption','git','cloud','cloud computing','firebase','security','Self-motivation','attention','organized','programming skills','Planning software','Designing and creating applications','Update and expand existing programs','debugging','writing code','github','azure','aws','sql','Abstract thinking','machine learning','communication','agile','react','node js','html','css','javascript','git','php','python','r','java','http','rest api','database','nosql','patience','interpersonal skills','seo','search engine','analytical skills','analysis','testing','resposive design','c++','c','c#','flutter','go','swift','kotlin','scala','statistics','big data','data science','deep learning','story telling','curiosity','data manipulation','data visualization','model deployment','data wrangling','data manipulation',]
    elif(job1 == 'Hospitality & Tourism Management'):
        skills = ['teamwork','responsible','ensuring','travel sales','front desk','management','maintaining','cleaning','professional','customer service', 'cultural awareness', 'communication' , 'multitasking' , 'cutural', 'work ethic', 'language' , 'professionalism' , 'teamwork' , 'problem-solving', 'detail oriented','flexibility', 'commercial awareness','enthusiasm']
    # if the input is inserted then it will append the keyword to the list
    skills.append(value)

    # clean the data and create df and count word

    text = " ".join(i for i in dff.summary)
    words = text.split()
    if(word == 'two'):
        words = [' '.join(words[i: i + 2]) for i in range(0, len(words), 3)]
    split_it = [item.lower() for item in words]
    counter = collections.Counter(split_it)
    most_occur = counter.most_common()
    your_list = [list(i) for i in most_occur]
    sentence = [(word, count) for word, count in your_list if word not in stop_words]

    result_series = pd.Series({k: v for k, v in sentence})
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
    [dash.dependencies.Input('job_title', 'value')])

def build_graph3(job1):
    if(job1 == 'Healthcare Administration'):
        text = 'Search Term: Advocate , Report findings, Presentation, PowerPoint, Logic model, Planning model, Finance, Budget, Evaluate, Marketing plan,  Communication skill, oral and written,Medical terminology, Medical coding, Leadership, Human resources'
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
    [dash.dependencies.Input('job_summary', 'value'),dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),])

def build_graph3(job1,country1,job2):
    dff = df1[(df1['job_type'] == job1)&(df1['Country'] == country1)& (df1['title'].isin(job2))]
    count = len(dff)
    return count


################################################################################
#  Creating graph Page 2 return total
################################################################################
@app.callback(dash.dependencies.Output('table1','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value')
               ])

def update_datatable(country1, job1,job2):

    dff = df1[(df1['Country'] == country1) & (df1['title'].isin(job1) & (df1['job_type'] == job2))]
    dff = dff.groupby(['company'],as_index=False).count()
    dff = dff.drop(['title_web','job_type','salary','location','summary','Country','title','website'], axis=1)
    dff=dff.rename(columns={"date": "Count"})
    dff = dff.sort_values('Count',ascending=False)
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30,style_table={'height': '700px', 'overflowY': 'auto','width': 'auto'} ,export_format='xlsx',export_headers='display')


@app.callback(dash.dependencies.Output('table4','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value')
               ])

def update_datatable(country1, job1,job2):

    dff = df1[(df1['Country'] == country1) & (df1['title'].isin(job1) & (df1['job_type'] == job2))]
    dff = dff.groupby(['location'],as_index=False).count()
    dff = dff.drop(['title_web','job_type','salary','company','summary','Country','title','website'], axis=1)
    dff=dff.rename(columns={"date": "Count"})
    dff = dff.sort_values('Count',ascending=False)
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30,style_table={'height': '700px', 'overflowY': 'auto','width': 'auto'} ,export_format='xlsx',export_headers='display')




@app.callback(dash.dependencies.Output('table2','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value')
               ])

def update_datatable(country1, job1,job2):

    dff = df1[(df1['Country'] == country1) & (df1['title'].isin(job1) & (df1['job_type'] == job2))]
    dff = dff.groupby(['title_web'],as_index=False).count()
    dff = dff.drop(['job_type','salary','location','summary','Country','title','website','company'], axis=1)
    dff=dff.rename(columns={"date": "Count"})
    dff = dff.sort_values('Count',ascending=False)
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30,style_table={'height': '700px', 'overflowY': 'auto','overflowX': 'auto','width': '500px'} ,export_format='xlsx',export_headers='display')


@app.callback(dash.dependencies.Output('table3','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value'),
               dash.dependencies.Input('word', 'value')
               ])

def update_datatable(country1, job1,job2,word):

    dff = df1[(df1['Country'] == country1) & (df1['title'].isin(job1) & (df1['job_type'] == job2))]
    text = " ".join(i for i in dff.summary)
    words = text.split()
    if (word == 'two'):
        words = [' '.join(words[i: i + 2]) for i in range(0, len(words), 3)]
    split_it = [item.lower() for item in words]
    counter = collections.Counter(split_it)
    most_occur = counter.most_common(200)
    your_list = [list(i) for i in most_occur]
    sentence = [(word, count) for word, count in your_list if word not in stop_words ]
    dff = pd.DataFrame(sentence, columns=['word', 'Count'])
    dff = dff.sort_values('Count',ascending=False)
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30,style_table={'height': '700px', 'overflowY': 'auto','overflowX': 'auto','width': '500px'} ,export_format='xlsx',export_headers='display')

################################################################################
#  Creating graph Page 2 (word cloud)
################################################################################
@app.callback(
    dash.dependencies.Output('NLP_wordcloud', 'figure'),
    [dash.dependencies.Input('country_summary', 'value'),
     dash.dependencies.Input('job_title', 'value'),
dash.dependencies.Input('job_summary', 'value')
     ])
def build_graph4(country1, job1,job2):
    dff = df1[(df1['Country'] == country1) & (df1['title'].isin(job1) & (df1['job_type'] == job2))]

    text = " ".join(i for i in dff.summary)

    # Generate a word cloud image stopwrod filter out word that is not nescessary in english
    wordcloud =WordCloud(relative_scaling = 0.3,
                      stopwords=stop_words,
                      min_font_size=1,
                      background_color="white",
                      width=1024,
                      height=768,
                      max_words=300,
                      colormap='plasma',
                      scale=3,
                      font_step=4,
                    #   contour_width=3,
                    #   contour_color='steelblue',
                      collocations=False,
                      margin=2
                      ).generate(text)

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
#  btn
################################################################################
@app.callback(
    dash.dependencies.Output('link_page3', 'href'),
    dash.dependencies.Input('country_career', 'value'))

def set_href2(chosen_country):
    c_name = {'Saudi Arabia': 'sa','Singapore': 'sg',
               'Canada': 'ca','Nigeria': 'ng','Ireland': 'ie',
               'Hong Kong': 'hk','Pakistan': 'pk','Kuwait':'kw','Luxembourg':'lu','South Africa': 'za',
              'United Kingdom':'uk','New Zealand' : 'nz','Malaysia' :'malaysia','India':'in','Philippines':'ph',
              'Australia' :'au','Indonesia': 'id','Argentina':'ar','Austria' :'ar','Germany': 'de',
              'Belgium':'be','Brazil':'br','Portugal':'pt','Chile' : 'cl','USA':'www'
               }
    id = c_name.get(chosen_country)
    link = "https://" +id+".indeed.com/"

    print(link)
    return link

################################################################################
#  Creating graph Page 3 (pie chart)
################################################################################

@app.callback(
    dash.dependencies.Output('pie_career', 'figure'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('job_career', 'value')])
def build_graph1(first1, second2):
    dff = df_grouping2[(df_grouping2['Country_name'] == first1) &
             (df_grouping2['Job_type'].isin(second2))]
    fig = px.bar(dff, x='Total', y='Job_type',title = "Total Selected Jobs")
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

# @app.callback(
#     dash.dependencies.Output('country_title', 'options')
#     dash.dependencies.Input('job_til', 'value'),
# )
# def set_cities_options(chosen):
#     dff = df_title[df_title.job_type == chosen]
#     new = [{'label': y, 'value': y} for y in sorted(dff.Country_name.unique())]
#     selected = [x['value'] for x in new]
#     return new, selected


################################################################################
#  Creating graph Page 2 (word cloud)
################################################################################
@app.callback(
    dash.dependencies.Output('wordcloud', 'figure'),
    [dash.dependencies.Input('country_title', 'value'),
     dash.dependencies.Input('certificate_title', 'value')
     ])
def build_graph4(country1, job1):
    dff = df_title[(df_title['Country_name'] == country1) & (df_title['job_type'] == job1)]

    title = dff['title'].to_list()
    word_could_dict = Counter(title)
    wordcloud = WordCloud(relative_scaling=0.3,
                          min_font_size=1,
                          background_color="white",
                          width=1024,
                          height=768,
                          max_words=2000,
                          colormap='plasma',
                          scale=3,
                          font_step=4,
                          #   contour_width=3,
                          #   contour_color='steelblue',
                          collocations=False,
                          margin=2).generate_from_frequencies(word_could_dict)
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

@app.callback(dash.dependencies.Output('table_title','children'),
              [dash.dependencies.Input('country_title', 'value'),
               dash.dependencies.Input('certificate_title', 'value')
               ])

def update_datatable(country1, job1):

    dff = df_title[(df_title['Country_name'] == country1) & (df_title['job_type'] == job1)]
    dff = dff.groupby(['title'],as_index=False).count()
    dff = dff.drop(['job_type'], axis=1)
    dff = dff.rename(columns={"Title": "Count"})
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30 ,export_format='xlsx',export_headers='display')


if __name__ == '__main__':
    app.run_server(debug=False)
