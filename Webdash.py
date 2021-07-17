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
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords

# reading data for summary page
df1 = pd.read_csv("all_info.csv")
df1 = df1.drop_duplicates(subset=['title_web','job_type','salary','location','summary','company','Country','title','website'])
# reading data for job page sort by date posted and drop duplicates
df2 = pd.read_csv("df.csv")
df2['Posted_date'] = pd.to_datetime(df2.Posted_date)
df2.sort_values(by='Posted_date')
df2 = df2.sort_values('Posted_date')
df2 = df2.drop_duplicates(subset=['certificate', 'Posted_date', 'Country_name'])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
app.title = 'Career Exploration'
server = app.server

################################################################################
# Creating sidebar
################################################################################

sidebar = html.Div(
    [
       html.H2("BYU-Pathway", id="head"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Job Analysis", href="/", active="exact"),
                dbc.NavLink("Job Posting", href="/page-1", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "2rem 1rem","background-color":"#f8bc34"},
)


card_total_page1 = [
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

card_search= [
    dbc.CardBody(
        [
            html.H5("Skill Search", className="card-title"),
            dbc.Input(id="input", placeholder='Analytics')
            ,
        ]
    )
]

card_salary = [
    dbc.CardBody(
        [
            html.H5("Salary (Year)"),
            html.Div(id="table5")
            ,
        ]
    )
]

card_max = [
    dbc.CardBody(
        [
            html.H5("Max Job Posted"),
            html.Div(id="maxi")
            ,
        ]
    )
]

card_avg = [
    dbc.CardBody(
        [
            html.H5("Average Job Posted"),
            html.Div(id="average")
            ,
        ]
    )
]

card_total = [
    dbc.CardBody(
        [
            html.H5("Total Job Posted"),
            html.Div(id="total")
            ,
        ]
    )
]

################################################################################
#  Creating page two layout (Summary)
################################################################################

page_2_layout = html.Div([

html.Div([
    dcc.Dropdown(id='country_summary',
                 options=[{'label': x, 'value': x} for x in df1.Country.unique()],
                 multi=False,
                 clearable=True,
                 className='drop',
                 placeholder="Country",
                 value = 'Canada'),

    dcc.Dropdown(id='job_summary',
                 value='Web & Computer Programming',
                 clearable=True,
                 multi=False,
                 className='drop',
                 placeholder="Certificate"),

    dcc.Dropdown(id='location',
                 clearable=True,
                 multi=False,
                 className='drop',
                 placeholder="Region"),

    dcc.Dropdown(id='job_level',
                 clearable=True,
                 multi=True,
                 className='drop',
                 placeholder="Job type"),
],id='drop_id'),

html.Div([
    dcc.Dropdown(id='job_title', options=[], value=[], multi=True,
                 clearable=True,
                 placeholder="Career",)],id='title_drop'),

    # displaying bar graph
html.Div([
    html.H2('Common Skills'),
    html.Div([
        dcc.Graph(id='NLP_bar')
    ]),
dcc.Dropdown(
    options=[
        {'label': 'one word', 'value': 'one'},
        {'label': 'two words combination', 'value': 'two'}
    ],
    id='word',
),
], id = 'graph_bar'),


html.Div([
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_salary, color="info", inverse=True,)),
        ],
        className="mb-4", id ='card_salary'
    ),
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_search, color="info", inverse=True, )),
        ],
        className="mb-4", id='card_search'
    ),
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_total_page1, color="info", inverse=True,)),
        ],
        className="mb-4",id ='card_total_page1'
    )],id = 'card'),
 # displaying wordcloud


    html.Div([
        html.H2('Word cloud for Common Words'),
        dcc.Graph(id='NLP_wordcloud')
    ], id ="cloud"),


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
    dcc.Dropdown(id='certificate_career',
                 options=[{'label': x, 'value': x} for x in df2.certificate.unique()],
                 value = ['Community Health', 'Public Health'],
                 multi = True),

        dcc.Dropdown(id='country_career',
                     options=[{'label': x, 'value': x} for x in df2.Country_name.unique()],
                     multi=False,
                     clearable=True,
                     placeholder="Country",
                     value = 'Canada'),
    ], className='drop_down_career'),
    dbc.Button("Learn more about jobs in your area", href='', id='link_page3', color="info"),
html.Div([
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_total, color="info", inverse=True,)),
        ],
        className="mb-4", id ='card_total'
    ),
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_avg, color="info", inverse=True, )),
        ],
        className="mb-4", id='card_avg'
    ),
    dbc.Row(
        [
            dbc.Col(dbc.Card(card_max, color="info", inverse=True,)),
        ],
        className="mb-4",id ='card_max'
    )],id = 'card'),

    html.Div([
        html.H2('Total Job Posting by Day'),
        dcc.Graph(id='time_serie_career')
    ], className='search'),

    html.Div([
        html.H2('Average Job Posting by Certificate'),
        dcc.Graph(id='bar_career')
    ], className='search'),

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


################################################################################
#  Creating graph Page 3 drop down
################################################################################

@app.callback(
    dash.dependencies.Output('job_summary', 'options'),
    dash.dependencies.Input('country_summary', 'value'),
)

def set_cities_options(chosen):
    dff = df1[df1.Country == chosen]
    new = [{'label': y, 'value': y} for y in sorted(dff.job_type.unique())]
    return new


################################################################################
#  Creating graph Page 2 drop down
################################################################################

@app.callback(
    dash.dependencies.Output('job_title', 'options'),
dash.dependencies.Output('job_title', 'value'),
    dash.dependencies.Input('job_summary', 'value'),
dash.dependencies.Input('country_summary', 'value'),
dash.dependencies.Input('location', 'value'),
dash.dependencies.Input('job_level', 'value'),
)
def set_cities_options(job_summ,country1,location,level):
    if(location is None):
        dff = df1[(df1['job_type'] == job_summ) & (df1['Country'] == country1)& (df1['job_level'].isin(level))]
    else:
        dff = df1[(df1['job_type'] == job_summ) & (df1['Country'] == country1) & (
                    df1['location'] == location)& (df1['job_level'].isin(level))]

    new = [{'label': y, 'value': y} for y in sorted(dff.title.unique())]
    selected = [x['value'] for x in new]
    return new, selected


@app.callback(
    dash.dependencies.Output('job_level', 'options'),
dash.dependencies.Output('job_level', 'value'),
    dash.dependencies.Input('job_summary', 'value'),
dash.dependencies.Input('country_summary', 'value'),
dash.dependencies.Input('location', 'value'),
)
def set_cities_options(job_summ,country1,location):
    if(location is None):
        dff = df1[(df1['job_type'] == job_summ) & (df1['Country'] == country1)]
    else:
        dff = df1[(df1['job_type'] == job_summ) & (df1['Country'] == country1) & (
                    df1['location'] == location)]

    new = [{'label': y, 'value': y} for y in sorted(dff.job_level.unique())]
    selected = [x['value'] for x in new]
    return new, selected


@app.callback(
dash.dependencies.Output('location', 'options'),
    [dash.dependencies.Input('country_summary', 'value'),
     dash.dependencies.Input('job_summary', 'value'),
dash.dependencies.Input('job_title', 'value'),
dash.dependencies.Input('job_level', 'value'),
     ])
def set_cities_options(country, job_sum, job_title, level):
    dff = df1[(df1['job_type'] == job_sum) & (df1['Country'] == country) & (df1['title'].isin(job_title)) & (df1['job_level'].isin(level))]
    new = [{'label': y, 'value': y} for y in sorted(dff.location.unique())]
    return new

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
dash.dependencies.Input('word', 'value'),
dash.dependencies.Input('location', 'value'),
dash.dependencies.Input('job_level', 'value'),
     ])
def build_graph3(country1, job1,job2, value,word,location,level):

    if(location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) &(df1['job_level'].isin(level)) ]
    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                    df1['location'] == location) &(df1['job_level'].isin(level)) ]


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
        skills = ['microsoft','powerbi','tableu','modeling','maintain','technical support','powerhouse team','design','information security','firewall,','encryption','cloud','cloud computing','firebase','security','Self-motivation','attention','organized','programming skills','Planning software','Designing and creating applications','Update and expand existing programs','debugging','writing code','github','azure','aws','sql','Abstract thinking','machine learning','communication','agile','react','node js','html','css','javascript','git','php','python','r','java','http','rest api','database','nosql','patience','interpersonal skills','seo','search engine','analytical skills','analysis','testing','resposive design','c++','c','c#','flutter','go','swift','kotlin','scala','statistics','big data','data science','deep learning','story telling','curiosity','data manipulation','data visualization','model deployment','data wrangling','data manipulation',]
    elif(job1 == 'Hospitality & Tourism Management'):
        skills = ['teamwork','responsible','ensuring','travel sales','front desk','management','maintaining','cleaning','professional','customer service', 'cultural awareness', 'communication' , 'multitasking' , 'cutural', 'work ethic', 'language' , 'professionalism' , 'teamwork' , 'problem-solving', 'detail oriented','flexibility', 'commercial awareness','enthusiasm']
    elif(job1 == 'Administrative Assistant'):
        skills = ['written', 'verbal','communication', 'time management', 'attention' , 'problem solving', 'technology', 'responsible', 'management' , 'billing', 'bookkeeping' , 'managing','efficient','computerized accounting','accounting software','monitor']
    elif(job1 == 'Medical Billing & Coding Fundamentals'):
        skills = ['billing','insurance','computer','accounting','bookkeeping','problem solving','teamwork','maintaining','answering','research','medical coding','insurance industry']
    elif(job1 == 'Project Management'):
        skills = ['management','estimate','scheduling','risk management','cost management','develop','lead','schedules','documentation','meetings','competetive','budget','planning','oversees','communication','project plans','progress reports']
    elif(job1 == 'TEFL'):
        skills = ['flexible','teaching','curriculum','online','amazingtalker','passionate','enthusiastic','motivated','certified','young students','excellent teaching']
    elif(job1 == 'Technical Support Engineer'):
        skills = ['technical issues','computer science','strong analatical','address customers','software','hardware','networking','helpdesk','jumpstarting']
    elif(job1 == 'Advanced Family History Research Certificate'):
        skills = ['data analysis','research','consultation','relationship','augmented']
    elif(job1 == 'Commercial Fundamental'):
        skills = ['marketing','sales','managing','professional','oversee','troubleshooting','feedbacks','digital marketing','search engine','lifecycle marketing']
    elif(job1 == 'Auto Service Technology' ):
        skills = ['driving experience','physical check','driving','truck','repair','ensure','servicing']
    elif(job1 == 'Construction Field Supervision'):
        skills = ['management','engineering','maintenance','reliability','sustainability','engineer','planning','facilities management','reliability engineer','oversee cost']
    elif(job1 == "Entrepreneurship" ):
        skills = ['sale','reserch','support','buisness','marketing','communication','customer aquisition','maintain discipline']
    elif(job1 == "Agribusiness Management"):
        skills = ['business','sales','marketing','management','finance']
    elif(job1 == "Computer Support"):
        skills = ['database','security','sales','complaints','oracle','network']

    # if the input is inserted then it will append the keyword to the list
    if(value != None):
        value = value.lower()
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
#  Creating graph Page 2 return total
################################################################################

@app.callback(
    dash.dependencies.Output('output1', 'children'),
    [dash.dependencies.Input('job_summary', 'value'),dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),dash.dependencies.Input('location', 'value'),
     dash.dependencies.Input('job_level', 'value')])

def build_graph3(job1,country1,job2,location,level):
    if (location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2))& (df1['job_level'].isin(level))]
    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                df1['location'] == location)& (df1['job_level'].isin(level))]

    count = len(dff)
    return count

################################################################################
#  Creating graph Page 2 return total
################################################################################

def convert(value):
    if "hour" in str(value):
        return "hour"
    elif "year" in str(value):
        return "year"
    elif "month" in str(value):
        return "month"
    else:
        return "not sure"

def split_it(year):
    return re.findall('\d*\.?\d+',year)

@app.callback(
    dash.dependencies.Output('table5', 'children'),
    [dash.dependencies.Input('job_summary', 'value'),dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),dash.dependencies.Input('location', 'value'),
     dash.dependencies.Input('job_level', 'value')])

def build_graph3(job1,country1,job2,location,level):

    if (location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
            df1['job_level'].isin(level))]
    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                df1['location'] == location) & (df1['job_level'].isin(level))]

    dff = dff.dropna(how='all')
    dff = dff.drop(['title_web', 'job_type', 'location', 'summary', 'Country', 'title', 'website', 'company', 'date'], axis=1)

    dff['type_of_salary'] = dff['salary'].apply(convert)
    dff['salary'] = dff['salary'].str.replace(',', '')
    dff['salary'] = dff['salary'].str.replace(' ', '')
    dff['salary_numeric'] = pd.DataFrame(dff['salary'].apply(split_it))
    dff['salary_numeric'] = dff['salary_numeric'].str[-1]
    dff['salary_numeric'] = dff['salary_numeric'].astype(float)
    dff = dff.groupby(['type_of_salary'], as_index=False).mean().round()

    year = 0
    hour = 0
    month = 0

    dff['year'] = dff['type_of_salary'] == 'year'
    dff['month'] = dff['type_of_salary'] == 'month'
    dff['hour'] = dff['type_of_salary'] == 'hour'
    df_month = dff[dff['type_of_salary'] == 'month']
    df_hour = dff[dff['type_of_salary'] == 'hour']
    df_year = dff[dff['type_of_salary'] == 'year']

    if(not df_month[df_month['month'] == True].empty):
        month = int(df_month['salary_numeric'] * 12)
    if(not df_hour[df_hour['hour'] == True].empty):
        hour = int(df_hour['salary_numeric'] * 1920)
    if(not df_year[df_year['year'] == True].empty):
        year = int(df_year['salary_numeric'])

    value = round((year+hour+month) /3)
    if (value == 0):
        value = "Not enough information"

    return value

################################################################################
#  Creating graph Page 2 return total
################################################################################
@app.callback(dash.dependencies.Output('table1','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value'),
               dash.dependencies.Input('location', 'value'),
               dash.dependencies.Input('job_level', 'value'),

               ])

def update_datatable(country1, job2,job1,location,level):

    if (location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) &(df1['job_level'].isin(level)) ]
    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                df1['location'] == location)&(df1['job_level'].isin(level)) ]


    dff = dff.groupby(['company'],as_index=False).count()
    dff = dff.drop(['title_web','job_type','salary','location','summary','Country','title','website','job_level'], axis=1)
    dff=dff.rename(columns={"date": "Count"})
    dff = dff.rename(columns={"company": "Company Name"})
    dff = dff.sort_values('Count',ascending=False)
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30,style_table={'height': '700px', 'overflowY': 'auto','width': '500px'} ,export_format='xlsx',export_headers='display')


@app.callback(dash.dependencies.Output('table4','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value'),
               dash.dependencies.Input('location', 'value'),
               dash.dependencies.Input('job_level', 'value'),
               ])

def update_datatable(country1, job2,job1,location,level):
    if (location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2))& (df1['job_level'].isin(level))]

    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                df1['location'] == location)& (df1['job_level'].isin(level))]


    dff = dff.groupby(['location'],as_index=False).count()
    dff = dff.drop(['title_web','job_type','salary','company','summary','Country','title','website','job_level'], axis=1)
    dff=dff.rename(columns={"date": "Count"})
    dff = dff.sort_values('Count',ascending=False)
    dff = dff.rename(columns={"location": "Location"})
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30,style_table={'height': '700px', 'overflowY': 'auto','width': '500px'} ,export_format='xlsx',export_headers='display')




@app.callback(dash.dependencies.Output('table2','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value'),
               dash.dependencies.Input('location', 'value'),
               dash.dependencies.Input('job_level', 'value')
               ])

def update_datatable(country1, job2,job1,location,level):
    if (location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2))& (df1['job_level'].isin(level))]
    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                df1['location'] == location)& (df1['job_level'].isin(level))]


    dff = dff.groupby(['title_web'],as_index=False).count()
    dff = dff.drop(['job_type','salary','location','summary','Country','title','website','company','job_level'], axis=1)
    dff=dff.rename(columns={"date": "Count"})
    dff = dff.rename(columns={"title_web": "Posted Title"})
    dff = dff.sort_values('Count',ascending=False)
    data_1 = dff.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (dff.columns)]
    return dt.DataTable(data=data_1, columns=columns,page_size=30,style_table={'height': '700px', 'overflowY': 'auto','overflowX': 'auto','width': '500px'} ,export_format='xlsx',export_headers='display')


@app.callback(dash.dependencies.Output('table3','children'),
              [dash.dependencies.Input('country_summary', 'value'),
               dash.dependencies.Input('job_title', 'value'),
               dash.dependencies.Input('job_summary', 'value'),
               dash.dependencies.Input('word', 'value'),
               dash.dependencies.Input('location', 'value'),
               dash.dependencies.Input('job_level', 'value')
               ])

def update_datatable(country1, job2,job1,word,location,level):
    if (location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2))& (df1['job_level'].isin(level))]
    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                df1['location'] == location)& (df1['job_level'].isin(level))]


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
    dff = dff.rename(columns={"word": "Most Common Word"})
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
dash.dependencies.Input('job_summary', 'value'),
dash.dependencies.Input('location', 'value'),
dash.dependencies.Input('job_level', 'value')

     ])
def build_graph4(country1, job2,job1,location,level):
    if (location is None):
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2))& (df1['job_level'].isin(level))]
    else:
        dff = df1[(df1['job_type'] == job1) & (df1['Country'] == country1) & (df1['title'].isin(job2)) & (
                df1['location'] == location)& (df1['job_level'].isin(level))]

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

@app.callback(
    dash.dependencies.Output('time_serie_career', 'figure'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('certificate_career', 'value')])

def build_graph(first, second):
    dff = df2[(df2['Country_name'] == first) & (df2['certificate'].isin(second))]
    fig = px.line(dff, x="Posted_date", y="Total", color="certificate", labels={
        "Posted_date": "Date",
        "Total": "Total Entry level jobs",
        "Job_type": "Certificates"
    },
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
              'Belgium':'be','Brazil':'br','Portugal':'pt','Chile' : 'cl','United States':'www', 'Czech Republic':'cz','Vietnam':'vn',
              'Sweden' :'se','Korea':'kr','Romania':'ro','Nigeria':'ng','Japan':'jp','Hungary':'hu','Greece':'gr','Finland':'fi','Russia':'ru',
              'Thailand':'th','Italy':'it'
               }
    id = c_name.get(chosen_country)
    link = "https://" +id+".indeed.com/"
    return link

###############################################################################
# Creating graph Page 3 (pie chart)
###############################################################################

@app.callback(
    dash.dependencies.Output('bar_career', 'figure'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('certificate_career', 'value')])

def build_graph1(first1, second2):

    dff = df2[(df2['Country_name'] == first1) &
             (df2['certificate'].isin(second2))]

    df_grouping2 = dff.groupby(['Country_name','certificate']).mean().round()
    df = df_grouping2.reset_index()

    fig = px.bar(df, x='certificate', y='Total', text='Total', labels= {'certificate':'Chosen Certificate', 'Total' : 'Job Posted'},
                 color_discrete_sequence=["#FFC328"])
    fig.update_layout(
        font_family="Arial",
        title_font_family="Arial",
        title_font_color="#999999",
        plot_bgcolor="white"
    )

    return fig


@app.callback(
    dash.dependencies.Output('total', 'children'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('certificate_career', 'value')])

def minimum(job1,country1):
    dff = df2[(df2['Country_name'] == job1) &
              (df2['certificate'].isin(country1))]

    count = sum(dff['Total'])
    return count


@app.callback(
    dash.dependencies.Output('maxi', 'children'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('certificate_career', 'value')])

def maximum(job1, country1):
    dff = df2[(df2['Country_name'] == job1) &
              (df2['certificate'].isin(country1))]

    count = max(dff['Total'])
    return count


@app.callback(
    dash.dependencies.Output('average', 'children'),
    [dash.dependencies.Input('country_career', 'value'),
     dash.dependencies.Input('certificate_career', 'value')])

def average(job1, country1):
    dff = df2[(df2['Country_name'] == job1) &
              (df2['certificate'].isin(country1))]

    count = dff['Total'].mean().round()
    return count


if __name__=='__main__':
    app.run_server(debug=False,port=8000)