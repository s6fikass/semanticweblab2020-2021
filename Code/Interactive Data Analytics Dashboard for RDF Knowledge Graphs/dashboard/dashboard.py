import plotly.express as px
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
import dash_table
import dash
import pandas as pd
import json
import csv
import base64
import datetime
import io
import sys
import os
import rdflib
from rdflib import Namespace
from rdflib.plugins.sparql import prepareQuery
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD


# Build App
app = dash.Dash(__name__,suppress_callback_exceptions=True)

server = app.server

#function to query dbpedia endpoint
def sql(value, endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(value)
    sparql.setReturnFormat(JSON)
    processed_results = sparql.query().convert()
    cols = processed_results['head']['vars']

    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)

    return pd.DataFrame(out, columns=cols)


def parse_contents(contents, filename, date):
    
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

#layout

app.layout = html.Div([
    
   html.Div(
        className="navbar",
        children=[ 
            html.H3(
                children="Interactive Data Analytics Dashboard for RDF Knowledge Graphs", 
                className="navbar--title"),
            html.Div(
                children=[
                dcc.Tabs(id='tabs', value='Dashboard', children=[
                    dcc.Tab(label='Dashboard', value='Dashboard'),
                    dcc.Tab(label='Tables', value='Tables'),
                    dcc.Tab(label='Charts', value='Charts'),
                    dcc.Tab(label='Query Samples', value='Samples'),
                    dcc.Tab(label='Tutorial', value='Tutorial'),
                ]),
                html.Div(id='tabs-content', className="tabs-content")
            ])
    ]),
    html.Div(
        className="querybox",
        children=[
            html.H4(
                className="querybox--title",
                children="SPARQL Query"
            ),
            dcc.Textarea(
                id= "query-endpoint", 
                value="",
                placeholder="Enter your SPARQL query endpoint.",
                className="querybox--endpoint"
            ),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '95',
                    'height': '13px',
                    'lineHeight': '13px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'margin': 'auto',
                    'font-size': 'small'
                },
                multiple=True
            ),
            dcc.Textarea(
                id= "query-text", 
                value="",
                placeholder="Enter your SPARQL query.",
                className="querybox--textarea",
                n_clicks=0
            ),
            html.Button(
                id="submit-btn", className="querybox--btn", 
                children="SUBMIT", 
                n_clicks=0,
                style={'display': 'block','margin': 'auto'}
            ),
            dcc.Textarea(
                id= "query-endpoint2", 
                value="",
                placeholder="Enter your SPARQL query endpoint.",
                className="querybox--endpoint"
            ),
            dcc.Upload(
                id='upload-data2',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '95',
                    'height': '13px',
                    'lineHeight': '13px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'margin': 'auto',
                    'font-size': 'small'
                },
                multiple=True
            ),
            dcc.Textarea(
                id= "query-text2", 
                value="",
                placeholder="Enter your SPARQL query.",
                className="querybox--textarea",
                n_clicks=0
            ),
            html.Button(
                id="compare-btn", className="querybox--btn", 
                children="COMPARE", 
                n_clicks=0,
                style={'display': 'block','margin': 'auto'}
            ),
    ]),      
   

    
    #hidden div 
    dcc.Loading(id="loading", 
                children=[html.Div( 
                    id='intermediate-value', 
                    children=[],
                    style={'display': 'none'}),
                    html.Div( 
                    id='endpoint-query1-intermediate-value1', 
                    title='',
                    children=[],
                    style={'display': 'none'})
                    ,
                    html.Div( 
                    id='endpoint-query1-intermediate-value2', 
                    title='',
                    children=[],
                    style={'display': 'none'})
                    ,
                    html.Div( 
                    id='endpoint-query1-type', 
                    title='',
                    children=[],
                    style={'display': 'none'})],
                type="circle",
                #fullscreen=True,
                style={
                    'margin-top': '500px',
                    'background': 'transparent',
        

                    
                }
                
            ),

     #hidden div 
    dcc.Loading(id="loading2", 
                children=[html.Div( 
                    id='intermediate-value2', 
                    children=[],
                    style={'display': 'none'}),
                    html.Div( 
                    id='endpoint-query2-intermediate-value1', 
                    title='',
                    children=[],
                    style={'display': 'none'})
                    ,
                    html.Div( 
                    id='endpoint-query2-intermediate-value2', 
                    title='',
                    children=[],
                    style={'display': 'none'})
                    ,
                    html.Div( 
                    id='endpoint-query2-type', 
                    title='',
                    children=[],
                    style={'display': 'none'})],
                type="circle",
                #fullscreen=True,
                style={
                    'margin-top': '500px',  
                    'background': 'transparent',
                }
               ),
    
                
    
#    html.Div( 
#        id='intermediate-value2', 
#        children=[],
#        style={'display': 'none'}
#    )
            
])#layout div

#-----------------------------------------read query samples file----------------------------------------------------------------
#make query examples table from csv file
with open('data/Book3.csv', 'rt') as f:
    csv_reader = csv.reader(f)
    headers = []
    queries = []
    endpoints = []
    for line in csv_reader:
        headers.append(line[0])
        queries.append(line[1])
        endpoints.append(line[2])

    data = {'Query Examples':  headers}
    df = pd.DataFrame (data, columns = ['Query Examples'])
    ResultListdataframe = pd.DataFrame (data=[], columns = [])


#-----------------------------------------initial empty table ----------------------------------------------------------------
#initial empty table
start_table_df = pd.DataFrame(columns=[''])

#-----------------------------------------selected tab callback ----------------------------------------------------------------
#render page for selected tab
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
)
def render_content(tab):
    if tab == 'Dashboard':
        return html.Div([
            html.Div(id='output-data-upload'),
            html.Div(id='output-data-upload2'),
            html.Div([
                html.H3('Statistics'),
                html.Div(id='records-size',children=[]),
                html.Div(id='records-size2',children=[])
            ])
        ])
    elif tab == 'Tables':
        return html.Div([
            html.Div(
                id='MainTableDiv',
                children=[
                    dash_table.DataTable(
                        id='Resulttable',
                        data=start_table_df.to_dict('records'), 
                        columns = [{'id': c, 'name': c} for c in start_table_df.columns],
                        style_cell=dict(textAlign='left'),   
                        editable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable="single",
                        row_selectable="multi",
                        row_deletable=True,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        export_format="csv"
                    )
                ],
                style= {'display': 'none'}
            ),
            html.Div(
                id='MainTableDiv2',
                children=[
                    dash_table.DataTable(
                        id='Resulttable2',
                        data=start_table_df.to_dict('records'), 
                        columns = [{'id': c, 'name': c} for c in start_table_df.columns],
                        style_cell=dict(textAlign='left'),   
                        editable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable="single",
                        row_selectable="multi",
                        row_deletable=True,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        export_format="csv"
                    )
                ],
                style= {'display': 'none'}
            )
        ])
    elif tab == 'Charts':
        return html.Div([
               html.Div(
                   id="result-graph",
                   className="result-graph",
                   children=[
                        dcc.Graph(
                         id='graph',
                        ),
                
                     ],
                 style= {'display': 'none'}  
               ),
               html.Div(
                   id="result-graph2",
                   className="result-graph",
                   children=[
                        dcc.Graph(
                         id='graph2',
                        ),
                
                     ],
                 style= {'display': 'none'}  
               )
  
        ])
    elif tab == 'Samples':
        return html.Div([
            html.Div(
                className="examples",
                children=[
                    dash_table.DataTable(
                        id='querySamples',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        style_table={'width': '800px'},
                        style_cell=dict(textAlign='left'), 
                        column_selectable='single'
                    )
                ]),
             
        ])
    elif tab == 'Tutorial':
        return html.Div([
            html.H3('Tutorial'),
            html.Div(
                className="tutorial",
                children=[
                ]),
        ])
    
#---------------------------------------------------Update endpoint query 1 value Callback--------------------------------------------------------------
@app.callback([Output("query-endpoint", "value"),Output("endpoint-query1-type", "title")],
              [Input('endpoint-query1-intermediate-value1', 'title'),Input('endpoint-query1-intermediate-value2', 'title'), Input('upload-data', 'last_modified')],             
              )
def update_endpoint(title1,title2, list_of_dates):
    ctx = dash.callback_context
    if not ctx.triggered:
        endpoint = 'No changes yet'
    else:
        endpoint = ctx.triggered[0]['prop_id'].split('.')[0]
    if(endpoint == 'endpoint-query1-intermediate-value1' or endpoint == 'upload-data'):
        return title1,'1'
    else:
        return title2,'2'  
    
#---------------------------------------------------Update endpoint query 2 value Callback for compare--------------------------------------------------------------
@app.callback([Output("query-endpoint2", "value"),Output("endpoint-query2-type", "title")],
              [Input('endpoint-query2-intermediate-value1', 'title'),Input('endpoint-query2-intermediate-value2', 'title'), Input('upload-data', 'last_modified')],             
              )
def update_endpoint(title1,title2, list_of_dates):
    ctx = dash.callback_context
    if not ctx.triggered:
        endpoint = 'No changes yet'
    else:
        endpoint = ctx.triggered[0]['prop_id'].split('.')[0]
    if(endpoint == 'endpoint-query2-intermediate-value1' or endpoint == 'upload2-data'):
        return title1,'1'
    else:
        return title2,'2'  
    
    
#---------------------------------------------------Upload File Callback for query1--------------------------------------------------------------
UPLOAD_DIRECTORY = "../data/"
@app.callback(Output('endpoint-query1-intermediate-value1', 'title'),
              [Input('upload-data', 'contents'),
              Input('upload-data', 'filename'),
              Input('upload-data', 'last_modified'),
             
              ])
def update_output(list_of_contents, list_of_names, list_of_dates):
   
        if list_of_contents is not None:
            data = list_of_contents[0].encode("utf8").split(b";base64,")[1]
            with open(os.path.join(UPLOAD_DIRECTORY, list_of_names[0]), "wb") as fp:
                fp.write(base64.decodebytes(data))
                return list_of_names[0]
        return ''
#---------------------------------------------------Upload File Compare Callback for query2--------------------------------------------------------------
@app.callback(Output('endpoint-query2-intermediate-value1', 'title'),
              [Input('upload-data2', 'contents'),
              Input('upload-data2', 'filename'),
              Input('upload-data2', 'last_modified'),
             
              ])
def update_output(list_of_contents, list_of_names, list_of_dates):
   
        if list_of_contents is not None:
            data = list_of_contents[0].encode("utf8").split(b";base64,")[1]
            with open(os.path.join(UPLOAD_DIRECTORY, list_of_names[0]), "wb") as fp:
                fp.write(base64.decodebytes(data))
                return list_of_names[0]
        return ''

#---------------------------------------------------Query Samples Callback--------------------------------------------------------------
# write selected query sample in the query text area    
@app.callback(
    [Output("query-text", "value"), Output('endpoint-query1-intermediate-value2', 'title')],
    [Input("querySamples", "active_cell")],
    [State("query-text", "value"),State("query-endpoint", "value")]
)
def get_active_cell(active_cell,prev_query,prev_endpoint):
    if(active_cell):
        
        return queries[next(iter(active_cell.values()))], endpoints[next(iter(active_cell.values()))] 
    return prev_query,prev_endpoint

#---------------------------------------------------Get-data Function--------------------------------------------------------------
def get_data_function(value, endpoint,endpointType):
        if endpointType == '1':
            g = rdflib.Graph()
            g.parse("../data/karl.n3", format=rdflib.util.guess_format("../data/karl.n3"))
    #            qres = g.query(
    #                """select * where {<http://de.dbpedia.org/resource/Karlsruhe> ?p ?o .}""")
            qres = g.query(value)
            ResultListdataframe = pd.DataFrame (qres.bindings)
     #       for row in qres:
     #           print("%s knows %s" % row)
            return ResultListdataframe.to_json(date_format='iso', orient='split')
        else:
            ResultListdataframe = sql(value, endpoint)    
            return ResultListdataframe.to_json(date_format='iso', orient='split')

#---------------------------------------------------Get-data Callback--------------------------------------------------------------
@app.callback(Output("intermediate-value", 'children'),
               Input("submit-btn", "n_clicks"),
              [State("query-text","value"), State("query-endpoint","value"),State("endpoint-query1-type", "title")])
def get_data(n_clicks, value, endpoint,endpointType):
     if n_clicks > 0:
        return get_data_function(value, endpoint,endpointType)

#---------------------------------------------------Get-data Compare Callback--------------------------------------------------------------
@app.callback(Output("intermediate-value2", 'children'),
               Input("compare-btn", "n_clicks"),
              [State("query-text2","value"), State("query-endpoint2","value"),State("endpoint-query2-type", "title")])
def get_data(n_clicks, value, endpoint,endpointType):
     if n_clicks > 0:
        return get_data_function(value, endpoint,endpointType)

#---------------------------------------------------Table Function--------------------------------------------------------------
def gen_table_function(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows):
    if n_clicks > 0:
        selectedRowsIndex=derived_virtual_selected_rows
        resultListdataframe = pd.read_json(jsonified_ResultListdataframe, orient='split')
        mycolumns = [{'name': index, 'id': index,"deletable": True, "selectable": True} for index in resultListdataframe.columns]

        return resultListdataframe.to_dict('records'),mycolumns,{'display': 'block'}
    return start_table_df.to_dict('records'), [{'id': '', 'name': ''}],{'display': 'none'}    
#---------------------------------------------------Table Callback--------------------------------------------------------------
#generate table from query    
@app.callback(
    [Output("Resulttable", "data"), Output('Resulttable', 'columns'), Output('MainTableDiv', 'style')],
    [Input("submit-btn", "n_clicks"),
    Input("intermediate-value", "children"),
    Input('Resulttable', "derived_virtual_selected_rows")],
)    

def gen_table(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows):
    return gen_table_function(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows)

#---------------------------------------------------Table Compare Callback--------------------------------------------------------------
#generate table from query2    
@app.callback(
    [Output("Resulttable2", "data"), Output('Resulttable2', 'columns'), Output('MainTableDiv2', 'style')],
    [Input("compare-btn", "n_clicks"),
    Input("intermediate-value2", "children"),
    Input('Resulttable2', "derived_virtual_selected_rows")],
)    

def gen_table(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows):
    return gen_table_function(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows)

#---------------------------------------------------Statistics Function---------------------------------------------------------------
#function to listen to submit button and take textarea content
def statistics_function(n_clicks, value,endpoint):
    if n_clicks > 0:
        return nsql(value,endpoint)

#Query the endpoint
def get_results(value,endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(value)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

#Get the number of records
def nsql(value,endpoint):
    results = get_results(value,endpoint)
    txt = "Number of records found: "
    for result in results["results"]["bindings"]:
        length = len(results["results"]["bindings"])
    return txt, length

#---------------------------------------------------Statistics Callback---------------------------------------------------------------
@app.callback(
    Output("records-size", "children"),
    [Input("submit-btn", "n_clicks")],
    [State("query-text","value"), State("query-endpoint","value")]
)
def statistics(n_clicks, value,endpoint):
    return statistics_function(n_clicks, value,endpoint)

#---------------------------------------------------Statistics Compare Callback---------------------------------------------------------------
@app.callback(
    Output("records-size2", "children"),
    [Input("compare-btn", "n_clicks")],
    [State("query-text2","value"), State("query-endpoint2","value")]
)
def statistics(n_clicks, value,endpoint):
    return statistics_function(n_clicks, value,endpoint)

#---------------------------------------------------Charts Function--------------------------------------------------------------

def gen_graph_function(n_clicks, jsonified_ResultListdataframe):
    
    if n_clicks > 0:
       
        ResultListdataframe = pd.read_json(jsonified_ResultListdataframe, orient='split')

        if 'coord' in ResultListdataframe.columns:
            print('coordinate presents')
            #lon, lat = list()
            #lon_lat = string2latlon(ResultListdataframe['coord'])
            ResultListdataframe['coord'] = ResultListdataframe['coord'].str.split("\(", expand=True)[1]
            ResultListdataframe['coord'] = ResultListdataframe['coord'].str.split("\)", expand=True)
            ResultListdataframe['lon'] = ResultListdataframe['coord'].str.split(" ", expand=True)[0].astype(float)
            ResultListdataframe['lat'] = ResultListdataframe['coord'].str.split(" ", expand=True)[1].astype(float)
            print("lon", ResultListdataframe['lon'])
            print("lat", ResultListdataframe['lat'])
    
        # initialize figure
        fig = go.Figure()

        # add trace
        fig.add_trace(go.Bar(x=ResultListdataframe.iloc[:, 0],
                                 y=ResultListdataframe.iloc[:, 1],
                                 name='bar'))

        fig.add_trace(go.Scatter(x=ResultListdataframe.iloc[:, 0],
                                 y=ResultListdataframe.iloc[:, 1],
                                 mode='markers',
                                 name='marker',
                                 visible=False))

        fig.add_trace(go.Scatter(x=ResultListdataframe.iloc[:, 0],
                                  y=ResultListdataframe.iloc[:, 1],
                                  mode='lines',
                                  name='line',
                                  visible=False))
        
        
        buttonlist = []
        for col in ResultListdataframe.columns:
              buttonlist.append(
                dict(
                args=['y',[ResultListdataframe[str(col)]] ],
                label=str(col),
                method='restyle'
            )
          )
                
    
        buttonlist_x = []
        for col in ResultListdataframe.columns:
              buttonlist_x.append(
                dict(
                args=['x',[ResultListdataframe[str(col)]] ],
                label=str(col),
                method='restyle'
            )
          )

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=list([

                    dict(label="bar",
                        method="update",
                        args=[{"visible": [True, False, False]},
                        {"title": "Bar Graph",}]),
                    dict(label="Scatter",
                        method="update",
                        args=[{"visible": [False, True, False]},
                        {"title": "Scatter Graph",}]),
                    dict(label="line",
                        method="update",
                        args=[{"visible": [False, False, True]},
                        {"title": "Line Graph",}]), 
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=-0.9,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                    
                ),
                dict(buttons=buttonlist,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=-0.9,
                xanchor="left",
                y=0.90,
                yanchor="top"

                    ),
                                
                dict(buttons=buttonlist_x,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=-0.9,
                xanchor="left",
                y=0.60,
                yanchor="top"

                    )
            ]
        )
        
        fig.update_layout(
        annotations=[
        dict(text="AXIS-Y", x=-0.9, xref="paper", y=0.94, yref="paper",align="left", showarrow=False),
        dict(text="AXIS-X", x=-0.9, xref="paper", y=0.60,yref="paper", showarrow=False),
       
        ])
        



        return fig,{'display': 'block'}

    return { },{'display': 'block'}
#---------------------------------------------------Charts Callback--------------------------------------------------------------
@app.callback(
    [Output("graph", "figure"), Output('result-graph', 'style')],
    [Input("submit-btn", "n_clicks"),
    Input("intermediate-value", "children"),
    ],
)
def gen_graph(n_clicks, jsonified_ResultListdataframe):
    return gen_graph_function(n_clicks, jsonified_ResultListdataframe)
    
#---------------------------------------------------Charts Compare Callback--------------------------------------------------------------
@app.callback(
    [Output("graph2", "figure"), Output('result-graph2', 'style')],
    [Input("compare-btn", "n_clicks"),
    Input("intermediate-value2", "children"),
    ],
)
def gen_graph(n_clicks, jsonified_ResultListdataframe):
    return gen_graph_function(n_clicks, jsonified_ResultListdataframe)


if __name__ == '__main__':
    app.run_server(debug=True)