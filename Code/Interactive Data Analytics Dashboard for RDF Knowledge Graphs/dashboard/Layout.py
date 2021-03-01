import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_uploader as du
from dash.dependencies import Input, Output, State
from MainApp import app


du.configure_upload(app, 'data/',use_upload_id=False)

tab_style = {
    'backgroundColor': '#d3e0ea',
    'color': 'rgb(70, 79, 85)'
}

tab_selected_style = {
    'backgroundColor': 'rgb(70, 162, 187)',
    'color': 'white',
}

app.layout = html.Div([
    
   html.Div(
        className="navbar",
        children=[ 
            html.H3(
                children="Interactive Data Analytics Dashboard for RDF Knowledge Graphs", 
                className="navbar--title"),
            html.Div(
                children=[
                dcc.Tabs(id='tabs', value='Tutorial', children=[
                    dcc.Tab(label='Tables', value='Tables',style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Charts', value='Charts',style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Maps', value='Maps',style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Query Samples', value='Samples',style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Tutorial', value='Tutorial',style=tab_style, selected_style=tab_selected_style),
                ]),
                html.Div(id='tabs-content', className="tabs-content")
            ])
    ]),
    html.Div(
        className="querybox",
        children=[
            html.Span(
                className="main-box",
                children=[
                    dcc.Textarea(
                        id= "query-endpoint", 
                        value="",
                        placeholder="Enter your SPARQL Endpoint.",
                        className="querybox--endpoint",
                    ),
                    du.Upload(
                        id='upload-data',
                        text='Drag and Drop Here to upload!',
                        filetypes=['nt', 'ttl','rdf','n3','xml','tql'],
                        default_style={
                            'width': '95%',
                            'height': '103px',
                            'lineHeight': '13px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'margin': 'auto',
                            'fontSize': 'small',
                            'color': 'gray'
                        },
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
                    html.A(
                        children='toggle compare box',
                        id="toggle-compare",
                        className="toggle-compare",
                        n_clicks=0,
                    ),
                ]
            ),

            html.Span(
                id="compare-box",
                className="compare-box",
                children=[
                    dcc.Textarea(
                        id= "query-endpoint2", 
                        value="",
                        placeholder="Enter your SPARQL Endpoint.",
                        className="querybox--endpoint"
                    ),
                    du.Upload(
                        id='upload-data2',
                        text='Drag and Drop Here to upload!',
                        filetypes=['nt', 'ttl','rdf','n3','xml','tql'],
                        default_style={
                            'width': '95%',
                            'height': '103px',
                            'lineHeight': '13px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'margin': 'auto',
                            'fontSize': 'small',
                            'color': 'gray'
                        },
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
                    )
                ]
            ),
            
            #Alert div: Getting query
            html.Div([
                dbc.Alert(
                    ["There is no data for submitting. Check the endpoint and query textboxes."],
                    id="alert-value",
                    is_open=False,
                    dismissable=True,
                    style={
                            'position': 'absolute',
                            'top': '15%', 
                            'right': '20%', 
                            'width': '40%',
                            'height': '5%',
                            'lineHeight': '20px',
                            'borderWidth': '1px',
                            'borderStyle': 'groove',
                            'margin': 'auto',
                            'fontSize': 'small',
                            'color': 'red',
                            'textAlign': 'center',
                      },
                         
                    

                )
            ]),
            
            html.Div([
                dbc.Alert(
                    ["There is no data for comparing. Check the endpoint and query textboxes."],
                    id="alert-value2",
                    is_open=False,
                    dismissable=True,
                    style={
                            'position': 'absolute',
                            'top': '20%', 
                            'right': '20%', 
                            'width': '40%',
                            'height': '5%',
                            'lineHeight': '20px',
                            'borderWidth': '1px',
                            'borderStyle': 'groove',
                            'margin': 'auto',
                            'fontSize': 'small',
                            'color': 'red',
                            'textAlign': 'center'
                      },
                   
                 )
            ]),  
            
            html.Div([
                dbc.Alert(
                    ["There is a problem with the query or endpoint's URL."],
                    id="alert-except",
                    is_open=False,
                    dismissable=True,
                    style={
                            'position': 'absolute',
                            'top': '15%', 
                            'right': '20%', 
                            'width': '40%',
                            'height': '5%',
                            'lineHeight': '20px',
                            'borderWidth': '1px',
                            'borderStyle': 'groove',
                            'margin': 'auto',
                            'fontSize': 'small',
                            'color': 'red',
                            'textAlign': 'center'
                      },
                   
                 )
            ]), 
            
            html.Div([
                dbc.Alert(
                    ["There is a problem with the query or endpoint's URL."],
                    id="alert-except2",
                    is_open=False,
                    dismissable=True,
                    style={
                            'position': 'absolute',
                            'top': '20%', 
                            'right': '20%', 
                            'width': '40%',
                            'height': '5%',
                            'lineHeight': '20px',
                            'borderWidth': '1px',
                            'borderStyle': 'groove',
                            'margin': 'auto',
                            'fontSize': 'small',
                            'color': 'red',
                            'textAlign': 'center'
                      },
                   
                 )
            ]), 

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
                style={
                    'marginTop': '500px',
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
                style={
                    'marginTop': '500px',  
                    'background': 'transparent',
                }
               ),
                        
])#layout div