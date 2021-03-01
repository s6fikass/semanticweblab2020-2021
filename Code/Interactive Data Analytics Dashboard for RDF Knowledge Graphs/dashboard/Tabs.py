import import_ipynb
import dash

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from MainApp import app
from HelperFunctions import df
from HelperFunctions import start_table_df



# selected tab callback

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
                        style_table={'height': '327px', 'overflowY': 'auto'},
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
                        export_format="csv",
                        style_data_conditional=[
                            {
                                'if': {
                                    'state': 'active'  # 'active' | 'selected'
                                },
                                'backgroundColor': 'rgb(211, 224, 234)',
                                'border': '1px solid rgb(0, 116, 217)'
                            }
                        ],
                    )
                ],
                style= {
                    'display': 'none',
                    'marginTop': '5px'
                }
            ),
            html.Div([
                 dbc.Alert(
                     ["There are no data for generating a table."],
                     id="alert-table",
                     is_open=False,
                     fade=True,
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
                            'textAlign': 'center', 
                            },
                        

                 ),    
            ]),
            html.Div(
                id='MainTableDiv2',
                children=[
                    dash_table.DataTable(
                        id='Resulttable2',
                        style_table={'height': '327px', 'overflowY': 'auto'},
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
                        export_format="csv",
                        style_data_conditional=[
                            {
                                'if': {
                                    'state': 'active'  # 'active' | 'selected'
                                },
                                'backgroundColor': 'rgb(211, 224, 234)',
                                'border': '1px solid rgb(0, 116, 217)'
                            }
                        ],
                    )
                ],
                style= {
                    'display': 'none',
                    'marginTop': '5px'
                }
            ),
            
            html.Div([
                 dbc.Alert(
                     ["There are no data for generating a table."],
                     id="alert-table2",
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
                         
                    
                 ),    
                    ]), 
        ]),
    elif tab == 'Charts':
        return html.Div([
                 html.Div(
                   id="result-graph",
                   className="result-graph",
                   children=[
                        dcc.Graph(
                            id='graph',
                            figure={
                               #"layout": {
                                   # "height": 400,
                                #    'overflow': 'scroll',
                                #},
                            },
                        )],#dcc graph
                    style= {
                      'display': 'none',
                      },  
               ), 
                html.Div([
                    dbc.Alert(
                         ["There are not enough data for generating the charts."],
                         id="alert-chart",
                         is_open=False,
                         fade=True,
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
                     ),    
                 ]),
                html.Div(
                       id="result-graph2",
                       className="result-graph",
                       children=[
                            dcc.Graph(
                                id='graph2',
                                figure={
                               # "layout": {
                               #     "height": 400,
                               # },
                                },
                            ),
                         ],
                        style={'display': 'none'}
                ),  
                html.Div([
                     dbc.Alert(
                         ["There are not enough data for generating the charts."],
                         id="alert-chart2",
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
                            },)   
                ])                
            ])
    elif tab == 'Maps':
        return html.Div([
               html.Div(
                   id="result-map",
                   className="result-map",
                   children=[
                        dcc.Graph(
                            id='map',
                            figure={
                               # "layout": {
                               #    "height": 340,
                               # },
                            },
                        ),                
                     ],
                 style= {'display': 'none'}  
               ),
               html.Div([
                    dbc.Alert(
                        ["\t"
                         "No coordinates have been found to display on the map.\n "
                         "You can choose latitude and longitude by dropdown menus."],
                        id="alert-map",
                        is_open=False,
                        fade=True,
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
                                'color': 'orange',
                                'textAlign': 'center',
                                },
                         
                   
                    ),
               ]),
                
               html.Div([
                    dbc.Alert(
                    ["There are not enough data for generating a map. "],
                    id="alert-map3",
                    is_open=False,
                    fade=True,
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
                            'textAlign': 'center',
                            },
                         
                    
                        )
               ]),            
               html.Div(
                   id="result-map2",
                   className="result-map",
                   children=[
                        dcc.Graph(
                            id='map2',
                            figure={
                              #  "layout": {
                              #     "height": 340,
                              #  },
                            },
                        ),

                     ],
                 style= {'display': 'none'}  
               ),
               html.Div([
                    dbc.Alert(
                    ["No coordinates have been found to display on the map. \n" 
                     "You can choose latitude and longitude by dropdown menus."],
                    id="alert-map2",
                    is_open=False,
                    fade=True,
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
                            'color': 'orange',
                            'textAlign': 'center',
                            },
                         
                    
                        )
               ]),
            html.Div([
                    dbc.Alert(
                    ["There are not enough data for generating a map. "],
                    id="alert-map4",
                    is_open=False,
                    fade=True,
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
               ])
            
        ])
    elif tab == 'Samples':
        return html.Div([
            html.Div(
                className="examples",
                children=[
                    dash_table.DataTable(
                        id='querySamples',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_data_conditional=[
                            {
                                'if': {
                                    'state': 'active'  # 'active' | 'selected'
                                },
                                'backgroundColor': 'rgb(211, 224, 234)',
                                'border': '1px solid rgb(0, 116, 217)'
                            }
                        ],
                        data=df.to_dict('records'),
                        style_table={'width': '800px'},
                        style_cell=dict(textAlign='left'), 
                        column_selectable='single'
                    )
                ]),
        ])
    elif tab == 'Tutorial':
        return html.Div([
            html.Div(
                className="tutorial",
                children=[
                    html.H3('Tutorial'),

                    html.A("How to query an endpoint", href='#endpoint-link'),
                    html.Br(),
                    html.A("How to query an RDF file", href='#upload-link'),
                    html.Br(),
                    html.A("How to see the query result as a table", href='#table-link'),
                    html.Br(),
                    html.A("How to see the query result as a chart", href='#chart-link'),
                    html.Br(),
                    html.A("How to use query samples", href='#samples-link'),
                    html.Br(),
                    html.A("How to see the query result as a map", href='#map-link'),
                    html.Br(),
                    html.A("How to compare two queries", href='#compare-link'),
                    html.Br(),
                    html.A("How to use query samples", href='#samples-link'),

                    html.H4("- How to query an endpoint:", id='endpoint-link'),
                    html.Img(src=app.get_asset_url('01.png'), style={'width': '700px', 'vertical-align': 'top'}),

                    html.H4("- How to query an RDF file:", id='upload-link'),
                    html.Img(src=app.get_asset_url('02.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('03.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('04.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('05.png'), style={'width': '700px', 'vertical-align': 'top'}),

                    html.H4("- How to see the query result as a table:", id='table-link'),
                    html.Img(src=app.get_asset_url('06.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('07.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('08.png'), style={'width': '700px', 'vertical-align': 'top'}),

                    html.H4("- How to see the query result as a chart:", id='chart-link'),
                    html.Img(src=app.get_asset_url('06.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('09.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('010.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('011.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('012.png'), style={'width': '700px', 'vertical-align': 'top'}),

                    html.H4("- How to see the query result as a map:", id='map-link'),
                    html.Img(src=app.get_asset_url('06.png'), style={'width': '700px', 'vertical-align': 'top'}),
                    html.Img(src=app.get_asset_url('013.png'), style={'width': '700px', 'vertical-align': 'top'}),

                    html.H4("- How to compare two queries:", id='compare-link'),
                    html.Img(src=app.get_asset_url('015.png'), style={'width': '700px', 'vertical-align': 'top'}),

                    html.H4("- How to use query samples:", id='samples-link'),
                    html.Img(src=app.get_asset_url('014.png'), style={'width': '700px', 'vertical-align': 'top'}),
                ]),
        ])
