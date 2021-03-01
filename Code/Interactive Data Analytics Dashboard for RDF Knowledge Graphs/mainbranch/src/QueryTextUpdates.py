
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_table
import dash
import pandas as pd
from MainApp import app
from HelperFunctions import queries,endpoints


@app.callback([Output("query-endpoint", "value"),Output("endpoint-query1-type", "title")],
              [Input('endpoint-query1-intermediate-value1', 'title'),Input('endpoint-query1-intermediate-value2', 'title'), Input('upload-data', 'fileNames')],             
              )
def update_endpoint(title1,title2, fileNames):
    ctx = dash.callback_context
    if not ctx.triggered:
        endpoint = 'No changes yet'
    else:
        endpoint = ctx.triggered[0]['prop_id'].split('.')[0]
    if((endpoint == 'endpoint-query1-intermediate-value1' or endpoint == 'upload-data') and (ctx.triggered[0]['value'] != '' and ctx.triggered[0]['value'] != None)):

        return title1,'1'
    else:
        return title2,'2' 
    
#---------------------------------------------------Update endpoint query 2 value Callback for compare--------------------------------------------------------------
@app.callback([Output("query-endpoint2", "value"),Output("endpoint-query2-type", "title")],
              [Input('endpoint-query2-intermediate-value1', 'title'),Input('endpoint-query2-intermediate-value2', 'title'), Input('upload-data2', 'fileNames')],             
              )
def update_endpoint2(title1,title2, fileNames):
    ctx = dash.callback_context
    if not ctx.triggered:
        endpoint = 'No changes yet'
    else:
        endpoint = ctx.triggered[0]['prop_id'].split('.')[0]
    if((endpoint == 'endpoint-query2-intermediate-value1' or endpoint == 'upload-data2') and (ctx.triggered[0]['value'] != '' and ctx.triggered[0]['value'] != None)):
        return title1,'1'
    else:
        return title2,'2'  
    
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