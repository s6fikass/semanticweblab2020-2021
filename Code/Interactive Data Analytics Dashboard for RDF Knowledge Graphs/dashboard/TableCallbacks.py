
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_table
import dash
import pandas as pd
from MainApp import app
from HelperFunctions import start_table_df


def gen_table_function(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows):

    selectedRowsIndex=derived_virtual_selected_rows
    resultListdataframe = pd.read_json(jsonified_ResultListdataframe, orient='split')
    mycolumns = [{'name': index, 'id': index,"deletable": True, "selectable": True} for index in resultListdataframe.columns]

    return  resultListdataframe, mycolumns  

#---------------------------------------------------Table Callback--------------------------------------------------------------
#generate table from query    
@app.callback(
    [Output("alert-table", "is_open"),Output("Resulttable", "data"), 
     Output("Resulttable", "columns"), Output("MainTableDiv", "style")],
    [Input("submit-btn", "n_clicks"),
    Input("intermediate-value", "children"),
    Input("Resulttable", "derived_virtual_selected_rows")],
)    

def gen_table(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows):
    
    if n_clicks > 0:
        
        if jsonified_ResultListdataframe is None:
            
            return True, start_table_df.to_dict('records'), [{'id': '', 'name': ''}], {'display': 'none'}
        
        else:
            resultListdataframe,mycolumns =  gen_table_function(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows)
            return False, resultListdataframe.to_dict('records'), mycolumns, {'display': 'block'}
        
    return False, start_table_df.to_dict('records'), [{'id': '', 'name': ''}], {'display': 'none'}    

#---------------------------------------------------Table Compare Callback--------------------------------------------------------------
#generate table from query2    
@app.callback(
    [Output("alert-table2", "is_open"),Output("Resulttable2", "data"),
     Output("Resulttable2", "columns"), Output("MainTableDiv2", "style")],
    [Input("compare-btn", "n_clicks"),
    Input("intermediate-value2", "children"),
    Input("Resulttable2", "derived_virtual_selected_rows")],
)    

def gen_table(n_clicks, jsonified_ResultListdataframe, derived_virtual_selected_rows):
 
    if n_clicks > 0:   
        
        if jsonified_ResultListdataframe is None:

            return True, start_table_df.to_dict('records'), [{'id': '', 'name': ''}], {'display': 'none'}
        
        else:
            resultListdataframe,mycolumns =  gen_table_function(n_clicks, jsonified_ResultListdataframe,derived_virtual_selected_rows)
            return False, resultListdataframe.to_dict('records'), mycolumns, {'display': 'block'}
        
    return False, start_table_df.to_dict('records'), [{'id': '', 'name': ''}], {'display': 'none'}