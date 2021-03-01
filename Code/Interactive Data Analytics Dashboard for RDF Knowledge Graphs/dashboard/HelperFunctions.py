
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import datetime
import io
import sys
import os
import csv
import pandas as pd
import time
from MainApp import app

#-----------------------------------------initial empty table ----------------------------------------------------------------

start_table_df = pd.DataFrame(columns=[''])

#---------------------------------------------------Upload File Callback for query1--------------------------------------------------------------
UPLOAD_DIRECTORY = "data/"
@app.callback(Output('endpoint-query1-intermediate-value1', 'title'),
              [
              Input('upload-data', 'fileNames'),
              Input('upload-data', 'isCompleted'),
             
              ])
def update_output(filenames, iscompleted):   
        if not iscompleted:
            start = time.perf_counter()
            return
        if filenames is not None:
            end = time.perf_counter()
#             Log_File_Data(filenames[0],os.path.getsize(UPLOAD_DIRECTORY+filenames[0]), round(end - start,3))
            return filenames[0]
        return ''
#---------------------------------------------------Upload File Compare Callback for query2--------------------------------------------------------------
@app.callback(Output('endpoint-query2-intermediate-value1', 'title'),
              [
              Input('upload-data2', 'fileNames'),
              Input('upload-data2', 'isCompleted'),
             
              ])
def update_output(filenames, iscompleted):   
        if not iscompleted:
            start = time.perf_counter()
            return
        if filenames is not None:
            end = time.perf_counter()
#             Log_File_Data(filenames[0],os.path.getsize(UPLOAD_DIRECTORY+filenames[0]), round(end - start,3))
            return filenames[0]
        return ''
    
    
#-----------------------------------------Log File Upload data----------------------------------------------------------------
def Log_File_Data(File_Name, File_Size, Total_Time_Taken):
    first_upload = True
    if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, 'FileUploadLog.csv')):
        first_upload = False
    with open(os.path.join(UPLOAD_DIRECTORY, 'FileUploadLog.csv'), 'a', newline='') as file:
        writer = csv.writer(file, delimiter= ';')
        if (first_upload):
            writer.writerow(["File Name", "File Size in Bytes","Total Time Taken in seconds"])
        writer.writerow([File_Name, File_Size, Total_Time_Taken])
    
#-----------------------------------------Log Query data----------------------------------------------------------------
def Log_Query_Data(Endpoint, query, rows, cols, Total_Time_Taken):
    first_upload = True
    if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, 'QueryLog.csv')):
        first_upload = False
    with open(os.path.join(UPLOAD_DIRECTORY, 'QueryLog.csv'), 'a', newline='') as file:
        writer = csv.writer(file, delimiter= ';')
        if (first_upload):
            writer.writerow(["Endpoint","query","Rows","Columns","Total Time Taken in seconds"])
        writer.writerow([Endpoint,query,rows,cols,Total_Time_Taken])

#-----------------------------------------Log RDF Parsing Time----------------------------------------------------------------
def Log_Parse_Data(Endpoint,Total_Time_Taken):
    first_upload = True
    if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, 'RDFParseLog.csv')):
        first_upload = False
    with open(os.path.join(UPLOAD_DIRECTORY, 'RDFParseLog.csv'), 'a', newline='') as file:
        writer = csv.writer(file, delimiter= ';')
        if (first_upload):
            writer.writerow(["Endpoint","Total Time Taken in seconds"])
        writer.writerow([Endpoint,Total_Time_Taken])
        
#-----------------------------------------read query samples file----------------------------------------------------------------
#make query examples table from csv file
with open(os.path.join(UPLOAD_DIRECTORY, 'Book3.csv'), 'rt') as f:
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
       
#-----------------------------------------toggle query box callback----------------------------------------------------------------
# toggle query box

@app.callback(Output('compare-box', 'style'),
              Input('toggle-compare', 'n_clicks')
             )
def toggle_compare(n_clicks):
    if n_clicks > 0:
        if (n_clicks % 2) == 0:
            return {'top': '-335px'}
        else:   
            return {'top': '0'}
    return {}