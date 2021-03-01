import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_table
import dash
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointNotFound, EndPointInternalError, Unauthorized, URITooLong, SPARQLWrapperException
import urllib.request,urllib.parse,urllib.error
import pandas as pd
import json
import rdflib
from pyparsing import ParseException
from rdflib.plugin import get as plugin
import time
from MainApp import app
from HelperFunctions import Log_Query_Data, Log_Parse_Data


def extract_lon_lat(ResultListdataframe):
    
        ResultListdataframe['coord'] = ResultListdataframe['coord'].str.split("\(", expand=True)[1]
        ResultListdataframe['coord'] = ResultListdataframe['coord'].str.split("\)", expand=True)
        ResultListdataframe['lon'] = ResultListdataframe['coord'].str.split(" ", expand=True)[0].astype(float)
        ResultListdataframe['lat'] = ResultListdataframe['coord'].str.split(" ", expand=True)[1].astype(float)
        
        return ResultListdataframe


def sql(value, endpoint):

    
    if 'sparql' in endpoint.split('/'):
        start = time.perf_counter()
        sparql = SPARQLWrapper(endpoint)

        sparql.setQuery(value)
        sparql.setReturnFormat(JSON)

        try: 
            processed_results = sparql.query().convert()
        # to catch possible errors
        except (EndPointInternalError, EndPointNotFound):
            return  None

        except (QueryBadFormed, SPARQLWrapperException):
            return None

        except (Unauthorized, URITooLong):
            return None

        except urllib.error.URLError:
            return None

        except urllib.error.HTTPError:
            return None

        # get the columns and the values for each columns 
        if len(processed_results["results"]["bindings"]) > 0:
            cols = processed_results['head']['vars']

            out = []
            for row in processed_results['results']['bindings']:
                item = []
                for c in cols: 
                    item.append(row.get(c, {}).get('value'))
                out.append(item)
            # save the results as data frame
            ResultListdataframe = pd.DataFrame(out, columns=cols)

            #to parse long string to remove extra url
            for col in ResultListdataframe.columns:
                ResultListdataframe[col] = ResultListdataframe[col].str.split(pat="/").str[-1]

            # remove strings in coordinates
            if set(['lon','lat']).issubset(ResultListdataframe.columns): 
                if 'coord' in ResultListdataframe.columns:
                    ResultListdataframe['coord'] = ResultListdataframe['coord'].str.split("\(", expand=True)[1]
                    ResultListdataframe['coord'] = ResultListdataframe['coord'].str.split("\)", expand=True)
                return ResultListdataframe
 

            elif 'coord' in ResultListdataframe.columns:
                ResultListdataframe = extract_lon_lat(ResultListdataframe)
                return ResultListdataframe

            else:
                return ResultListdataframe 

            end = time.perf_counter()
            Log_Query_Data(endpoint,value, len(ResultListdataframe.index),ResultListdataframe.shape[1],round(end - start,3))


         
    else:
        return None

def get_data_function(value, endpoint,endpointType):
        if endpointType == '1':
            start = time.perf_counter()
            g = rdflib.Graph()
            g.parse("data/" + endpoint, format=rdflib.util.guess_format("data/" + endpoint))
            end = time.perf_counter()
            Log_Parse_Data(endpoint,round(end - start,3))
            start2 = time.perf_counter()
            try:
                qres = g.query(value)
            except ParseException:
                return True,False,None

            ResultListdataframe = pd.DataFrame (qres.bindings)
            end2 = time.perf_counter()
            Log_Query_Data(endpoint,value, len(ResultListdataframe.index),ResultListdataframe.shape[1],round(end2 - start2,3))            
            return False, False, ResultListdataframe.to_json(date_format='iso', orient='split')
        
        # online querying  
        elif endpointType == '2':

            ResultListdataframe = sql(value, endpoint) 
            
            if ResultListdataframe is None:
              
                return True, False, None
            else:
                return False, False, ResultListdataframe.to_json(date_format='iso', orient='split')
        else:
            return False, False, None

#---------------------------------------------------Get-data Callback--------------------------------------------------------------
@app.callback([Output("alert-except", "is_open"),Output("alert-value", "is_open"),Output("intermediate-value", 'children')],
               Input("submit-btn", "n_clicks"),
              [State("query-text","value"), State("query-endpoint","value"),State("endpoint-query1-type", "title")])

def get_data(n_clicks, value, endpoint,endpointType):
     if n_clicks > 0:
            
            if (len(value)==0 or len(endpoint)==0):
                return False,True, None
            else:
                return get_data_function(value, endpoint,endpointType)
          
    
     return False, False, None

#---------------------------------------------------Get-data Compare Callback--------------------------------------------------------------
@app.callback([Output("alert-except2", "is_open"),Output("alert-value2", "is_open"),Output("intermediate-value2", 'children')],
               Input("compare-btn", "n_clicks"),
              [State("query-text2","value"), State("query-endpoint2","value"),State("endpoint-query2-type", "title")])
def get_data(n_clicks, value, endpoint,endpointType):
     if n_clicks > 0:
            
            if (len(value)==0 or len(endpoint)==0):
                return False,True, None
            else:
                return get_data_function(value, endpoint,endpointType)
    
     return False, False, None
