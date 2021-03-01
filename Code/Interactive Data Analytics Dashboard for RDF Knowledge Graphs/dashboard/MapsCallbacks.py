import plotly.express as px
import plotly.graph_objs as go
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash
import pandas as pd

from MainApp import app


def plot_map_function(ResultListdataframe, df_lat, df_lon, index0, index1):
    
        fig = go.Figure()
        
        
        fig.add_trace(go.Scattergeo(
                        lat = df_lat,
                        lon = df_lon,
                        #text = df_lon.astype(str),
                        text = ResultListdataframe.iloc[:, 0].astype(str),
                        mode = 'markers',
                                                         
        ))  
                    
        buttonlist = []
        for col in ResultListdataframe.columns:
                buttonlist.append(
                    dict(
                        args=['text', ResultListdataframe[col].astype(str)],
                        label=str(col),
                        method='restyle', 
                        )
                ),
                
        buttonlist_lon = []
        for col in ResultListdataframe.columns:
                buttonlist_lon.append(
                    dict(
                        args=['lon', [ResultListdataframe[str(col)]]],
                        label=str(col),
                        method='restyle', 
                        )
                ),
                
        buttonlist_lat = []
        for col in ResultListdataframe.columns:
                buttonlist_lat.append(
                    dict(
                        args=['lat', [ResultListdataframe[str(col)]]],
                        label=str(col),
                        method='restyle', 
                        )
                )

                    
        fig.update_geos(
            visible=True,
            resolution=110,
            showcountries=True, countrycolor="Black",
            showsubunits=True, subunitcolor="Blue",
            scope="world",
          )

        fig.update_layout(
                autosize=True,
                updatemenus=[
                    dict(
                        buttons=buttonlist,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=-0.9,
                        xanchor="left",
                        y=1.25,
                        yanchor="top",
                        active=0,
                        ), 
                    
                     dict(
                        buttons=buttonlist_lat,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=-0.9,
                        xanchor="left",
                        y=0.90,
                        yanchor="top",
                        active=index0,
                        ), 
                    
                      dict(
                        buttons=buttonlist_lon,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=-0.9,
                        xanchor="left",
                        y=0.60,
                        yanchor="top",
                        active=index1,
                        ), 
                    
                    

                     ],
              annotations=[   
                    dict(text="hoverlabel", x=-0.9, xref="paper", y=1.29, yref="paper",align="left", showarrow=False),
                    dict(text="latitude", x=-0.9, xref="paper", y=0.94, yref="paper",align="left", showarrow=False),
                    dict(text="longitude", x=-0.9, xref="paper", y=0.61, yref="paper", showarrow=False),
                    ],
                   
                    #height = 253, 
                    #width = 1000,
                  #margin={"r":0,"t":0,"l":0,"b":0}

                )

            
        return fig

#map function
def gen_map_function(n_clicks, jsonified_ResultListdataframe):
    
    if n_clicks > 0:
        
        if jsonified_ResultListdataframe is None:
            
              return False, True, { }, {'display': 'none'}
        
        else:
       
            ResultListdataframe = pd.read_json(jsonified_ResultListdataframe, orient='split')
        
            if len(ResultListdataframe.columns) > 1:
                
                if set(['lon','lat']).issubset(ResultListdataframe.columns): 
                    index0 = ResultListdataframe.columns.get_loc('lat') 
                    index1 = ResultListdataframe.columns.get_loc('lon')  
                    fig = plot_map_function(ResultListdataframe, ResultListdataframe['lat'], ResultListdataframe['lon'], index0,index1)

                    return False, False, fig, {'display': 'block'}

                else:
                    fig = plot_map_function(ResultListdataframe, ResultListdataframe.iloc[:, 0], ResultListdataframe.iloc[:, 1], 0, 1)

                    return True, False, fig, {'display': 'block'}
            else:
                return False, True, { }, {'display': 'none'}
            
        
    return False, False, { }, {'display': 'none'}


# map callback
@app.callback(
    [Output("alert-map", "is_open"),Output("alert-map3", "is_open"),Output("map", "figure"), Output('result-map', 'style')],
    [Input("submit-btn", "n_clicks"),
    Input("intermediate-value", "children")],
)
def gen_map(n_clicks, jsonified_ResultListdataframe):
    return gen_map_function(n_clicks, jsonified_ResultListdataframe)
    
# map compare callback
@app.callback(
    [Output("alert-map2", "is_open"),Output("alert-map4", "is_open"),Output("map2", "figure"), Output('result-map2', 'style')],
    [Input("compare-btn", "n_clicks"),
    Input("intermediate-value2", "children")] ,
)
def gen_map(n_clicks, jsonified_ResultListdataframe):
    return gen_map_function(n_clicks, jsonified_ResultListdataframe)