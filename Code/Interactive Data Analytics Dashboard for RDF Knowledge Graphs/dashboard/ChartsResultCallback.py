import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_table
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
from MainApp import app


def gen_graph_function(jsonified_ResultListdataframe):
    
           
        if jsonified_ResultListdataframe is None:
            
            return True, { },{'display': 'none'}
        
        else: 
            ResultListdataframe = pd.read_json(jsonified_ResultListdataframe, orient='split')
            
            if len(ResultListdataframe.columns) > 1:
                
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
                        label="Y= "+str(col),
                        method='restyle'
                    )
                  )


                buttonlist_x = []
                for col in ResultListdataframe.columns:
                    buttonlist_x.append(
                        dict(
                        args=['x',[ResultListdataframe[str(col)]] ],
                        label="X= "+str(col),
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
                                ]),
                            dict(label="Scatter",
                                method="update",
                                args=[{"visible": [False, True, False]},
                                ]),
                            dict(label="line",
                                method="update",
                                args=[{"visible": [False, False, True]},
                                ]), 
                            ]),
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=-0.9,
                            xanchor="left",
                            y=2,
                            yanchor="top"
                        ),
                        dict(buttons=buttonlist,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=2,
                        yanchor="top",
                        active=1

                            ),

                        dict(buttons=buttonlist_x,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=-0.9,
                        xanchor="left",
                        y=0.7,
                        yanchor="top"

                            )
                    ],

                  #  annotations=[
                  #      dict(text="AXIS-Y", x=0.1, xref="paper", y=1.90, yref="paper",align="left", showarrow=False),
                  #      dict(text="AXIS-X", x=-0.9, xref="paper", y=-0.2,yref="paper", showarrow=False),
                  #      ],
                )

                fig.update_layout(
                    height=400,
                    autosize=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    template="plotly_white",
                )


                
                return False, fig, {'display': 'block'}
            else:
                return True, { }, {'display': 'none'}

#Chart Callback
@app.callback(
    [Output("alert-chart", "is_open"),Output("graph", "figure"), Output("result-graph", "style")],
    [Input("submit-btn", "n_clicks"),
    Input("intermediate-value", "children"),
    ],
)
def gen_graph(n_clicks, jsonified_ResultListdataframe):
    if n_clicks > 0:
        return gen_graph_function(jsonified_ResultListdataframe)
    return False, { }, {'display': 'none'}
#Charts Compare Callback
@app.callback(
    [Output("alert-chart2", "is_open"),Output("graph2", "figure"),  Output("result-graph2", "style")],
    [Input("compare-btn", "n_clicks"),
    Input("intermediate-value2", "children"),
    ],
)
def gen_graph(n_clicks, jsonified_ResultListdataframe):
    if n_clicks > 0:
        return  gen_graph_function(jsonified_ResultListdataframe)
    return False, { },{'display': 'none'}