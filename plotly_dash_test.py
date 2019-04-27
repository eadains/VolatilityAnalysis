import dash
import functools
import numpy as np
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from vol_calcs import HorizontalSkew, VolCone, HistoricalVolatility, VolDiffIndex

app = dash.Dash()
server = app.server
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

app.layout = html.Div([
    dcc.Dropdown(options=
                 [{"label": "S&P 500 Index", "value": "SPXW"},
                  {"label": "Russell 2000 Index", "value": "RUTW"},
                  {"label": "Nasdaq 100 Index", "value": "NDX"}],
                 id="dropdown", value="SPXW"),
    dcc.Graph(id="hor_skew"),
    dcc.Graph(id="vol_cone"),
    dcc.Graph(id='hv'),
    dcc.Graph(id='hv_index')
])


@app.callback(
    Output(component_id='hor_skew', component_property='figure'),
    [Input(component_id='dropdown', component_property='value')]
)
def update_hor_skew(input_value):
    skew_object = HorizontalSkew(input_value)
    fitted = skew_object.fitted_data
    data = skew_object.recent_data

    traces = [
        go.Scatter(name="Fitted Data", x=fitted.x, y=fitted.y, mode="lines", hoverlabel={'namelength': -1}),
        go.Scatter(name="Actual Data", x=data.x, y=data.y, mode="markers", fill="tonextx",
                   hoverlabel={'namelength': -1})
    ]

    return {
        "data": traces,
        "layout": go.Layout(
            xaxis={"title": "Days to Expiration", "range": [0, fitted[0].size]},
            yaxis={"title": "Implied Volatility"},
            title="Implied Volatility by Days to Expiration"
        )
    }


@app.callback(
    Output(component_id='vol_cone', component_property='figure'),
    [Input(component_id='dropdown', component_property='value')]
)
def update_vol_cone(input_value):
    vol_cone = VolCone(input_value)
    vol_data = vol_cone.vol_cone
    skew_data = vol_cone.current_skew

    # Get index of vol_data as numpy array
    x_cone_points = vol_data.index.values

    traces = [
        go.Scatter(name='Volatility Max', x=x_cone_points, y=vol_data['std_max'], hoverlabel={'namelength': -1}),
        go.Scatter(name='Volatility Average', x=x_cone_points, y=vol_data['std_avg'], hoverlabel={'namelength': -1}),
        go.Scatter(name='Volatility Minimum', x=x_cone_points, y=vol_data['std_min'], hoverlabel={'namelength': -1}),
        go.Scatter(name='Current Skew', x=skew_data.x, y=skew_data.y, hoverlabel={'namelength': -1},
                   mode="lines+markers")
    ]

    return {
        "data": traces,
        "layout": go.Layout(
            xaxis={"title": "Days to Expiration", "range": [0, x_cone_points.size]},
            yaxis={"title": "Volatility"},
            title="Volatility Cone"
        )
    }


@app.callback(
    Output(component_id='hv', component_property='figure'),
    [Input(component_id='dropdown', component_property='value')]
)
def update_hv(input_value):
    # removes W from end for option tickers. ie SPXW becomes SPX
    if input_value[-1] == 'W':
        index_symbol = input_value[:-1]
    else:
        index_symbol = input_value

    hv = HistoricalVolatility(index_symbol)
    # Get mean and fill array with value to create straight line
    hv30_mean = np.nanmean(hv.hv30.y)
    hv30_mean_y = np.full(hv.hv30.x[-252:].size, hv30_mean)

    traces = [
        # Indexing to only display last year of data
        go.Scatter(name='HV10', x=hv.hv10.x[-252:], y=hv.hv10.y[-252:], hoverlabel={'namelength': -1}),
        go.Scatter(name='HV30', x=hv.hv30.x[-252:], y=hv.hv30.y[-252:], hoverlabel={'namelength': -1}),
        go.Scatter(name='HV60', x=hv.hv60.x[-252:], y=hv.hv60.y[-252:], hoverlabel={'namelength': -1}),
        go.Scatter(name='HV30 Average', x=hv.hv30.x[-252:], y=hv30_mean_y, hoverlabel={'namelength': -1})
    ]

    return {
        "data": traces,
        "layout": go.Layout(
            xaxis={"title": "Days"},
            yaxis={"title": "Volatility"},
            title="Historical Volatility"
        )
    }


@app.callback(
    Output(component_id='hv_index', component_property='figure'),
    [Input(component_id='dropdown', component_property='value')]
)
def update_hv_index(input_value):
    # removes W from end for option tickers. ie SPXW becomes SPX
    if input_value[-1] == 'W':
        index_symbol = input_value[:-1]
    else:
        index_symbol = input_value

    hv_index = VolDiffIndex(index_symbol)
    hv_index_mean = np.nanmean(hv_index.hv30_index.y)
    hv30_mean_y = np.full(hv_index.hv30_index.x[-252:].size, hv_index_mean)

    traces = [
        # Indexing to only display last year of data
        go.Scatter(name='HV30 Diff', x=hv_index.hv30_index.x[-252:], y=hv_index.hv30_index.y[-252:],
                   hoverlabel={'namelength': -1}),
        go.Scatter(name='Diff Average', x=hv_index.hv30_index.x[-252:], y=hv30_mean_y, hoverlabel={'namelength': -1})
    ]

    return {
        "data": traces,
        "layout": go.Layout(
            xaxis={"title": "Days"},
            yaxis={"title": "Volatility Difference"},
            title="Implied/Historical Volatility Difference"
        )
    }


# @app.callback(
#     Output(component_id='vol_predict', component_property='figure'),
#     [Input(component_id='dropdown', component_property='value')]
# )
# @functools.lru_cache(maxsize=16)
# def update_vol_predict(input_value):
#     # removes W from end for option tickers. ie SPXW becomes SPX
#     if input_value[-1] == 'W':
#         index_symbol = input_value[:-1]
#     else:
#         index_symbol = input_value
#
#     model = VolModelPredict(index_symbol, 60)
#
#     traces = [
#         go.Scatter(name='Volatility Prediction', x=model.vol_prediction.x, y=model.vol_prediction.y,
#                    hoverlabel={'namelength': -1})
#     ]
#
#     return {
#         "data": traces,
#         "layout": go.Layout(
#             xaxis={"title": "Days Ahead"},
#             yaxis={"title": "Predicted Volatility"},
#             title="60 Day Volatility Prediction GARCH"
#         )
#     }

if __name__ == '__main__':
    app.run_server()
