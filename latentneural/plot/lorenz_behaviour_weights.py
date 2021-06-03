import pandas as pd
import numpy as np
import plotly.express as px


def lorenz_behaviour_weights(condition: int, behaviour_w: np.ndarray):
    data = pd.DataFrame({
        'weight': behaviour_w[condition, :, :].flatten(), 
        'latent': [np.floor_divide(i, behaviour_w.shape[-1]) for i in range(behaviour_w[condition,:,:].size)],
        'behavioural': [np.mod(i, behaviour_w.shape[-1]) for i in range(behaviour_w[condition,:,:].size)]})
    fig = px.histogram(
        data, 
        x=["behavioural", "latent"], 
        y="weight", 
        color="latent", 
        barmode="group", 
        title="Latent Factor Loadings of Condition #%d" % (condition,),
        labels={
            "weight": "Weight",
            "behavioural": "Behavioural",
            "latent": "Latent Factor"
        },
        nbins=behaviour_w.shape[-1]
    ).update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        ),
        xaxis_title="Behavioural Dimension",
        yaxis_title="Weight"
    )
    return fig