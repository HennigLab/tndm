import pandas as pd
import numpy as np
import plotly.express as px


def lorenz_dynamics(condition: int, latent: np.ndarray):
    x, y, z = latent[condition,:,:,:].T
    data = pd.DataFrame({'x': x.T.flatten(), 'y': y.T.flatten(), 'z': z.T.flatten(), 'trial': [np.floor_divide(i, latent.shape[2]) for i in range(z.size)]})
    fig = px.line_3d(
        data, 
        x="x", 
        y="y", 
        z="z", 
        color="trial", 
        title="Latent Dynamics of Condition #%d" % (condition,),
        labels={
            "trial": "Trial Number"
        })
    return fig