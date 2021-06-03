import plotly.graph_objects as go
import numpy as np
import plotly.express as px


def lorenz_behaviour_dynamics(condition: int, trial: int, time: np.ndarray, behaviour_w: np.ndarray, latent: np.ndarray, behaviour: np.ndarray):
    noisless = latent[condition,trial,:,-behaviour_w.shape[1]:] @ behaviour_w[condition]
    fig = go.Figure()

    for d in range(behaviour_w.shape[2]):
        # Create and style traces
        fig.add_trace(go.Scatter(x=time, y=noisless[:, d], name='%d Noisless' % (d),
                                line=dict(color=px.colors.qualitative.Plotly[d], width=10), opacity=0.3))
        fig.add_trace(go.Scatter(x=time, y=behaviour[condition,trial,:,d], name='%d Noisy' % (d),
                                line = dict(color=px.colors.qualitative.Plotly[d], width=2)))

    # Edit the layout
    fig.update_layout(title='Behavioural Trajectories of Condition #%d, Trial #%d' % (condition, trial),
                    xaxis_title='Time',
                    yaxis_title='Value')

    return fig