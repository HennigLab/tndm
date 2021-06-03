import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def lorenz_neural_dynamics(condition: int, trial: int, rates: np.ndarray, spikes: np.ndarray, time: np.ndarray):
    fig_rates = go.Figure(data=go.Heatmap(x=time, y=list(range(rates.shape[-1])), z=rates[condition, trial, :, :].T, colorscale="Blues"))

    fig_rates.update_layout(title='Firing Rates of Condition #%d, Trial #%d' % (condition, trial),
                    xaxis_title='Time',
                    yaxis_title='Neurons')

    fig_spikes = go.Figure(data=go.Heatmap(x=time, y=list(range(spikes.shape[-1])), z=spikes[condition, trial, :, :].T, colorscale="Blues"))

    fig_spikes.update_layout(title='Spike Trains of Condition #%d, Trial #%d' % (condition, trial),
                    xaxis_title='Time',
                    yaxis_title='Neurons')

    return fig_rates, fig_spikes