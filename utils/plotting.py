import numpy as np
import plotly.graph_objects as go


def result_trace(x, y, name, color):
    return go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name=name,
        marker=dict(
            color=color,
            line=dict(
                color=color
            )
        )
    )


def result_mean_trace(x, y, name, color):
    return go.Scatter(
        x=x,
        y=len(x) * [np.mean(y)],
        mode='lines',
        name=name,
        line=dict(
            color=color,
            dash='dot'
        )
    )
