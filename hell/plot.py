import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def hit_ratio(results: pd.DataFrame) -> go.Figure:
    hit_ratio = results.groupby("method").hit.mean()
    hit_ratio = hit_ratio.sort_values(ascending=False)

    fig = px.bar(
        hit_ratio,
        x=hit_ratio.index,
        y=hit_ratio.values,
        color=hit_ratio.index,
        title="Hit ratio",
        labels={
            "y": "Ratio",
            "method": "Method"
        }
    )

    return fig


def ranking_metric(results: pd.DataFrame, metric: str, title: str) -> go.Figure:
    positive_results = results[results.hit]
    method_order = positive_results.groupby("method")[metric].median() \
        .sort_values(ascending=False).index

    method_order = {method: order for order, method in enumerate(method_order)}
    positive_results = positive_results.sort_values(
        by="method",
        key=lambda m: m.map(method_order)
    )

    fig = px.box(
        positive_results,
        x="method",
        y=metric,
        color="method",
        title=title,
        labels={
            "method": "Method",
            "metric": "Value"
        }
    )

    return fig


def xgb_results(results) -> go.Figure:
    fig = go.Figure()

    train_trace = go.Scatter(
        y=results["validation_0"]["rmse"],
        name="train",
        mode="lines"
    )

    val_trace = go.Scatter(
        y=results["validation_1"]["rmse"],
        name="val",
        mode="lines"
    )

    fig.add_trace(train_trace)
    fig.add_trace(val_trace)

    fig.update_layout(
        title="XGB fit",
        xaxis_title="Iteration",
        yaxis_title="RMSE"
    )

    return fig
