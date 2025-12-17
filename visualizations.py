"""Plotly figures used by the Streamlit interface."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import plotly.graph_objects as go

from training_sim import BaseModel, NeuralNetworkModel

Array = np.ndarray
_BLUE = "#1f77b4"
_RED = "#d62728"
_GREEN = "#2ca02c"


def make_loss_curve(history: List[Dict[str, Optional[float]]]) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(
            title="Loss vs Epoch",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="simple_white",
        )
        return fig
    epochs = [point["epoch"] for point in history]
    train_losses = [point["train_loss"] for point in history]
    val_losses = [point.get("val_loss") for point in history]
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines+markers",
            name="Train Loss",
            line=dict(color=_BLUE, width=3),
        )
    )
    if any(value is not None for value in val_losses):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_losses,
                mode="lines+markers",
                name="Validation Loss",
                line=dict(color=_RED, dash="dash", width=3),
            )
        )
    fig.update_layout(
        title="Loss vs Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def make_accuracy_curve(history: List[Dict[str, Optional[float]]]) -> Optional[go.Figure]:
    if not history or history[0].get("train_acc") is None:
        return None
    epochs = [point["epoch"] for point in history]
    train_accuracies = [point["train_acc"] for point in history]
    val_accuracies = [point.get("val_acc") for point in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_accuracies,
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color=_GREEN, width=3),
        )
    )
    if any(value is not None for value in val_accuracies):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_accuracies,
                mode="lines+markers",
                name="Validation Accuracy",
                line=dict(color="rgba(34,160,70,0.5)", dash="dash", width=3),
            )
        )
    fig.update_layout(
        title="Accuracy vs Epoch",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0.0, 1.05]),
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def make_regression_plot(
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
    model: Optional[BaseModel],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_train.flatten(),
            y=y_train.flatten(),
            mode="markers",
            name="Train Data",
            marker=dict(color=_BLUE, size=8, opacity=0.9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_val.flatten(),
            y=y_val.flatten(),
            mode="markers",
            name="Test Data",
            marker=dict(color="rgba(31,119,180,0.35)", size=8, symbol="x"),
        )
    )
    if model is not None:
        x_min = float(min(x_train.min(), x_val.min()))
        x_max = float(max(x_train.max(), x_val.max()))
        grid = np.linspace(x_min - 0.5, x_max + 0.5, 120).reshape(-1, 1)
        predictions = model.predict(grid).flatten()
        fig.add_trace(
            go.Scatter(
                x=grid.flatten(),
                y=predictions,
                mode="lines",
                name="Model Prediction",
                line=dict(color=_GREEN, width=3),
            )
        )
        residual_indices = np.linspace(0, x_train.shape[0] - 1, min(25, x_train.shape[0]), dtype=int)
        residual_x: List[float] = []
        residual_y: List[float] = []
        predicted_points = model.predict(x_train).flatten()
        for index in residual_indices:
            residual_x.extend([float(x_train[index, 0]), float(x_train[index, 0]), None])
            residual_y.extend([float(y_train[index, 0]), float(predicted_points[index]), None])
        fig.add_trace(
            go.Scatter(
                x=residual_x,
                y=residual_y,
                mode="lines",
                name="Residuals",
                line=dict(color=_RED, width=1, dash="dot"),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title="Regression Fit Evolution",
        xaxis_title="Feature",
        yaxis_title="Target",
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def _classification_colors(
    predictions: Array,
    targets: Array,
) -> List[str]:
    predicted_labels = (predictions >= 0.5).astype(int).flatten()
    true_labels = targets.flatten().astype(int)
    return [
        _GREEN if predicted == actual else _RED
        for predicted, actual in zip(predicted_labels, true_labels)
    ]


def make_classification_plot(
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
    model: Optional[BaseModel],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    if model is not None:
        train_predictions = model.predict(x_train)
        train_colors = _classification_colors(train_predictions, y_train)
        test_predictions = model.predict(x_val)
        test_colors = _classification_colors(test_predictions, y_val)
    else:
        train_colors = [_BLUE for _ in range(x_train.shape[0])]
        test_colors = ["rgba(31,119,180,0.35)" for _ in range(x_val.shape[0])]
        train_predictions = None
        test_predictions = None
    fig.add_trace(
        go.Scatter(
            x=x_train[:, 0],
            y=x_train[:, 1],
            mode="markers",
            name="Train Data",
            marker=dict(size=9, color=train_colors, line=dict(color="#1b4f72", width=0.5)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_val[:, 0],
            y=x_val[:, 1],
            mode="markers",
            name="Test Data",
            marker=dict(size=11, symbol="x", color=test_colors),
        )
    )
    if model is not None:
        x_span = np.linspace(
            min(x_train[:, 0].min(), x_val[:, 0].min()) - 0.5,
            max(x_train[:, 0].max(), x_val[:, 0].max()) + 0.5,
            200,
        )
        y_span = np.linspace(
            min(x_train[:, 1].min(), x_val[:, 1].min()) - 0.5,
            max(x_train[:, 1].max(), x_val[:, 1].max()) + 0.5,
            200,
        )
        grid_x, grid_y = np.meshgrid(x_span, y_span)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        probabilities = model.predict(grid_points).reshape(grid_x.shape)
        fig.add_trace(
            go.Contour(
                x=x_span,
                y=y_span,
                z=probabilities,
                contours=dict(start=0.0, end=1.0, size=0.25, showlabels=False),
                showscale=False,
                colorscale=[[0.0, "rgba(31,119,180,0.45)"], [1.0, "rgba(44,160,44,0.45)"]],
                opacity=0.4,
                name="Probability Field",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Contour(
                x=x_span,
                y=y_span,
                z=probabilities,
                contours=dict(start=0.5, end=0.5, size=0.5, coloring="none"),
                showscale=False,
                line=dict(color=_GREEN, width=3),
                name="Decision Boundary",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def make_network_diagram(model: BaseModel) -> Optional[go.Figure]:
    if not isinstance(model, NeuralNetworkModel):
        return None
    weight_layers = model.get_weight_matrices()
    input_size = weight_layers[0].shape[0]
    hidden_size = weight_layers[0].shape[1]
    output_size = weight_layers[1].shape[1]
    layer_positions = [0.0, 1.0, 2.0]
    node_positions: List[Dict[str, float]] = []
    for layer_index, (size, x_pos) in enumerate(
        zip([input_size, hidden_size, output_size], layer_positions)
    ):
        if size == 1:
            y_positions = [0.5]
        else:
            y_positions = np.linspace(0.1, 0.9, size)
        for node_index, y_pos in enumerate(y_positions):
            node_positions.append(
                {
                    "layer": layer_index,
                    "node": node_index,
                    "x": x_pos,
                    "y": float(y_pos),
                }
            )
    fig = go.Figure()
    max_weight = max(
        np.max(np.abs(matrix)) for matrix in weight_layers if matrix.size > 0
    )
    for layer_index, matrix in enumerate(weight_layers):
        source_layer = [pos for pos in node_positions if pos["layer"] == layer_index]
        target_layer = [pos for pos in node_positions if pos["layer"] == layer_index + 1]
        for source_idx, source_node in enumerate(source_layer):
            for target_idx, target_node in enumerate(target_layer):
                weight_value = matrix[source_idx, target_idx]
                width = 1.0 + 4.0 * (abs(weight_value) / max_weight if max_weight else 0.0)
                color = _GREEN if weight_value >= 0 else _RED
                fig.add_trace(
                    go.Scatter(
                        x=[source_node["x"], target_node["x"]],
                        y=[source_node["y"], target_node["y"]],
                        mode="lines",
                        line=dict(color=color, width=width),
                        hoverinfo="text",
                        text=f"w={weight_value:+.3f}",
                        showlegend=False,
                    )
                )
    for layer_index, (size, x_pos) in enumerate(
        zip([input_size, hidden_size, output_size], layer_positions)
    ):
        if size == 1:
            y_positions = [0.5]
        else:
            y_positions = np.linspace(0.1, 0.9, size)
        labels: List[str] = []
        if layer_index == 0:
            labels = [f"x{idx + 1}" for idx in range(size)]
        elif layer_index == 1:
            labels = [f"h{idx + 1}" for idx in range(size)]
        else:
            labels = ["ŷ"]
        fig.add_trace(
            go.Scatter(
                x=[x_pos] * size,
                y=y_positions,
                mode="markers+text",
                marker=dict(size=24, color="#f7f7f7", line=dict(color="#7f8c8d", width=1.5)),
                text=labels,
                textposition="middle center",
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Neural Network Weight Snapshot",
        template="simple_white",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )
    return fig
