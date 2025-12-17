from __future__ import annotations

import time
from typing import Dict, Optional

import streamlit as st

from training_sim import (
    MODEL_REGISTRY,
    BaseModel,
    generate_dataset,
    instantiate_model,
)
from visualizations import (
    make_accuracy_curve,
    make_classification_plot,
    make_loss_curve,
    make_network_diagram,
    make_regression_plot,
)


def initialise_state() -> None:
    if "initialised" in st.session_state:
        return
    default_model = list(MODEL_REGISTRY.keys())[0]
    st.session_state.initialised = True
    st.session_state.model_name = default_model
    st.session_state.learning_rate = 0.05
    st.session_state.max_epochs = 120
    st.session_state.data_seed = 42
    st.session_state.history = []
    st.session_state.is_training = False
    st.session_state.current_epoch = 0
    st.session_state.model = instantiate_model(default_model, st.session_state.learning_rate)
    load_dataset()


def load_dataset(new_seed: Optional[int] = None) -> None:
    if new_seed is not None:
        st.session_state.data_seed = int(new_seed)
    x_train, x_val, y_train, y_val = generate_dataset(
        st.session_state.model_name, st.session_state.data_seed
    )
    st.session_state.data = {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
    }


def reset_model() -> None:
    st.session_state.model = instantiate_model(
        st.session_state.model_name, st.session_state.learning_rate
    )
    st.session_state.history = []
    st.session_state.current_epoch = 0
    st.session_state.is_training = False


def reboot_environment(new_seed: Optional[int] = None) -> None:
    reset_model()
    load_dataset(new_seed=new_seed)


def perform_training_step() -> None:
    model: BaseModel = st.session_state.model
    model.set_learning_rate(st.session_state.learning_rate)
    data = st.session_state.data
    metrics = model.train_epoch(
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
    )
    st.session_state.history.append(metrics)
    st.session_state.current_epoch = metrics["epoch"]


def maybe_continue_training() -> bool:
    if not st.session_state.is_training:
        return False
    if st.session_state.current_epoch >= st.session_state.max_epochs:
        st.session_state.is_training = False
        return False
    perform_training_step()
    return True


def main() -> None:
    st.set_page_config(
        page_title="ML Training Visualiser",
        page_icon="📈",
        layout="wide",
    )
    initialise_state()
    st.title("Machine Learning Training Visualiser")
    st.caption(
        "Interactively explore how weights, loss, and decision boundaries evolve during learning."
    )

    with st.sidebar:
        st.header("Experiment Setup")
        model_names = list(MODEL_REGISTRY.keys())
        selected_model = st.selectbox(
            "Model type",
            options=model_names,
            index=model_names.index(st.session_state.model_name),
        )
        if selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            reboot_environment()
        learning_rate = st.slider(
            "Learning rate",
            min_value=0.001,
            max_value=0.3,
            value=float(st.session_state.learning_rate),
            step=0.001,
            format="%.3f",
        )
        if learning_rate != st.session_state.learning_rate:
            st.session_state.learning_rate = learning_rate
            st.session_state.model.set_learning_rate(learning_rate)
        max_epochs = st.slider(
            "Number of epochs",
            min_value=10,
            max_value=400,
            value=int(st.session_state.max_epochs),
            step=5,
        )
        if max_epochs != st.session_state.max_epochs:
            st.session_state.max_epochs = max_epochs
            st.session_state.current_epoch = min(st.session_state.current_epoch, max_epochs)
        seed_value = st.number_input(
            "Data seed",
            min_value=0,
            max_value=10_000,
            value=int(st.session_state.data_seed),
            step=1,
        )
        if st.button("Regenerate data"):
            reboot_environment(new_seed=seed_value)
        st.divider()
        st.markdown(
            "**Legend**\n- Blue: raw data\n- Green: correct predictions / model fit\n- Red: error magnitude"
        )
        st.markdown(
            "Adjust hyperparameters and step through training to watch gradient descent minimise loss."
        )

    control_cols = st.columns(3)
    play_label = "Play Training" if not st.session_state.is_training else "Pause Training"
    if control_cols[0].button(play_label, use_container_width=True):
        st.session_state.is_training = not st.session_state.is_training
        if st.session_state.is_training:
            st.rerun()
    if control_cols[1].button("Next Epoch", use_container_width=True):
        if st.session_state.current_epoch < st.session_state.max_epochs:
            perform_training_step()
    if control_cols[2].button("Reset Model", use_container_width=True):
        reboot_environment()

    should_queue_rerun = maybe_continue_training()

    latest_metrics: Optional[Dict[str, Optional[float]]] = (
        st.session_state.history[-1] if st.session_state.history else None
    )
    progress = (
        st.session_state.current_epoch / st.session_state.max_epochs
        if st.session_state.max_epochs
        else 0.0
    )
    st.progress(progress)
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Epoch",
        f"{st.session_state.current_epoch}/{st.session_state.max_epochs}",
    )
    train_loss_value = (
        f"{latest_metrics['train_loss']:.4f}" if latest_metrics else "—"
    )
    val_loss_value = (
        f"{latest_metrics['val_loss']:.4f}" if latest_metrics else "—"
    )
    metric_cols[1].metric("Train Loss", train_loss_value)
    metric_cols[2].metric("Validation Loss", val_loss_value)
    if latest_metrics and latest_metrics.get("train_acc") is not None:
        accuracy_text = f"{latest_metrics['train_acc']:.3f}"
        if latest_metrics.get("val_acc") is not None:
            accuracy_text += f" / {latest_metrics['val_acc']:.3f}"
        metric_cols[3].metric("Accuracy (train/val)", accuracy_text)
    else:
        metric_cols[3].metric("Accuracy (train/val)", "—")

    data = st.session_state.data
    model: BaseModel = st.session_state.model
    plot_cols = st.columns([2.2, 1.4])
    if model.task_type == "regression":
        regression_fig = make_regression_plot(
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            model,
        )
        plot_cols[0].plotly_chart(regression_fig, use_container_width=True)
    else:
        classification_title = (
            "Logistic Decision Boundary"
            if st.session_state.model_name == "Logistic Regression"
            else "Neural Network Decision Boundary"
        )
        classification_fig = make_classification_plot(
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            model,
            classification_title,
        )
        plot_cols[0].plotly_chart(classification_fig, use_container_width=True)
    network_fig = make_network_diagram(model)
    if network_fig is not None:
        plot_cols[1].plotly_chart(network_fig, use_container_width=True)
    else:
        plot_cols[1].markdown(
            "**How to read the charts**\n\n"
            "- The loss chart traces gradient descent.\n"
            "- Accuracy tracks correct predictions in green.\n"
            "- Residual lines (red) emphasise current errors."
        )

    loss_fig = make_loss_curve(st.session_state.history)
    st.plotly_chart(loss_fig, use_container_width=True)
    accuracy_fig = make_accuracy_curve(st.session_state.history)
    if accuracy_fig is not None:
        st.plotly_chart(accuracy_fig, use_container_width=True)

    st.info(
        "Use play/pause or next epoch to animate training. Hover charts for exact values per epoch."
    )

    if should_queue_rerun:
        time.sleep(0.4)
        st.rerun()


if __name__ == "__main__":
    main()
