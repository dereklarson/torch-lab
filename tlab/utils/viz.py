"""Grab-bag of methods for visualizing data.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import einops
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from tlab.experiment import Experiment
from tlab.observation import Observations
from tlab.optimize import Optimizer
from tlab.utils.util import fourier_basis, to_numpy

DEF_PLOTS = ["train_loss", "test_loss"]


def live_plot(**kwargs):
    fig = go.FigureWidget()
    _layout(fig, title="Live Training", size=(None, 400), log_x=True, log_y=True)
    return fig


def add_plots(fig: go.Figure, tag: str, plots=DEF_PLOTS):
    if not fig:
        return
    args = {"group": tag, "thinning": 0}
    for param in plots:
        fig.add_trace(_scatter(x=None, y=None, name=param, tag=tag, **args))


def update_plots(fig, obs: Observations, group_idx: int, plots=DEF_PLOTS):
    N = len(plots)
    idx = N * group_idx
    for param in plots:
        fig.data[idx].y = obs.data[param]
        idx += 1


def _scatter(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    tag: str,
    group: Optional[str] = None,
    thinning: int = 100,
    **kwargs,
):
    if thinning > 0:
        stride = len(x) // thinning
        x = x[::stride]
        y = y[::stride]
    legend_params = {}
    if group is not None:
        legend_params["legendgroup"] = group
        legend_params["legendgrouptitle_text"] = tag
        legend_params["name"] = name
    else:
        legend_params["name"] = tag + " " + name

    return go.Scatter(x=x, y=y, **legend_params)


def _layout(
    fig: go.Figure,
    title: str,
    size: Tuple[int, int],
    log_x=False,
    log_y=False,
    **kwargs,
) -> None:
    fig.update_layout(title=title, width=size[0], height=size[1])
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")


def plot_loss(optim: Optimizer, log_x=False, log_y=True) -> go.Figure:
    fig = go.Figure()
    _layout(fig, "Loss curve", (600, 400), log_x=log_x, log_y=log_y)
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    x = np.array(range(len(optim.train_losses)))
    fig.add_trace(_scatter(x, optim.train_losses, name=f"Train"))
    fig.add_trace(_scatter(x, optim.test_losses, name=f"Test"))
    return fig


def grid_plot(obs: Observations):
    if len(obs.ranges) > 3:
        logging.warning("Observations has more than 3 ranges, can't determine subplots")
    elif len(obs.ranges) == 3:
        pass
    elif len(obs.ranges) == 2:
        params = sorted(obs.ranges.keys())
        shape = [len(obs.ranges[par]) for par in params]
        fig = make_subplots(rows=shape[0], cols=shape[1], start_cell="bottom-left")

    for idx, config in enumerate(obs.configs):
        # TEMP
        obs_dict = obs
        tag = obs["tag"]
        loc = {"row": idx // shape[-1] + 1, "col": idx % shape[-1] + 1}
        x = np.array(range(len(obs_dict["train_loss"])))
        y_train = np.array(obs_dict["train_loss"])
        y_test = np.array(obs_dict["test_loss"])
        norm = (np.array(obs_dict["weight_norm"]),)

        fig.add_trace(_scatter(x, y_train, name="Train", tag=tag), **loc)
        fig.add_trace(_scatter(x, y_test, name="Test", tag=tag), **loc)
        fig.add_trace(_scatter(x, norm, name="WNorm", tag=tag), secondary_y=True, **loc)
    return fig


def group_plot(
    obs: Observations,
    fields: Tuple[List[str], ...],
    size: Tuple[int, int] = (1200, 800),
    title: str = "Unnamed",
    log_x=False,
    log_y=True,
    thinning=1000,
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _layout(fig, title, size, log_x=log_x, log_y=log_y)

    for idx, obs_dict in obs.items():
        tag = obs_dict["tag"]
        params = {"group": str(idx), "tag": tag, "thinning": thinning}
        x = np.array(range(len(obs_dict["train_loss"])))
        for param in fields[0]:
            fig.add_trace(_scatter(x, np.array(obs_dict[param]), name=param, **params))
        if len(fields) > 1:
            for param in fields[1]:
                if "comp_wnorm" in param:
                    name = param.split(".")[-1]
                else:
                    name = param
                fig.add_trace(
                    _scatter(x, np.array(obs_dict[param]), name=name, **params),
                    secondary_y=True,
                )
    return fig


def imshow(tensor, p: int, labels: Optional[Dict[str, str]] = None, **kwargs):
    tensor = torch.squeeze(tensor)
    px.imshow(
        to_numpy(tensor, flat=False),
        labels=labels or {},
        **kwargs,
    ).show()


def plot_ft(mat: torch.Tensor, p: int, **kwargs):
    basis = fourier_basis(p)
    labels = {"x": "X input", "y": "Embedding Dimension"}
    _kwargs = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
    _kwargs.update(kwargs)
    if len(mat.shape) == 3:
        ft_frames = to_numpy(torch.einsum("fpe,Ep -> feE", mat, basis))
        fig = px.imshow(ft_frames, animation_frame=0, labels=labels, **_kwargs)
        fig["layout"].pop("updatemenus")
        fig.show()
    else:
        ft = to_numpy(torch.einsum("pe,Ep -> eE", mat, basis))
        px.imshow(ft, labels=labels, **_kwargs).show()


def _select_from_exp(exp: Experiment, constraints: Dict[str, int] = None):
    constraints = constraints or {}
    for xcon in exp.configure():
        if any(xcon.params.get(key) not in val for key, val in constraints.items()):
            continue
        try:
            all_params = xcon.get_model_state(exp.path)
            yield xcon, all_params
        except Exception as exc:
            print(f"Error loading {xcon.filebase}: {exc}")
            continue


def plot_experiment_ft(
    exp: Experiment,
    param: str = "embed.W_E",
    constraints: Dict[str, int] = None,
    transpose: bool = False,
    comb: bool = False,
):
    weights = []
    for _, all_params in _select_from_exp(exp, constraints=constraints):
        param_data = all_params[param]
        if transpose:
            param_data = param_data.T
        if comb:
            mat = all_params["blocks.0.mlp.W_out"].T @ all_params["unembed.W_U"]
            param_data = mat.T
        weights.append(param_data)

    weight_stack = torch.stack(weights, dim=0)
    plot_ft(weight_stack, exp.defaults.get("value_range"))


def _update_layout(fig, **kwargs):
    for k, v in kwargs.items():
        try:
            arg = {k: v}
            fig.update_layout(**arg)
        except:
            logging.warning(f"Skipping invalid update_layout arg: {k}={v}")
    try:
        fig["layout"].pop("updatemenus")
    except:
        pass


def _create_slider(
    exp: Experiment,
    cols: int,
    constraints: Dict[str, int] = None,
) -> Dict[str, Any]:
    """Create a slider indexing the different configurations of an experiment."""
    frames = exp.count
    slider_steps = []
    for xcon, all_params in _select_from_exp(exp, constraints=constraints):
        step = dict(
            method="update",
            label=xcon.repr,
            args=[{"visible": [False] * (frames * cols)}],
        )
        for i in range(frames * cols):
            if (cols * (xcon.idx - 1)) <= i < (cols * xcon.idx):
                step["args"][0]["visible"][i] = True
        slider_steps.append(step)

    slider = dict(
        active=0,
        currentvalue={"prefix": "params: "},
        pad={"t": 50},
        steps=slider_steps,
    )
    return slider


def plot_tensor_vectors(
    exp: Experiment,
    param: str = "embed.W_E",
    constraints: Dict[str, int] = None,
    transpose: bool = True,
):
    traces = []
    cols = 1
    for xcon, all_params in _select_from_exp(exp, constraints=constraints):
        param_data = all_params[param]
        if transpose:
            param_data = param_data.T
        cols = max(cols, param_data.shape[0])
        for _, trace in enumerate(param_data, 1):
            traces.append(
                go.Scatter(
                    visible=(xcon.idx == 1),
                    line=dict(color="#22CCDD", width=2),
                    name="",
                    y=to_numpy(trace),
                )
            )

    fig = make_subplots(1, cols)
    for idx, trace in enumerate(traces):
        fig.add_trace(trace, 1, (idx % cols) + 1)

    slider = _create_slider(exp, cols, constraints=constraints)
    title = f"Tensor '{param}' as row vectors, varying: {list(exp.ranges.keys())}"
    fig.update_layout(sliders=[slider], title=title)
    fig.show()


def plot_attention_patterns(
    exp: Experiment, constraints: Dict[str, int] = None, **kwargs
):
    traces = []
    cols = 1
    for xcon, all_params in _select_from_exp(exp, constraints=constraints):
        cols = max(cols, xcon.model.n_heads)
        W_E = all_params["embed.W_E"]
        W_Q = all_params["blocks.0.attn.W_Q"]
        W_K = all_params["blocks.0.attn.W_K"]
        QK = torch.einsum("ahe,ahE -> aeE", W_Q, W_K)
        eQK = torch.einsum("ve,aeE -> avE", W_E, QK)
        attn = torch.einsum("avE,VE -> avV", eQK, W_E)
        for _, trace in enumerate(attn, 1):
            traces.append(go.Heatmap(visible=(xcon.idx == 1), z=to_numpy(trace)))

    fig = make_subplots(1, cols)
    for idx, trace in enumerate(traces):
        fig.add_trace(trace, 1, (idx % cols) + 1)

    slider = _create_slider(exp, cols, constraints=constraints)
    title = f"Attention Pattern, varying: {list(exp.ranges.keys())}"
    _update_layout(fig, title=title, sliders=[slider], **kwargs)
    fig.show()
    return fig


def plot_output_patterns(
    exp: Experiment,
    constraints: Dict[str, int] = None,
    include_unembed: bool = True,
    **kwargs,
):
    traces = []
    cols = 1
    for xcon, all_params in _select_from_exp(exp, constraints=constraints):
        cols = max(cols, xcon.model.n_heads)
        W_E = all_params["embed.W_E"]
        W_O = all_params["blocks.0.attn.W_O"]
        W_V = all_params["blocks.0.attn.W_V"]
        W_U = all_params["unembed.W_U"]
        try:
            OV = torch.einsum("aeh,ahE -> aeE", W_O, W_V)
        except:
            # Original dimensions were d_embed, n_heads * d_head, so rearrange
            expanded = einops.rearrange(W_O, "e (a h) -> a e h")
            OV = torch.einsum("aeh,ahE -> aeE", expanded, W_V)
        output = torch.einsum("aeE, vE -> ave", OV, W_E)
        if include_unembed:
            output = torch.einsum("Ev, aVE -> avV", W_U, output)
        for _, trace in enumerate(output, 1):
            traces.append(go.Heatmap(visible=(xcon.idx == 1), z=to_numpy(trace)))

    fig = make_subplots(1, cols)
    for idx, trace in enumerate(traces):
        fig.add_trace(trace, 1, (idx % cols) + 1)

    slider = _create_slider(exp, cols, constraints=constraints)
    title = f"Attention Pattern, varying: {list(exp.ranges.keys())}"
    _update_layout(fig, title=title, sliders=[slider], **kwargs)
    fig.show()
    return fig


def plot_final(
    exp: Experiment,
    constraints: Dict[str, int] = None,
    param: str = "embed.W_E",
    transpose: bool = True,
):
    for xcon, all_params in _select_from_exp(exp, constraints=constraints):
        param_data = all_params[param]
        if transpose:
            param_data = param_data.T


def line(x, y=None, hover=None, xaxis="", yaxis="", **kwargs):
    if type(y) == torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x) == torch.Tensor:
        x = to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()


def lines(
    lines_list,
    x=None,
    mode="lines",
    labels=None,
    xaxis="",
    yaxis="",
    title="",
    log_y=False,
    hover=None,
    **kwargs,
):
    # Helper function to plot multiple lines
    if type(lines_list) == torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x = np.arange(len(lines_list[0]))
    fig = go.Figure(layout={"title": title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line) == torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(
            go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs)
        )
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()


def tuple_lines(data, **kwargs):
    # 'data' will be a list of lists/tuples
    # series = np.empty((len(data), len(data[0])))
    data = np.array(data).swapaxes(0, 1)
    return lines(data, **kwargs)


def group_tuple_plot(
    obs: Observations,
    fields: Tuple[List[str], ...],
    size: Tuple[int, int] = (1200, 800),
    title: str = "Unnamed",
    log_x=False,
    log_y=True,
    thinning=1000,
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _layout(fig, title, size, log_x=log_x, log_y=log_y)

    for idx, obs_dict in obs.items():
        tag = obs_dict["tag"]
        params = {"group": str(idx), "tag": tag, "thinning": thinning}
        x = np.array(range(len(obs_dict["train_loss"])))
        for param in fields[0]:
            try:
                iter(obs_dict[param][0])
                data = np.array(obs_dict[param]).swapaxes(0, 1)
                for idx, series in enumerate(data):
                    fig.add_trace(_scatter(x, series, name=f"{param}-{idx}", **params))
            except TypeError:
                fig.add_trace(
                    _scatter(x, np.array(obs_dict[param]), name=param, **params)
                )
        if len(fields) > 1:
            for param in fields[1]:
                name = param.split(".")[-1]
                fig.add_trace(
                    _scatter(x, np.array(obs_dict[param]), name=name, **params),
                    secondary_y=True,
                )
    return fig
