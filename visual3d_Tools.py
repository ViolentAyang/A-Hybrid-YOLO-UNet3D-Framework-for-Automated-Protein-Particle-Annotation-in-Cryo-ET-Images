#!/usr/bin/env python

import os
import math
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import zarr

try:
    import zarr
except ImportError:
    os.system('pip install zarr')
    import zarr

pio.renderers.default = 'iframe'

runs = sorted(glob('./data/train/overlay/ExperimentRuns/*'))
runs = [os.path.basename(x) for x in runs]

def read_run(run: str) -> pd.DataFrame:
    paths = glob(f"./data/train/overlay/ExperimentRuns/{run}/Picks/*.json")
    df = pd.concat([pd.read_json(x) for x in paths]).reset_index(drop=True)

    for axis in "x", "y", "z":
        df[axis] = df.points.apply(lambda x: x["location"][axis])
    for key in "transformation_", "instance_id":
        df[key] = df.points.apply(lambda x: x[key])
    return df

def plot_particles(
    df: pd.DataFrame, scale: float = 1.0, marker_size: float = 2.0
) -> plotly.graph_objs._figure.Figure:
    df = df.copy()
    df[["x", "y", "z"]] *= scale
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="pickable_object_name")
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        title=df.run_name.iloc[0],
        scene=dict(
            yaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.25, y=-1.25, z=1.25)),
        ),
        width=800,
        height=800,
        template="plotly_dark",
    )
    return fig

def read_zarr(run: str) -> zarr.hierarchy.Group:
    return zarr.open(
        f"./data/train/static/ExperimentRuns/{run}/VoxelSpacing10.000/denoised.zarr",
        mode="r",
    )

def plot_zarr_images(arr: zarr.core.Array, ncols: int = 6, axsize: float = 2.0):
    nslices = len(arr)
    nrows = math.ceil(nslices / ncols)

    fig = plt.figure(figsize=(axsize * ncols, axsize * nrows))
    for i in range(nslices):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(arr[i])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
    plt.show()

def visualize_runs_basic(run_indices=None, slice_index=2):
    if run_indices is None:
        run_indices = range(min(7, len(runs)))
    
    for i in run_indices:
        if i >= len(runs):
            continue
        
        print(f"\n=== Run {i}: {runs[i]} ===")
        df = read_run(runs[i])
        
        plot_particles(df)
        
        zarr_group = read_zarr(runs[i])
        plot_zarr_images(zarr_group[slice_index])

def visualize_single_run(run_index=0, scale=0.1, slice_index=2, zarr_index=0, 
                        show_basic=True, show_animated=True):
    if run_index >= len(runs):
        print(f"Run index {run_index} is out of range. Available runs: 0-{len(runs)-1}")
        return
    
    run = runs[run_index]
    print(f"\n=== Detailed visualization for Run {run_index}: {run} ===")
    
    try:
        df = read_run(run)
        zarr_group = read_zarr(run)
        
        if show_basic:
            print("Basic particle visualization:")
            plot_particles(df)
            print("Zarr slice visualization:")
            plot_zarr_images(zarr_group[slice_index])
        
        if show_animated:
            print("Animated slice visualization:")
            fig = plot_particles(df, scale=scale)
            plot_animatable_slices(zarr_group[zarr_index], title=df.run_name.iloc[0], fig=fig)
            
    except Exception as e:
        print(f"Error processing run {run}: {e}")

def _normalize_array(zarr_array: zarr.core.Array) -> np.ndarray:
    arr = np.array(zarr_array)
    mins = arr.min(axis=(1, 2), keepdims=True)
    maxs = arr.max(axis=(1, 2), keepdims=True)
    arr = ((arr - mins) / (maxs - mins) * 255).astype(np.uint8)
    return arr

def _plot_slice_surface(array: zarr.core.Array, z_index: int):
    return go.Surface(
        z=z_index * np.ones((array.shape[1], array.shape[2])),
        surfacecolor=array[z_index],
        colorscale="gray",
        cmin=0,
        cmax=255,
        showscale=False,
    )

def plot_animatable_slices(
    arr: zarr.core.Array,
    step: int = 10,
    init_z: int = 0,
    title: str = "",
    fig: plotly.graph_objs._figure.Figure | None = None,
) -> plotly.graph_objs._figure.Figure:
    base_traces = list(fig.data) if fig is not None else []

    arr = _normalize_array(arr)
    z_dim = len(arr)

    z_indices = range(0, z_dim, step)

    fig = go.Figure(
        data=base_traces + [_plot_slice_surface(arr, init_z)],
        layout=go.Layout(
            title=title,
            scene=dict(
                yaxis=dict(autorange="reversed"),
                zaxis=dict(range=[0, z_dim], autorange=False),
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(eye=dict(x=1.25, y=-1.25, z=1.25)),
            ),
            width=800,
            height=800,
            template="plotly_dark",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="▶",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=500, redraw=True),
                                    fromcurrent=True,
                                ),
                            ],
                        )
                    ],
                    font=dict(color="black"),
                )
            ],
        ),
    )

    frames = []
    for i in z_indices:
        frame = go.Frame(data=base_traces + [_plot_slice_surface(arr, i)], name=str(i))
        frames.append(frame)
    fig.frames = frames

    sliders = [
        dict(
            active=0,
            currentvalue=dict(prefix="z-index: "),
            pad=dict(t=50),
            steps=[
                dict(
                    label=str(i),
                    method="animate",
                    args=[
                        [str(i)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=200, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                )
                for i in z_indices
            ],
        )
    ]
    fig.update_layout(sliders=sliders)
    return fig

def visualize_runs_animated(run_indices=None, scale=0.1, zarr_index=0):
    if run_indices is None:
        run_indices = range(min(7, len(runs)))
    
    for i in run_indices:
        if i >= len(runs):
            continue
        
        print(f"\n=== Animated visualization for Run {i}: {runs[i]} ===")
        run = runs[i]
        df = read_run(run)
        fig = plot_particles(df, scale=scale)
        zarr_group = read_zarr(run)
        plot_animatable_slices(zarr_group[zarr_index], title=df.run_name.iloc[0], fig=fig)

def main():
    print(f"Found {len(runs)} experimental runs")
    print("Available runs:", [f"{i}: {run}" for i, run in enumerate(runs)])
    
    print("\n" + "="*50)
    print("Running basic visualization for first few runs...")
    visualize_runs_basic(run_indices=range(min(3, len(runs))))
    
    print("\n" + "="*50)
    print("Running animated visualization for first run...")
    if len(runs) > 0:
        visualize_single_run(run_index=0, show_basic=False, show_animated=True)

if __name__ == "__main__":
    main() 