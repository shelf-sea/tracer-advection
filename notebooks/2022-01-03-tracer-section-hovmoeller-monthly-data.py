# %%
import xarray as xr
import cf_xarray as cfxr
import git
from pathlib import Path
import cmasher as cmr
import cmocean as cmo
from matplotlib import pyplot as plt
import numpy as np
from water_masses.filtering import butter_lowpass_filter, butter_bandstop_filter
from water_masses.time_series import detrend
from matplotlib import dates as pldates
from matplotlib import figure as plfigure

import dask
from dask.distributed import Client, LocalCluster


class Project(object):
    def __init__(self) -> None:
        self.root_path = Path(
            git.Repo(Path.cwd(), search_parent_directories=True).git.rev_parse(
                "--show-toplevel",
            )
        )
        self.fig_path = self.root_path.joinpath("figs")
        self.fig_path.mkdir(parents=True, exist_ok=True)

    def savefig(self, figure: plfigure.Figure, name: str) -> None:
        figure.savefig(self.fig_path.joinpath(name), facecolor="white")


proj = Project()
git_root = proj.root_path

# %%
INTERACTIVE = True
QUANTITY = ["s", "t", "chl", "pp", "po4", "no3"][0]
layer_depth = slice(None, 40)

# %%
if not INTERACTIVE:
    cluster = LocalCluster(
        processes=False, n_workers=7, memory_limit="4GB", threads_per_worker=4
    )
    client = Client(cluster)


# %%
def get_quantity(var):
    if var == "t":
        q = {
            "short": "t",
            "long": "thetao",
            "name": "temperature",
            "src": "phy",
            "cmap": {
                "seq": cmr.cm.pride,
                "seq2": cmr.cm.iceburn,
                "div": cmr.cm.fusion_r,
                "cyc": cmr.cm.copper,
            },
        }
    elif var == "s":
        q = {
            "short": "s",
            "long": "so",
            "name": "salinity",
            "src": "phy",
            "cmap": {
                "seq": cmr.cm.eclipse,
                "seq2": cmr.cm.copper_s_r,
                "div": cmr.cm.copper,
                "cyc": cmr.cm.copper,
            },
        }
    elif var == "chl":
        q = {
            "short": "chl",
            "long": "chl",
            "name": "chlorophyl",
            "src": "bgc",
            "cmap": {
                "seq": cmr.cm.copper_s_r,
                "seq2": cmr.cm.copper_s_r,
                "div": cmr.cm.copper,
                "cyc": cmr.cm.copper,
            },
        }
    elif var == "pp":
        q = {
            "short": "pp",
            "long": "nppv",
            "name": "net primary production",
            "src": "bgc",
            "cmap": {
                "seq": cmr.cm.copper_s_r,
                "seq2": cmr.cm.copper_s_r,
                "div": cmr.cm.copper,
                "cyc": cmr.cm.copper,
            },
        }
    elif var == "po4":
        q = {
            "short": "po4",
            "long": "po4",
            "name": "phosphate",
            "src": "bgc",
            "cmap": {
                "seq": cmr.cm.copper_s_r,
                "seq2": cmr.cm.copper_s_r,
                "div": cmr.cm.copper,
                "cyc": cmr.cm.copper,
            },
        }
    elif var == "no3":
        q = {
            "short": "no3",
            "long": "no3",
            "name": "nitrate",
            "src": "bgc",
            "cmap": {
                "seq": cmr.cm.copper_s_r,
                "seq2": cmr.cm.copper_s_r,
                "div": cmr.cm.copper,
                "cyc": cmr.cm.copper,
            },
        }
    else:
        raise NotImplementedError
    return q


quantity = get_quantity(QUANTITY)

# %%
data_path_t = git_root.joinpath(
    "data/external/cmems_mod_nws_{0}-{1}_my_7km-3D_P1M-m".format(
        quantity["src"], quantity["short"]
    )
).glob("????/*")
data_path_t

# %%
ds_t = xr.open_mfdataset(data_path_t, engine="h5netcdf")
ds_t

# %%
profile_list = [
    {"longitude": -18.5, "latitude": 53.5},
    {"longitude": -18.5, "latitude": 55.5},
    {"longitude": -13, "latitude": 55},
    {"longitude": -11.5, "latitude": 56.15},
    {"longitude": -10, "latitude": 57.5},
    {"longitude": -8, "latitude": 58.5},
    {"longitude": -6, "latitude": 59.5},
    {"longitude": -4, "latitude": 59.9},
    {"longitude": -2.1, "latitude": 59.95},
    {"longitude": -1, "latitude": 59.5},
    {"longitude": 0.1, "latitude": 58.7},
    {"longitude": 0.6, "latitude": 57.7},
    {"longitude": 0.9, "latitude": 56.3},
]
profile_list = [
    {"longitude": -17.5, "latitude": 53.5},  # p0
    {"longitude": -15.5, "latitude": 54.25},  # p1
    {"longitude": -15, "latitude": 55.5},  # p2
    {"longitude": -13.5, "latitude": 56.5},  # p3
    {"longitude": -11.7, "latitude": 57.8},  # p4
    {"longitude": -8, "latitude": 59},  # p5
    {"longitude": -6, "latitude": 59.9},  # p6
    {"longitude": -3.8, "latitude": 60.3},  # p7
    {"longitude": -2.4, "latitude": 59.95},  # p8
    {"longitude": -1.9, "latitude": 59.2},  # p9
    {"longitude": -0.7, "latitude": 58.7},  # p10
    {"longitude": 0.4, "latitude": 57.7},  # p11
    {"longitude": 0.9, "latitude": 56.3},  # p12
    {"longitude": 3, "latitude": 55.5},  # p13
]
orkney_shetland = 8
profiles = {f"P{n}": profile_list[n] for n in range(len(profile_list))}
f, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
_ds_presel = ds_t.sel(time="2016-01", method="nearest")[quantity["long"]]
_ds = _ds_presel.sel(depth=layer_depth).mean(dim="depth")
shared_kws = dict(
    cmap=quantity["cmap"]["seq"],
    vmin=np.nanquantile(_ds, 0.10),
    vmax=np.nanquantile(_ds, 0.90),
)
_ds.plot(ax=ax, **shared_kws)

levels = [0, 150, 400]
colors = cmr.take_cmap_colors(
    "cmr.neutral", len(levels), cmap_range=(0.6, 0.9), return_fmt="hex"
)
for isoline_depth, isoline_color in zip(levels, colors):
    _ds_presel.compute().where(_ds_presel > 1, 1).sel(
        depth=isoline_depth, method="nearest"
    ).squeeze().plot.contour(levels=1, colors=isoline_color)
[
    ax.plot(point["longitude"], point["latitude"], marker="x", color="k")
    for point in profiles.values()
]
[
    ax.text(point["longitude"] + 0.3, point["latitude"] + 0.3, s=name, color="k")
    for name, point in profiles.items()
]
proj.savefig(f, f"{QUANTITY.upper()}_overview.png")


# %%
def get_profile(ds, x, y, depth=slice(None, 30)):
    return (
        ds.sel(longitude=x, latitude=y, method="nearest")
        .sel(depth=depth)
        .mean(dim="depth")[quantity["long"]]
    )


# %%
if not INTERACTIVE:
    dask_object_list = [
        get_profile(ds_t, profile["longitude"], profile["latitude"], depth=layer_depth)
        for profile in profiles.values()
    ]
    basis_for_df = client.submit(np.stack, client.scatter(dask_object_list))
else:
    basis_for_df = np.stack(
        [
            get_profile(
                ds_t, profile["longitude"], profile["latitude"], depth=layer_depth
            )
            for profile in profiles.values()
        ]
    )

# %%
xrds = xr.Dataset(
    {
        quantity["name"]: (
            ["time", "station"],
            basis_for_df.result().T if not INTERACTIVE else basis_for_df.T,
        ),
    },
    coords={
        "time": ds_t.time.values,
        "station": range(len(list(profiles.keys()))),
    },
)
xrds

# %%
f, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
shared_kws = dict(
    cmap=quantity["cmap"]["seq2"],
    vmin=np.quantile(xrds[quantity["name"]], 0.01),
    vmax=np.quantile(xrds[quantity["name"]], 0.99),
)
xrds[quantity["name"]].plot.pcolormesh(ax=axs[0], **shared_kws)
xrds[quantity["name"]].plot.contourf(ax=axs[1], **shared_kws, levels=20)
[ax.invert_yaxis() for ax in axs.flat]
[ax.axvline(7.5, color="C6", linewidth=3, linestyle="dashed") for ax in axs.flat]
[ax.yaxis.set_major_locator(pldates.AutoDateLocator(minticks=30)) for ax in axs.flat]
f.tight_layout()
proj.savefig(f, f"{QUANTITY.upper()}_hovm.png")

# %%

filter_kws = {"cutlen": 4, "fs": 12, "order": 5}
xrds[quantity["name"] + " low-pass"] = (
    ("time", "station"),
    butter_lowpass_filter(
        detrend(xrds[quantity["name"]].T, dim="time"), **filter_kws
    ).T,
)

# %%
f, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
vlim = np.min(
    np.abs(
        np.array(
            [
                np.quantile(xrds[quantity["name"] + " low-pass"], 0.01),
                np.quantile(xrds[quantity["name"] + " low-pass"], 0.99),
            ]
        )
    )
)
shared_kws = dict(cmap=quantity["cmap"]["div"], vmin=-vlim, vmax=vlim)
xrds[quantity["name"] + " low-pass"].plot.pcolormesh(ax=axs[0], **shared_kws)
xrds[quantity["name"] + " low-pass"].plot.contourf(ax=axs[1], **shared_kws, levels=20)
[ax.invert_yaxis() for ax in axs.flat]
[ax.axvline(7.5, color="C6", linewidth=3, linestyle="dashed") for ax in axs.flat]
[ax.yaxis.set_major_locator(pldates.AutoDateLocator(minticks=30)) for ax in axs.flat]
f.tight_layout()
proj.savefig(f, f"{QUANTITY.upper()}_hovm_lowpass.png")

# %%
fs = 12  # sample rate
lowcut = (fs - 1.5) / fs  # lowpass cutoff (1/year)
highcut = (fs + 2) / fs  # highpass cutoff (1/year)
pass_order = {"bandstop": 5}  # filter type and order
da = xrds[quantity["name"]].T
xrds[quantity["name"] + " band-stop"] = (
    ("time", "station"),
    butter_bandstop_filter(
        detrend(da, dim="time"),
        *[lowcut, highcut, fs],
        **{"order": pass_order["bandstop"]},
    ).T,
)

# %%
aspect = 1 / 6
figsize = 10
f, axs = plt.subplots(ncols=2, nrows=1, figsize=(figsize, figsize / aspect))
vlim = np.min(
    np.abs(
        np.array(
            [
                np.quantile(xrds[quantity["name"] + " band-stop"], 0.01),
                np.quantile(xrds[quantity["name"] + " band-stop"], 0.99),
            ]
        )
    )
)
shared_kws = dict(cmap=quantity["cmap"]["div"], vmin=-vlim, vmax=vlim)
xrds[quantity["name"] + " band-stop"].plot.pcolormesh(ax=axs[0], **shared_kws)
xrds[quantity["name"] + " band-stop"].plot.contourf(ax=axs[1], **shared_kws, levels=20)
[ax.invert_yaxis() for ax in axs.flat]
[ax.axvline(7.5, color="C6", linewidth=3, linestyle="dashed") for ax in axs.flat]
[ax.yaxis.set_major_locator(pldates.AutoDateLocator(minticks=300)) for ax in axs.flat]
f.tight_layout()
proj.savefig(f, f"{QUANTITY.upper()}_hovm_bandstop.png")

# %%
