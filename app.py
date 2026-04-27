import os
import io
import base64
import warnings
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import linregress
from flask import Flask, render_template, request, Response

warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import rioxarray  # noqa
except Exception:
    rioxarray = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    ccrs = None
    cfeature = None

try:
    import imdlib as imd
except Exception:
    imd = None

app = Flask(__name__)

# =========================================================
# UPDATE ONLY THESE PATHS ACCORDING TO YOUR LAPTOP
# =========================================================
ERA_T_PATH = os.getenv('ERA_T_PATH', r'D:\REEMA\M.Tech\ACADEMICS\SEM_II\Applications_in_Agriculture\DATA\ERA5\ERA5_T_india.nc')
ERA_P_PATH = os.getenv('ERA_P_PATH', r'D:\REEMA\M.Tech\ACADEMICS\SEM_II\Applications_in_Agriculture\DATA\ERA5\ERA5_Precip_india.nc')
ERA_SM_PATH = os.getenv('ERA_SM_PATH', r'D:\REEMA\M.Tech\ACADEMICS\SEM_II\Applications_in_Agriculture\DATA\ERA5\ERA5_SM1_india.nc')

IMD_TMAX_DIR = os.getenv('IMD_TMAX_DIR', r'D:\REEMA\M.Tech\ACADEMICS\SEM_II\Applications_in_Agriculture\DATA\imd_data\tmax')
IMD_TMIN_DIR = os.getenv('IMD_TMIN_DIR', r'D:\REEMA\M.Tech\ACADEMICS\SEM_II\Applications_in_Agriculture\DATA\imd_data\tmin')
IMD_RAIN_DIR = os.getenv('IMD_RAIN_DIR', r'D:\REEMA\M.Tech\ACADEMICS\SEM_II\Applications_in_Agriculture\DATA\imd_data\rain')

INDIA_SHP_PATH = os.getenv('INDIA_SHP_PATH', r'D:\REEMA\M.Tech\ACADEMICS\SEM_II\India_boundary\India_State_Boundary.shp')

CACHE = {'era': None, 'imd': None, 'india': None}
INDIA_EXTENT = [68, 97, 6, 37]
SEASONS = {
    'Pre-monsoon': [3, 4, 5],
    'During monsoon': [6, 7, 8, 9],
    'Post-monsoon': [10, 11, 12]
}
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def month_numbers_and_labels(da):
    """Return only available month numbers and labels to avoid x-y length errors for partial date ranges."""
    months = np.asarray(da['month'].values, dtype=int)
    labels = [MONTH_LABELS[m - 1] for m in months]
    return months, labels

# fixed colour-scale style from reference notebook
TEMP_BINS = np.arange(-20, 41, 5)
PRECIP_BINS = np.linspace(0, 200, 13)
SM_BINS = np.arange(0.00, 0.61, 0.05)
CORR_BINS = np.linspace(-1, 1, 11)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    buf.seek(0)
    text = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return text


def coord_name(ds, choices):
    for c in choices:
        if c in ds.coords or c in ds.dims:
            return c
    raise ValueError(f'Coordinate not found from {choices}. Available: {list(ds.coords)}')


def var_name(ds, choices):
    for v in choices:
        if v in ds.data_vars:
            return v
    raise ValueError(f'Variable not found from {choices}. Available: {list(ds.data_vars)}')


def get_time_name(ds):
    return coord_name(ds, ['time', 'valid_time', 'date'])


def load_india():
    if CACHE['india'] is not None:
        return CACHE['india']
    if gpd is None or not os.path.exists(INDIA_SHP_PATH):
        CACHE['india'] = None
        return None
    india = gpd.read_file(INDIA_SHP_PATH).to_crs('EPSG:4326')
    CACHE['india'] = india
    return india


def region_name(lon, lat):
    india = load_india()
    if india is None or gpd is None or not np.isfinite(lon) or not np.isfinite(lat):
        return 'nearest grid area in India'
    try:
        pt = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs='EPSG:4326')
        hit = gpd.sjoin(pt, india, how='left', predicate='within')
        row = hit.iloc[0]
        for c in ['State_Name', 'STATE', 'state', 'ST_NM', 'NAME_1', 'Name', 'name']:
            if c in row and str(row[c]) != 'nan':
                return str(row[c])
    except Exception:
        pass
    return 'nearest grid area in India'


def standardize_lat_order(da, lat='latitude'):
    if lat in da.coords and da[lat].values[0] > da[lat].values[-1]:
        da = da.sortby(lat)
    return da


def india_clip(da, lon='longitude', lat='latitude'):
    """Clip to India boundary if rioxarray + shapefile are available; otherwise bbox clip."""
    da = standardize_lat_order(da, lat)
    if lon in da.coords:
        da = da.sel({lon: slice(INDIA_EXTENT[0], INDIA_EXTENT[1])})
    if lat in da.coords:
        da = da.sel({lat: slice(INDIA_EXTENT[2], INDIA_EXTENT[3])})

    india = load_india()
    if india is not None and rioxarray is not None and lon in da.coords and lat in da.coords:
        try:
            da = da.rio.set_spatial_dims(x_dim=lon, y_dim=lat, inplace=False).rio.write_crs('EPSG:4326', inplace=False)
            da = da.rio.clip(india.geometry, india.crs, drop=True)
        except Exception:
            pass
    return da


def map_axis(ax):
    india = load_india()
    if ccrs is not None:
        ax.set_extent(INDIA_EXTENT, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        except Exception:
            pass
        if india is not None:
            india.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
        except Exception:
            pass
    else:
        ax.set_xlim(INDIA_EXTENT[0], INDIA_EXTENT[1])
        ax.set_ylim(INDIA_EXTENT[2], INDIA_EXTENT[3])
        if india is not None:
            india.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')


def discrete_map(ax, da, title, bins, cmap_name, label, lon='longitude', lat='latitude'):
    da = india_clip(da, lon, lat)
    cmap = plt.get_cmap(cmap_name, len(bins) - 1)
    norm = colors.BoundaryNorm(bins, ncolors=cmap.N, clip=True)
    x = da[lon]
    y = da[lat]
    if ccrs is not None:
        mesh = ax.pcolormesh(x, y, da.values, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    else:
        mesh = ax.pcolormesh(x, y, da.values, cmap=cmap, norm=norm, shading='auto')
    map_axis(ax)
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.82, ticks=bins)
    if bins is SM_BINS:
        cb.ax.set_xticklabels([f'{v:.2f}' for v in bins])
    elif bins is PRECIP_BINS:
        cb.ax.set_xticklabels([f'{v:.0f}' for v in bins])
    cb.set_label(label)
    return mesh


def continuous_map(ax, da, title, cmap, label, vmin=None, vmax=None, lon='longitude', lat='latitude'):
    da = india_clip(da, lon, lat)
    if ccrs is not None:
        mesh = ax.pcolormesh(da[lon], da[lat], da.values, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', transform=ccrs.PlateCarree())
    else:
        mesh = ax.pcolormesh(da[lon], da[lat], da.values, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    map_axis(ax)
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.82)
    cb.set_label(label)
    return mesh


def trend_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0, 0.0, 0.0, 1.0
    slope, intercept, r, p, _ = linregress(x[mask], y[mask])
    return float(slope), float(intercept), float(r), float(p)


def safe_mean(v):
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def count_events_1d(arr):
    arr = np.asarray(arr, dtype=int)
    if arr.size == 0:
        return 0
    return int((arr[0] == 1) + np.sum((arr[1:] == 1) & (arr[:-1] == 0)))


def smooth_series(y, window=3):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')

# =========================================================
# ERA5 DATA + PLOTS
# =========================================================
def load_era():
    if CACHE['era'] is not None:
        return CACHE['era']
    missing = [p for p in [ERA_T_PATH, ERA_P_PATH, ERA_SM_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('ERA5 file path is wrong:\n' + '\n'.join(missing))

    tds = xr.open_dataset(ERA_T_PATH)
    pds = xr.open_dataset(ERA_P_PATH)
    smds = xr.open_dataset(ERA_SM_PATH)
    data = {
        'tds': tds, 'pds': pds, 'smds': smds,
        'time_t': get_time_name(tds), 'time_p': get_time_name(pds), 'time_sm': get_time_name(smds),
        'lat_t': coord_name(tds, ['latitude', 'lat']), 'lon_t': coord_name(tds, ['longitude', 'lon']),
        'lat_p': coord_name(pds, ['latitude', 'lat']), 'lon_p': coord_name(pds, ['longitude', 'lon']),
        'lat_sm': coord_name(smds, ['latitude', 'lat']), 'lon_sm': coord_name(smds, ['longitude', 'lon']),
        'tv': var_name(tds, ['t2m', 'T2M', 'temp', 'temperature']),
        'pv': var_name(pds, ['tp', 'TP', 'precip', 'precipitation', 'rainfall']),
        'smv': var_name(smds, ['swvl1', 'SM', 'sm', 'soil_moisture'])
    }
    CACHE['era'] = data
    return data


def era_subset(start, end):
    d = load_era()
    t = d['tds'].sel({d['time_t']: slice(start, end)})[d['tv']] - 273.15
    p = d['pds'].sel({d['time_p']: slice(start, end)})[d['pv']] * 1000
    sm = d['smds'].sel({d['time_sm']: slice(start, end)})[d['smv']]
    t = t.rename({d['lat_t']: 'latitude', d['lon_t']: 'longitude'}) if d['lat_t'] != 'latitude' or d['lon_t'] != 'longitude' else t
    p = p.rename({d['lat_p']: 'latitude', d['lon_p']: 'longitude'}) if d['lat_p'] != 'latitude' or d['lon_p'] != 'longitude' else p
    sm = sm.rename({d['lat_sm']: 'latitude', d['lon_sm']: 'longitude'}) if d['lat_sm'] != 'latitude' or d['lon_sm'] != 'longitude' else sm
    # Keep every ERA5 plot restricted to Indian bbox/boundary.
    t = india_clip(t, 'longitude', 'latitude')
    p = india_clip(p, 'longitude', 'latitude')
    sm = india_clip(sm, 'longitude', 'latitude')
    if t.sizes.get('time', t.sizes.get(d['time_t'], 0)) == 0:
        raise ValueError('No ERA5 data found for selected date range.')
    return t, p, sm


def era_annual_plot(start, end):
    t, p, sm = era_subset(start, end)
    t_map = t.groupby('time.year').mean('time').mean('year')
    p_map = p.groupby('time.year').mean('time').mean('year')
    sm_map = sm.groupby('time.year').mean('time').mean('year')
    proj = {'projection': ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw=proj)
    discrete_map(axes[0], t_map, 'Annual Mean Temperature (°C)', TEMP_BINS, 'coolwarm', '°C')
    discrete_map(axes[1], p_map, 'Annual Mean Precipitation (mm)', PRECIP_BINS, 'YlGnBu', 'mm')
    discrete_map(axes[2], sm_map, 'Annual Mean Soil Moisture (m³/m³)', SM_BINS, 'PuBuGn', 'm³/m³')
    fig.suptitle(f'ERA5 Annual Spatial Mean ({start} to {end})', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def era_seasonal_cycle_plot(start, end):
    t, p, sm = era_subset(start, end)
    t_s = t.groupby('time.month').mean('time').mean(dim=['latitude', 'longitude'])
    p_s = p.groupby('time.month').mean('time').mean(dim=['latitude', 'longitude'])
    sm_s = sm.groupby('time.month').mean('time').mean(dim=['latitude', 'longitude'])
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    plot_items = [
        (axes[0], t_s, 'red', 'o', 'Climatological Seasonal Cycle of Temperature (°C)', 'Temperature (°C)'),
        (axes[1], p_s, 'blue', 's', 'Climatological Seasonal Cycle of Precipitation (mm)', 'Precipitation (mm)'),
        (axes[2], sm_s, 'green', '^', 'Climatological Seasonal Cycle of Soil Moisture (m³/m³)', 'Soil Moisture (m³/m³)')
    ]
    for ax, da, color, marker, title, ylabel in plot_items:
        months, labels = month_numbers_and_labels(da)
        ax.plot(months, da.values, color=color, marker=marker)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel('Month')
    fig.suptitle('ERA5 Seasonal Cycle', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def era_jjas_plot(start, end):
    t, p, sm = era_subset(start, end)
    jjas = [6, 7, 8, 9]
    t_map = t.sel(time=t['time.month'].isin(jjas)).mean('time')
    p_map = p.sel(time=p['time.month'].isin(jjas)).groupby('time.year').sum('time').mean('year')
    sm_map = sm.sel(time=sm['time.month'].isin(jjas)).mean('time')
    proj = {'projection': ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw=proj)
    discrete_map(axes[0], t_map, 'JJAS Mean Temperature (°C)', TEMP_BINS, 'coolwarm', '°C')
    discrete_map(axes[1], p_map, 'JJAS Total Precipitation (mm/season)', PRECIP_BINS, 'YlGnBu', 'mm/season')
    discrete_map(axes[2], sm_map, 'JJAS Mean Soil Moisture (m³/m³)', SM_BINS, 'PuBuGn', 'm³/m³')
    fig.suptitle('ERA5 JJAS Monsoon Spatial Pattern', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def era_trend_plot(start, end):
    t, p, sm = era_subset(start, end)
    t_ts = t.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    p_ts = p.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    sm_ts = sm.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    items = [('Annual Temperature Trend', t_ts, 'Temperature (°C)', 'o'), ('Annual Precipitation Trend', p_ts, 'Precipitation (mm)', 's'), ('Annual Soil Moisture Trend', sm_ts, 'Soil Moisture (m³/m³)', '^')]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    for ax, (title, da, ylabel, marker) in zip(axes, items):
        years = da['year'].values
        vals = da.values
        slope, intercept, _, _ = trend_stats(years, vals)
        ax.plot(years, vals, marker=marker, label=ylabel)
        ax.plot(years, slope * years + intercept, linestyle='--', label=f'Trend (slope={slope:.4f})')
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('ERA5 Trend Lines', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def era_correlation_plot(start, end):
    t, p, sm = era_subset(start, end)
    proj = {'projection': ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw=proj)
    plots = []
    for name, months in SEASONS.items():
        tt = t.sel(time=t['time.month'].isin(months))
        pp = p.sel(time=p['time.month'].isin(months))
        ss = sm.sel(time=sm['time.month'].isin(months))
        plots.append((xr.corr(tt, ss, dim='time'), f'T vs SM {name}'))
    for name, months in SEASONS.items():
        pp = p.sel(time=p['time.month'].isin(months))
        ss = sm.sel(time=sm['time.month'].isin(months))
        plots.append((xr.corr(pp, ss, dim='time'), f'P vs SM {name}'))
    for ax, (da, title) in zip(axes.flat, plots):
        discrete_map(ax, da, title, CORR_BINS, 'coolwarm', 'Correlation r')
    fig.suptitle('ERA5 Seasonal Correlation Plot', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def era_extreme_histogram_plot(start, end):
    t, p, sm = era_subset(start, end)
    # reference-style histogram/bar: extreme months vs extreme events by season
    vals = []
    labels = list(SEASONS.keys())
    for months in SEASONS.values():
        tt = t.sel(time=t['time.month'].isin(months))
        ts = tt.mean(dim=['latitude', 'longitude'])
        thr = float(ts.quantile(0.90).values)
        extreme = (ts > thr).astype(int).values
        vals.append((count_events_1d(extreme), int(extreme.sum())))
    event_vals = [v[0] for v in vals]
    month_vals = [v[1] for v in vals]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, event_vals, width=0.4)
    axes[0].set_title('(a) Extreme Temperature Events', fontsize=10)
    axes[0].set_ylabel('Number of extreme events', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[1].bar(labels, month_vals, width=0.4)
    axes[1].set_title('(b) Extreme Temperature Months', fontsize=10)
    axes[1].set_ylabel('Number of extreme months', fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=9, rotation=10)
        ax.tick_params(axis='y', labelsize=9)
    fig.suptitle('ERA5 Extreme Months vs Extreme Events', fontsize=13, fontweight='bold')
    return fig_to_base64(fig)


def era_extreme_trend_plot(start, end):
    t, p, sm = era_subset(start, end)
    series = [
        ('Temperature Extremes', t.mean(dim=['latitude', 'longitude']), 'Extreme temperature months', 'Extreme temperature frequency', 'high'),
        ('Rainfall Extremes', p.mean(dim=['latitude', 'longitude']), 'Extreme rainfall months', 'Extreme rainfall frequency', 'high'),
        ('Soil Moisture Extremes', sm.mean(dim=['latitude', 'longitude']), 'Extreme soil moisture months', 'Extreme soil moisture frequency', 'low'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for row, (main_title, data, ylab_m, ylab_f, kind) in enumerate(series):
        thr = data.quantile(0.90) if kind == 'high' else data.quantile(0.10)
        extreme = (data > thr) if kind == 'high' else (data < thr)
        ext_months = extreme.astype(int).groupby('time.year').sum()
        years = ext_months['year'].values
        y1 = ext_months.values.astype(float)
        y2 = []
        for yr in years:
            arr = extreme.sel(time=extreme['time.year'] == yr).astype(int).values
            y2.append(count_events_1d(arr))
        y2 = np.asarray(y2, dtype=float)
        for col, (vals, ylabel) in enumerate([(y1, ylab_m), (y2, ylab_f)]):
            ax = axes[row, col]
            slope, intercept, _, pval = trend_stats(years, vals)
            ax.plot(years, smooth_series(vals), color='purple', linewidth=2.2)
            ax.plot(years, slope * years + intercept, color='green', linewidth=1.4)
            ax.set_title(f"({'a' if col == 0 else 'b'}) {main_title}", loc='left', fontweight='bold', fontsize=12)
            ax.set_xlabel('Year')
            ax.set_ylabel(ylabel)
            ax.text(0.05, 0.90, f'Trend = {slope*10:.2f}/decade (p = {pval:.2f})', transform=ax.transAxes, color='red', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2.5))
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.grid(False)
    fig.suptitle('ERA5 Extreme Trend Plot', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def era_metrics(start, end):
    t, p, sm = era_subset(start, end)
    t_ann = t.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    p_ann = p.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    sm_ann = sm.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    years = t_ann['year'].values
    ts, _, _, _ = trend_stats(years, t_ann.values)
    ps, _, _, _ = trend_stats(years, p_ann.values)
    sms, _, _, _ = trend_stats(years, sm_ann.values)
    p20 = float(p_ann.quantile(0.20).values)
    sm20 = float(sm_ann.quantile(0.20).values)
    dry_years = int(((p_ann < p20) & (sm_ann < sm20)).sum().values)
    low_map = p.mean('time') + sm.mean('time')
    low_map = india_clip(low_map)
    try:
        idx = np.unravel_index(np.nanargmin(low_map.values), low_map.shape)
        lat = float(low_map.latitude.values[idx[0]])
        lon = float(low_map.longitude.values[idx[1]])
    except Exception:
        lat, lon = np.nan, np.nan
    area = region_name(lon, lat)
    alert = dry_years >= 1 or (ps < 0 and sms < 0)
    return {
        'mean_temp': round(safe_mean(t_ann.values), 2),
        'mean_rain': round(safe_mean(p_ann.values), 2),
        'mean_sm': round(safe_mean(sm_ann.values), 3),
        'temp_slope': round(ts, 4), 'rain_slope': round(ps, 4), 'sm_slope': round(sms, 5),
        'dry_months': dry_years,
        'drought_status': 'Possible meteorological drought signal observed' if alert else 'No strong meteorological drought signal',
        'drought_class': 'alert' if alert else 'safe',
        'hotspot_lat': round(lat, 2) if np.isfinite(lat) else 'NA',
        'hotspot_lon': round(lon, 2) if np.isfinite(lon) else 'NA',
        'hotspot_area': area,
        'conclusion': f'Combined rainfall and soil-moisture stress is checked for the selected period. Watch grid is near {area} ({lat:.2f}, {lon:.2f}).' if np.isfinite(lat) else 'Selected ERA5 values were processed for India only.',
        'policy': 'Use this output with local rainfall reports, irrigation demand, reservoir level and crop condition before final drought declaration.'
    }

# =========================================================
# IMD DATA + PLOTS
# =========================================================
def load_imd():
    if CACHE['imd'] is not None:
        return CACHE['imd']
    if imd is None:
        raise RuntimeError('imdlib is not installed. Install it using: pip install imdlib')
    missing = [p for p in [IMD_TMAX_DIR, IMD_TMIN_DIR, IMD_RAIN_DIR] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('IMD folder path is wrong:\n' + '\n'.join(missing))
    tmax = imd.open_data('tmax', 1991, 2023, 'yearwise', IMD_TMAX_DIR).get_xarray()['tmax']
    tmin = imd.open_data('tmin', 1991, 2023, 'yearwise', IMD_TMIN_DIR).get_xarray()['tmin']
    rain = imd.open_data('rain', 1991, 2023, 'yearwise', IMD_RAIN_DIR).get_xarray()['rain']
    # IMD grid files may carry negative fill values. Mask them before monthly aggregation.
    tmax = tmax.where((tmax > -50) & (tmax < 60))
    tmin = tmin.where((tmin > -50) & (tmin < 60))
    rain = rain.where(rain >= 0)
    temp = ((tmax + tmin) / 2).resample(time='MS').mean()
    rain_m = rain.resample(time='MS').sum()
    CACHE['imd'] = {'temp': temp, 'rain': rain_m}
    return CACHE['imd']


def imd_subset(start, end):
    d = load_imd()
    t = d['temp'].sel(time=slice(start, end))
    r = d['rain'].sel(time=slice(start, end))
    if 'lat' in t.dims:
        t = t.rename({'lat': 'latitude', 'lon': 'longitude'})
    if 'lat' in r.dims:
        r = r.rename({'lat': 'latitude', 'lon': 'longitude'})
    r = r.where(r >= 0)
    t = india_clip(t, 'longitude', 'latitude')
    r = india_clip(r, 'longitude', 'latitude')
    if t.sizes.get('time', 0) == 0:
        raise ValueError('No IMD data found for selected date range.')
    return t, r


def imd_annual_plot(start, end):
    t, r = imd_subset(start, end)
    t_map = t.groupby('time.year').mean('time').mean('year')
    r_map = r.groupby('time.year').sum('time').mean('year')
    proj = {'projection': ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), subplot_kw=proj)
    discrete_map(axes[0], t_map, 'IMD Annual Mean Temperature (°C)', TEMP_BINS, 'coolwarm', '°C')
    continuous_map(axes[1], r_map, 'IMD Annual Rainfall (mm/year)', 'YlGnBu', 'mm/year', vmin=0)
    fig.suptitle(f'IMD Annual Spatial Mean ({start} to {end})', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def imd_seasonal_cycle_plot(start, end):
    t, r = imd_subset(start, end)
    t_s = t.groupby('time.month').mean('time').mean(dim=['latitude', 'longitude'])
    r_s = r.groupby('time.month').mean('time').mean(dim=['latitude', 'longitude'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=False)
    plot_items = [
        (axes[0], t_s, 'red', 'o', 'IMD Seasonal Cycle of Temperature (°C)', 'Temperature (°C)'),
        (axes[1], r_s, 'blue', 's', 'IMD Seasonal Cycle of Rainfall (mm/month)', 'Rainfall (mm/month)')
    ]
    for ax, da, color, marker, title, ylabel in plot_items:
        months, labels = month_numbers_and_labels(da)
        ax.plot(months, da.values, color=color, marker=marker, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel('Month')
    fig.suptitle('IMD Seasonal Cycle', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def imd_jjas_plot(start, end):
    t, r = imd_subset(start, end)
    jjas = [6, 7, 8, 9]
    t_map = t.sel(time=t['time.month'].isin(jjas)).mean('time')
    r_map = r.sel(time=r['time.month'].isin(jjas)).groupby('time.year').sum('time').mean('year')
    proj = {'projection': ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), subplot_kw=proj)
    discrete_map(axes[0], t_map, 'IMD JJAS Mean Temperature (°C)', TEMP_BINS, 'coolwarm', '°C')
    continuous_map(axes[1], r_map, 'IMD JJAS Rainfall (mm/season)', 'YlGnBu', 'mm/season', vmin=0)
    fig.suptitle('IMD JJAS Monsoon Spatial Pattern', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def imd_trend_plot(start, end):
    t, r = imd_subset(start, end)
    t_ts = t.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    r_ts = r.groupby('time.year').sum('time').mean(dim=['latitude', 'longitude'])
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    for ax, da, title, ylabel, marker, clr in [
        (axes[0], t_ts, 'IMD Annual Temperature Trend', 'Temperature (°C)', 'o', 'red'),
        (axes[1], r_ts, 'IMD Annual Rainfall Trend', 'Rainfall (mm/year)', 's', 'blue')]:
        years = da['year'].values
        vals = da.values
        slope, intercept, _, _ = trend_stats(years, vals)
        ax.plot(years, vals, color=clr, marker=marker, linewidth=1.8, label='Annual value')
        ax.plot(years, slope * years + intercept, color='orange', linestyle='--', label=f'Trend (slope={slope:.4f})')
        ax.set_xlabel('Year'); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('IMD Trend Lines', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def imd_correlation_plot(start, end):
    t, r = imd_subset(start, end)
    proj = {'projection': ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), subplot_kw=proj)
    for ax, (name, months) in zip(axes, SEASONS.items()):
        corr = xr.corr(t.sel(time=t['time.month'].isin(months)), r.sel(time=r['time.month'].isin(months)), dim='time')
        discrete_map(ax, corr, f'IMD Temperature vs Rainfall: {name}', CORR_BINS, 'coolwarm', 'Correlation r')
    fig.suptitle('IMD Seasonal Correlation Plot', fontsize=14, fontweight='bold')
    return fig_to_base64(fig)


def imd_extreme_histogram_plot(start, end):
    t, r = imd_subset(start, end)
    labels = list(SEASONS.keys())
    hot_events, hot_months, rain_events, rain_months = [], [], [], []
    for months in SEASONS.values():
        ts = t.sel(time=t['time.month'].isin(months)).mean(dim=['latitude', 'longitude'])
        rs = r.sel(time=r['time.month'].isin(months)).mean(dim=['latitude', 'longitude'])
        hot_extreme = (ts > float(ts.quantile(0.90).values)).astype(int).values
        rain_extreme = (rs > float(rs.quantile(0.90).values)).astype(int).values
        hot_events.append(count_events_1d(hot_extreme)); hot_months.append(int(np.nansum(hot_extreme)))
        rain_events.append(count_events_1d(rain_extreme)); rain_months.append(int(np.nansum(rain_extreme)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, vals, title, ylabel, clr in [
        (axes[0,0], hot_events, '(a) Extreme Temperature Events', 'Number of events', 'tomato'),
        (axes[0,1], hot_months, '(b) Extreme Temperature Months', 'Number of months', 'red'),
        (axes[1,0], rain_events, '(c) Extreme Rainfall Events', 'Number of events', 'royalblue'),
        (axes[1,1], rain_months, '(d) Extreme Rainfall Months', 'Number of months', 'blue')]:
        ax.bar(labels, vals, width=0.45, color=clr, alpha=0.85)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', labelsize=9, rotation=10); ax.tick_params(axis='y', labelsize=9)
    fig.suptitle('IMD Extreme Months vs Extreme Events', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_base64(fig)


def imd_extreme_trend_plot(start, end):
    t, r = imd_subset(start, end)
    series = [
        ('Temperature Extremes', t.mean(dim=['latitude', 'longitude']), 'Extreme temperature months', 'Extreme temperature events', 'high'),
        ('Rainfall Extremes', r.mean(dim=['latitude', 'longitude']), 'Extreme rainfall months', 'Extreme rainfall events', 'high'),
        ('Dry Rainfall Extremes', r.mean(dim=['latitude', 'longitude']), 'Low rainfall months', 'Low rainfall events', 'low')]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for row, (main_title, data, ylab_m, ylab_f, kind) in enumerate(series):
        thr = data.quantile(0.90) if kind == 'high' else data.quantile(0.10)
        extreme = (data > thr) if kind == 'high' else (data < thr)
        ext_months = extreme.astype(int).groupby('time.year').sum()
        years = ext_months['year'].values
        y1 = ext_months.values.astype(float)
        y2 = np.asarray([count_events_1d(extreme.sel(time=extreme['time.year'] == yr).astype(int).values) for yr in years], dtype=float)
        for col, (vals, ylabel) in enumerate([(y1, ylab_m), (y2, ylab_f)]):
            ax = axes[row, col]
            slope, intercept, _, pval = trend_stats(years, vals)
            ax.plot(years, smooth_series(vals), color='purple', linewidth=2.2)
            ax.plot(years, slope * years + intercept, color='green', linewidth=1.4)
            ax.set_title(f"({'a' if col == 0 else 'b'}) {main_title}", loc='left', fontweight='bold', fontsize=12)
            ax.set_xlabel('Year'); ax.set_ylabel(ylabel)
            ax.text(0.05, 0.90, f'Trend = {slope*10:.2f}/decade (p = {pval:.2f})', transform=ax.transAxes, color='red', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2.5))
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)
    fig.suptitle('IMD Extreme Trend Plot', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig_to_base64(fig)


def imd_metrics(start, end):
    t, r = imd_subset(start, end)
    t_ann = t.groupby('time.year').mean('time').mean(dim=['latitude', 'longitude'])
    r_ann = r.groupby('time.year').sum('time').mean(dim=['latitude', 'longitude'])
    years = t_ann['year'].values
    ts, _, _, _ = trend_stats(years, t_ann.values)
    rs, _, _, _ = trend_stats(years, r_ann.values)
    normal = float(r_ann.mean().values)
    latest = float(r_ann.values[-1])
    deficit = ((latest - normal) / normal) * 100 if normal else 0
    rain_map = india_clip(r.groupby('time.year').sum('time').mean('year'))
    try:
        idx = np.unravel_index(np.nanargmin(rain_map.values), rain_map.shape)
        lat = float(rain_map.latitude.values[idx[0]]); lon = float(rain_map.longitude.values[idx[1]])
    except Exception:
        lat, lon = np.nan, np.nan
    area = region_name(lon, lat)
    dry_years = int((r_ann < 0.8 * normal).sum().values) if normal else 0
    alert = dry_years > 0 or deficit <= -20
    return {
        'mean_temp': round(safe_mean(t_ann.values), 2),
        'mean_rain': round(safe_mean(r_ann.values), 2),
        'rainfall_normal': round(normal, 2), 'latest_rain': round(latest, 2), 'rain_deficit': round(deficit, 2), 'dry_years': dry_years,
        'temp_slope': round(ts, 4), 'rain_slope': round(rs, 4),
        'drought_status': 'Meteorological drought condition possible' if alert else 'No strong meteorological drought condition',
        'drought_class': 'alert' if alert else 'safe',
        'hotspot_lat': round(lat, 2) if np.isfinite(lat) else 'NA', 'hotspot_lon': round(lon, 2) if np.isfinite(lon) else 'NA', 'hotspot_area': area,
        'conclusion': f'IMD rainfall deficit is checked for the selected period using annual rainfall totals. Lowest rainfall grid is near {area} ({lat:.2f}, {lon:.2f}).' if np.isfinite(lat) else 'Selected IMD values were processed for India only.',
        'policy': 'Use rainfall-deficit output with district observations, crop-water demand and official drought reports before final decision.'
    }


# =========================================================
# FLASK ROUTES
# =========================================================
def empty_context():
    return {
        'start_date': '1991-01-01', 'end_date': '2023-12-31',
        'show_era5': True, 'show_imd': True,
        'error_message': None,
        'era5': {}, 'imd': {}, 'era5_metrics': None, 'imd_metrics': None
    }




@app.route('/india_boundary', methods=['GET'])
def india_boundary():
    """Serve India boundary from the same shapefile path used in analysis maps.
    Output is EPSG:4326 GeoJSON, which matches Leaflet/OSM.
    """
    if gpd is None:
        return Response('{"error":"geopandas is not installed"}', status=500, mimetype='application/json')
    if not os.path.exists(INDIA_SHP_PATH):
        return Response('{"error":"India shapefile path is not available. Please update INDIA_SHP_PATH in app.py."}', status=404, mimetype='application/json')
    try:
        india = gpd.read_file(INDIA_SHP_PATH).to_crs('EPSG:4326')
        india = india[india.geometry.notna()].copy()
        try:
            geom = india.geometry.union_all()
        except Exception:
            geom = india.unary_union
        boundary_gdf = gpd.GeoDataFrame({'name': ['India']}, geometry=[geom], crs='EPSG:4326')
        return Response(boundary_gdf.to_json(), mimetype='application/json')
    except Exception as exc:
        msg = str(exc).replace('"', "'")
        return Response('{"error":"Unable to load India boundary: %s"}' % msg, status=500, mimetype='application/json')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', **empty_context())


@app.route('/update_dashboard', methods=['POST'])
def update_dashboard():
    start = request.form.get('start_date', '1991-01-01')
    end = request.form.get('end_date', '2023-12-31')
    show_era5 = request.form.get('show_era5') == 'on'
    show_imd = request.form.get('show_imd') == 'on'
    ctx = empty_context()
    ctx.update({'start_date': start, 'end_date': end, 'show_era5': show_era5, 'show_imd': show_imd})
    errors = []

    if show_era5:
        try:
            ctx['era5'] = {
                'annual': era_annual_plot(start, end),
                'seasonal_cycle': era_seasonal_cycle_plot(start, end),
                'jjas': era_jjas_plot(start, end),
                'trend': era_trend_plot(start, end),
                'corr': era_correlation_plot(start, end),
                'extreme_hist': era_extreme_histogram_plot(start, end),
                'extreme_trend': era_extreme_trend_plot(start, end),
            }
            ctx['era5_metrics'] = era_metrics(start, end)
        except Exception as e:
            errors.append('ERA5: ' + str(e))

    if show_imd:
        try:
            ctx['imd'] = {
                'annual': imd_annual_plot(start, end),
                'seasonal_cycle': imd_seasonal_cycle_plot(start, end),
                'jjas': imd_jjas_plot(start, end),
                'trend': imd_trend_plot(start, end),
                'corr': imd_correlation_plot(start, end),
                'extreme_hist': imd_extreme_histogram_plot(start, end),
                'extreme_trend': imd_extreme_trend_plot(start, end),
            }
            ctx['imd_metrics'] = imd_metrics(start, end)
        except Exception as e:
            errors.append('IMD: ' + str(e))

    if not show_era5 and not show_imd:
        errors.append('Please select at least one dataset: ERA5 or IMD.')

    ctx['error_message'] = ' | '.join(errors) if errors else None
    return render_template('index.html', **ctx)


if __name__ == '__main__':
    app.run(debug=True)
