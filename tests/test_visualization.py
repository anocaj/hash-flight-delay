import pandas as pd
import numpy as np

from src import visualization as viz


def make_sample_df(n=200):
    rng = np.random.RandomState(42)
    dates = pd.date_range('2025-01-01', periods=10)
    df = pd.DataFrame({
        'FL_DATE': rng.choice(dates, size=n),
        'DEP_HOUR': rng.randint(0, 24, size=n),
        'ORIGIN': rng.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW'], size=n),
        'DEST': rng.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW'], size=n),
        'ARR_DEL15': rng.binomial(1, 0.2, size=n),
        'ARR_DELAY': rng.normal(loc=5, scale=20, size=n),
        'TAIL_NUM': rng.choice(['N1', 'N2', 'N3', 'N4'], size=n),
        'ORIGIN_LAT': rng.uniform(30.0, 45.0, size=n),
        'ORIGIN_LON': rng.uniform(-120.0, -70.0, size=n),
    })
    return df


def test_plots_smoke():
    df = make_sample_df(300)

    # missingness
    fig, axes = viz.plot_missingness(df)
    assert fig is not None

    # top airports
    fig, ax = viz.plot_top_airports_by_volume(df, n=5)
    assert fig is not None

    # hour-weekday heatmap
    fig, ax = viz.plot_hour_weekday_heatmap(df, hour_col='DEP_HOUR', date_col='FL_DATE')
    assert fig is not None

    # daily trend
    fig, ax = viz.plot_daily_delay_trend(df)
    assert fig is not None

    # inbound effect
    fig, ax = viz.plot_inbound_delay_effect(df)
    assert fig is not None

    # map (skip if folium missing)
    try:
        m = viz.plot_airport_map(df, top_n=5)
        assert m is not None
    except ImportError:
        # folium not installed in some test envs
        pass
