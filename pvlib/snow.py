"""
The ``snow`` module contains functions that model module snow cover and the
associated effects on PV module output
"""

import numpy as np
import pandas as pd
from pvlib.tools import sind, cosd, tand


def _time_delta_in_hours(times):
    delta = times.to_series().diff()
    return delta.dt.total_seconds().div(3600)


def fully_covered_nrel(snowfall, threshold_snowfall=1.):
    '''
    Calculates the timesteps when the row's slant height is fully covered
    by snow.
    ... (docstring remains unchanged)
    '''
    timestep = _time_delta_in_hours(snowfall.index)
    hourly_snow_rate = snowfall / timestep
    freq = pd.infer_freq(snowfall.index)
    if freq is not None:
        timedelta = pd.tseries.frequencies.to_offset(freq) / pd.Timedelta('1h')
        hourly_snow_rate.iloc[0] = snowfall[0] / timedelta
    else:
        hourly_snow_rate[0] = 0
    return hourly_snow_rate > threshold_snowfall


def coverage_nrel(snowfall, poa_irradiance, temp_air, surface_tilt,
                  initial_coverage=0, threshold_snowfall=1.,
                  can_slide_coefficient=-80., slide_amount_coefficient=0.197):
    '''
    Calculates the fraction of the slant height of a row of modules covered by
    snow at every time step.
    ... (docstring remains unchanged)
    '''
    new_snowfall = fully_covered_nrel(snowfall, threshold_snowfall)
    snow_coverage = pd.Series(np.nan, index=poa_irradiance.index)
    can_slide = temp_air > poa_irradiance / can_slide_coefficient
    slide_amt = slide_amount_coefficient * sind(surface_tilt) * \
        _time_delta_in_hours(poa_irradiance.index)
    slide_amt[~can_slide] = 0.
    slide_amt[new_snowfall] = 0.
    slide_amt.iloc[0] = 0
    sliding_period_ID = new_snowfall.cumsum()
    cumulative_sliding = slide_amt.groupby(sliding_period_ID).cumsum()
    snow_coverage[new_snowfall] = 1.0
    if np.isnan(snow_coverage.iloc[0]):
        snow_coverage.iloc[0] = initial_coverage
    snow_coverage.ffill(inplace=True)
    snow_coverage -= cumulative_sliding
    return snow_coverage.clip(lower=0)


def dc_loss_nrel(snow_coverage, num_strings):
    '''
    Calculates the fraction of DC capacity lost due to snow coverage.
    ... (docstring remains unchanged)
    '''
    return np.ceil(snow_coverage * num_strings) / num_strings


def _townsend_effective_snow(snow_total, snow_events):
    '''
    Calculates effective snow using the total snowfall received each month and
    the number of snowfall events each month.
    ... (docstring remains unchanged)
    '''
    snow_events_no_zeros = np.maximum(snow_events, 1)
    effective_snow = 0.5 * snow_total * (1 + 1 / snow_events_no_zeros)
    return np.where(snow_events > 0, effective_snow, 0)


def loss_townsend(snow_total, snow_events, surface_tilt, relative_humidity,
                  temp_air, poa_global, slant_height, lower_edge_height,
                  angle_of_repose=40):
    '''
    Calculates monthly snow loss based on the Townsend monthly snow loss
    model [1]_.
    ... (docstring remains unchanged)
    '''

    C1 = 5.7e04
    C2 = 0.51

    snow_total_prev = np.roll(snow_total, 1)
    snow_events_prev = np.roll(snow_events, 1)

    effective_snow = _townsend_effective_snow(snow_total, snow_events)
    effective_snow_prev = _townsend_effective_snow(
        snow_total_prev,
        snow_events_prev
    )
    effective_snow_weighted = (
        1 / 3 * effective_snow_prev
        + 2 / 3 * effective_snow
    )
    effective_snow_weighted_m = effective_snow_weighted / 100

    lower_edge_height_clipped = np.maximum(lower_edge_height, 0.01)
    gamma = (
        slant_height
        * effective_snow_weighted_m
        * cosd(surface_tilt)
        / (lower_edge_height_clipped**2 - effective_snow_weighted_m**2)
        * 2
        * tand(angle_of_repose)
    )

    ground_interference_term = 1 - C2 * np.exp(-gamma)
    relative_humidity_fraction = relative_humidity / 100
    temp_air_kelvin = temp_air + 273.15
    effective_snow_weighted_in = effective_snow_weighted / 2.54
    poa_global_kWh = poa_global / 1000

    loss_fraction = (
        C1
        * effective_snow_weighted_in
        * cosd(surface_tilt)**2
        * ground_interference_term
        * relative_humidity_fraction
        / temp_air_kelvin**2
        / poa_global_kWh**0.67
    )

    return np.clip(loss_fraction, 0, 1)
