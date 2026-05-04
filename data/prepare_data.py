import pandas as pd
import numpy as np

def generate_second_ohlc_realistic(
    minute_group,
    bar_duration_seconds=60,
    ticks_per_sec=10,
    vol_scale=0.3,
    tick_size=0.01,
    min_volume_frac=0.02,
):
    if len(minute_group) == 0:
        return pd.DataFrame()

    row = minute_group.iloc[0]
    O = row['Open']
    C = row['Close']
    H = row['High']
    L = row['Low']
    total_volume = row.get('Volume', 0)
    is_fill = row.get('is_fill', False)
    start_time = minute_group.index[0]

    n_seconds = bar_duration_seconds
    total_ticks = n_seconds * ticks_per_sec
    t = np.arange(total_ticks)

    # --- Генерация шума ---
    if is_fill:
        price_ticks = O + (C - O) * t / (total_ticks - 1)
    else:
        trend = O + (C - O) * t / (total_ticks - 1)
        price_range = H - L
        main_noise_std = max(price_range * vol_scale, tick_size * 0.2)

        noise = np.random.normal(0, 1, total_ticks).cumsum()
        noise -= noise[0] + (t / (total_ticks - 1)) * (noise[-1] - noise[0])
        micro_noise_std = main_noise_std * 0.15
        micro_noise = np.random.normal(0, micro_noise_std, total_ticks)

        price_ticks = trend + noise * main_noise_std + micro_noise
        price_ticks[0] = O
        price_ticks[-1] = C

    # --- Формирование секундных свечей ---
    end_time = start_time + pd.Timedelta(seconds=n_seconds - 0.001)
    tick_index = pd.date_range(start=start_time, end=end_time, periods=total_ticks)
    tick_series = pd.Series(price_ticks, index=tick_index)

    sec_bars = tick_series.resample('s').ohlc()
    sec_bars.columns = ['Open', 'High', 'Low', 'Close']

    full_idx = pd.date_range(start=start_time, periods=n_seconds, freq='s')
    if len(sec_bars) < n_seconds:
        sec_bars = sec_bars.reindex(full_idx, method='ffill')
    elif len(sec_bars) > n_seconds:
        sec_bars = sec_bars.iloc[:n_seconds]

    sec_bars.iloc[0, sec_bars.columns.get_loc('Open')] = O
    sec_bars.iloc[-1, sec_bars.columns.get_loc('Close')] = C

    # --- Распределение объёма ---
    if total_volume > 0 and not is_fill:
        # Вес: (High-Low) + микро-смещение, чтобы не было нулевых весов
        raw_weights = (sec_bars['High'] - sec_bars['Low']) + tick_size * 0.1
        uniform_share = total_volume * min_volume_frac
        proportional_share = total_volume - uniform_share

        uniform_weight = 1.0 / n_seconds
        proportional_weights = raw_weights / raw_weights.sum()

        weights = uniform_share * uniform_weight + proportional_share * proportional_weights
        volumes = np.floor(weights).astype(int)          # это Series с секундным индексом

        remainder = int(total_volume) - volumes.sum()
        if remainder > 0:
            # Случайные позиции для добавления остатка
            add_indices = np.random.choice(n_seconds, size=remainder, replace=False)
            volumes.iloc[add_indices] += 1               # исправлено: используем .iloc

        sec_bars['Volume'] = volumes
    else:
        sec_bars['Volume'] = 0

    return sec_bars