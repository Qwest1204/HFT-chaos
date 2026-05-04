import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yfinance as yf

class MarketStateDataset(Dataset):
    """
    Датасет для обучения трансформера на мульти-активном временном ряду.

    Каждый элемент возвращает:
        X : (seq_len, n_assets, n_features)   – исторические признаки
        y : (n_assets,)                       – целевое значение (возврат или класс)

    Параметр target_mode:
        'regression'     -> y = доходность следующего дня (ret1)
        'classification' -> y = 0 (down), 1 (flat), 2 (up)
            Пороги задаются через thresholds = (down_thresh, up_thresh)
    """
    def __init__(
        self,
        seq_len: int = 60,
        target_mode: str = 'regression',   # 'regression' или 'classification'
        thresholds: tuple = (-0.005, 0.005),  # для классификации: down, flat, up
        start_idx: int = None,
        end_idx: int = None,
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        normalize: bool = True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.target_mode = target_mode
        self.thresholds = thresholds
        self.tickers = tickers
        self.n_assets = len(tickers)

        # 1. Загрузка сырых данных
        self.df, self.raw, self.index = self._load_data(tickers)

        # 2. Вычисление всех признаков (n_features=8, shape = (n_times, n_features, n_assets))
        self.features = self._calculate_features(self.df, tickers)

        # 3. Адаптивная Z‑нормализация (если включена)
        if normalize:
            self.features = self._rolling_zscore(self.features, window=30)

        # 4. Определяем валидный диапазон индексов:
        #    - нельзя брать начало раньше, чем seq_len (чтобы был полный вход)
        #    - нельзя брать конец позже, чем длина ряда минус 1 (чтобы был target)
        self.valid_start = self.seq_len
        self.valid_end = len(self.df) - 1
        if start_idx is None:
            start_idx = self.valid_start
        if end_idx is None:
            end_idx = self.valid_end
        self.start_idx = max(start_idx, self.valid_start)
        self.end_idx = min(end_idx, self.valid_end)
        if self.start_idx >= self.end_idx:
            raise ValueError("Invalid start/end indices: dataset empty.")

    # ------------------------------------------------------------------
    # Вспомогательные методы – загрузка и расчёт признаков (на основе предоставленного кода)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_data(tickers):
        """Загружает OHLCV для всех тикеров через yfinance, возвращает плоский DataFrame."""
        data = yf.download(
            tickers,
            start='2020-05-10',
            end='2025-05-10',
            group_by='ticker',
            progress=False,
            auto_adjust=False
        )
        df_list = []
        for ticker in tickers:
            ticker_df = data[ticker].copy()
            ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
            df_list.append(ticker_df)
        df = pd.concat(df_list, axis=1, join='inner')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Проверка наличия всех колонок
        expected_cols = [f"{ticker}_{field}" for ticker in tickers
                         for field in ['Open', 'High', 'Low', 'Close', 'Volume']]
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns after download: {missing}")
        return df, df.values.astype(np.float32), df.index

    @staticmethod
    def _sma(arr, window):
        ret = np.cumsum(arr, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        ret[:window-1] = np.nan
        return ret / window

    @staticmethod
    def _rolling_std(arr, window):
        return pd.Series(arr).rolling(window).std().fillna(0).values

    def _calculate_features(self, df, tickers):
        """Рассчитывает 8 технических признаков для каждого актива. Форма (N, 8, n_assets)."""
        n = len(df)
        n_features = 8
        features = np.zeros((n, n_features, len(tickers)), dtype=np.float32)

        for i, ticker in enumerate(tickers):
            close = df[f"{ticker}_Close"].values.astype(float)
            high = df[f"{ticker}_High"].values.astype(float)
            low = df[f"{ticker}_Low"].values.astype(float)
            volume = df[f"{ticker}_Volume"].values.astype(float)

            # Защита от нулевых/отрицательных цен
            close = np.maximum(close, 1e-6)
            high = np.maximum(high, close)
            low = np.minimum(low, close)

            # 1. Лог доходность 1 день
            ret1 = np.diff(np.log(close), prepend=np.log(close[0]))
            # 2. Лог доходность 5 дней
            ret5 = np.log(close) - np.log(np.roll(close, 5))
            ret5[:5] = 0.0
            # 3. RSI(14)
            delta = np.diff(close, prepend=close[0])
            gain = np.clip(delta, 0, None)
            loss = np.clip(-delta, 0, None)
            avg_gain = self._sma(gain, 14)
            avg_loss = self._sma(loss, 14)
            rs = np.divide(avg_gain, avg_loss + 1e-8)
            rsi = 100 - 100 / (1 + rs)
            # 4. Bollinger %B
            sma20 = self._sma(close, 20)
            std20 = self._rolling_std(close, 20) + 1e-8
            bb_pct_b = (close - (sma20 - 2*std20)) / (4*std20 + 1e-8)
            # 5. ATR / close
            tr = np.maximum(high - low,
                            np.abs(high - np.roll(close, 1)),
                            np.abs(low - np.roll(close, 1)))
            atr = self._sma(tr, 14)
            atr_rel = atr / (close + 1e-8)
            # 6. Объемный импульс
            vol_ma20 = self._sma(volume, 20)
            vol_ratio = volume / (vol_ma20 + 1e-8)
            # 7. Нормализованная позиция в диапазоне дня
            daily_pos = (close - low) / (high - low + 1e-8)
            # 8. Моментум 5 дней
            mom5 = close / (np.roll(close, 5) + 1e-8) - 1
            mom5[:5] = 0.0

            # Заполняем массив с очисткой некорректных значений
            features[:, 0, i] = np.nan_to_num(ret1, nan=0.0, posinf=1.0, neginf=-1.0)
            features[:, 1, i] = np.nan_to_num(ret5, nan=0.0, posinf=1.0, neginf=-1.0)
            features[:, 2, i] = np.nan_to_num(rsi / 100.0, nan=0.5, posinf=1.0, neginf=0.0)
            features[:, 3, i] = np.nan_to_num(bb_pct_b, nan=0.5, posinf=1.0, neginf=0.0)
            features[:, 4, i] = np.nan_to_num(atr_rel, nan=0.0, posinf=1.0, neginf=0.0)
            features[:, 5, i] = np.nan_to_num(vol_ratio, nan=1.0, posinf=10.0, neginf=0.0)
            features[:, 6, i] = np.nan_to_num(daily_pos, nan=0.5, posinf=1.0, neginf=0.0)
            features[:, 7, i] = np.nan_to_num(mom5, nan=0.0, posinf=1.0, neginf=-1.0)
            features[:, :, i] = np.clip(features[:, :, i], -5.0, 5.0)

        return features

    @staticmethod
    def _rolling_zscore(features, window=30):
        """Применяет Z‑нормализацию по скользящему окну к каждому фичу/активу независимо."""
        n = features.shape[0]
        norm = np.zeros_like(features)
        for t in range(window, n):
            wnd = features[t-window:t, :, :]
            mean = np.mean(wnd, axis=0, keepdims=True)
            std = np.std(wnd, axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            norm[t] = (features[t] - mean) / std
        # первые window точек оставляем нулевыми (как в оригинале)
        return norm.astype(np.float32)

    # ------------------------------------------------------------------
    # Интерфейс Dataset
    # ------------------------------------------------------------------
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # idx смещён на self.start_idx, реальный индекс в массиве features:
        t = self.start_idx + idx
        # Вход: окно длиной seq_len до t включительно
        X = self.features[t - self.seq_len + 1 : t + 1]   # (seq_len, n_features, n_assets)
        # Цель: следующее состояние (t+1) – первый признак (ret1) или его класс
        target_data = self.features[t + 1, :, :]          # (n_assets,)

        if self.target_mode == 'regression':
            y = torch.tensor(target_data, dtype=torch.float32)
        elif self.target_mode == 'classification':
            down_thr, up_thr = self.thresholds
            classes = np.zeros_like(target_data, dtype=np.int64)
            classes[target_data < down_thr] = 0   # down
            classes[(target_data >= down_thr) & (target_data <= up_thr)] = 1  # flat
            classes[target_data > up_thr] = 2     # up
            y = torch.tensor(classes, dtype=torch.long)
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        # X транспонируем для удобства: (seq_len, n_assets, n_features)
        X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # -> (seq_len, n_assets, n_features)
        return X, y