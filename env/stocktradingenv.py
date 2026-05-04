import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple, Optional
import yfinance as yf

class StockTradingEnv(gym.Env):
    """
    Среда для торговли 20 акциями с двумя головами:
    - direction: [-1, 1] для каждой акции
    - size: [0, 1] для каждой акции (доля капитала на сделку)
    """

    def __init__(
        self,
        tickers: List[str],               # список из 20 тикеров
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        lookback_window: int = 60,        # длина истории для RWKV
        transaction_cost_pct: float = 0.001,   # 0.1% комиссия
        slippage_factor: float = 0.0005,       # 0.05% базовая + проскальзывание от объема
        max_position_pct: float = 0.05,        # не более 5% капитала на одну акцию
        min_signal_threshold: float = 0.05,    # игнорировать |signal| < 0.05
        reward_scaling: float = 1e-4,          # масштабирование награды для стабильности
        use_short: bool = False,               # разрешить короткие продажи
        max_steps_per_episode: int = 252,      # 1 торговый год
        seed: Optional[int] = None
    ):
        super().__init__()
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_capital = initial_capital
        self.lookback = lookback_window
        self.transaction_cost = transaction_cost_pct
        self.slippage_factor = slippage_factor
        self.max_position_pct = max_position_pct
        self.min_signal = min_signal_threshold
        self.reward_scaling = reward_scaling
        self.use_short = use_short
        self.max_steps = max_steps_per_episode
        self.seed(seed)


        # Загружаем данные через API (локально кэшируем для скорости)
        self._load_data(start_date, end_date)
        self._calculate_features()

        # Пространство наблюдений: (lookback, n_features, n_assets)
        # n_features = количество признаков на одну акцию
        self.n_features = self.features_array.shape[1]  # реальное число признаков (например, 8)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback, self.n_features, self.n_assets),
            dtype=np.float32
        )

        # Action space: объединённый вектор (direction + size)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self.n_assets,),
            dtype=np.float32
        )

        self.reset()

    def _load_data(self, start_date: str, end_date: str):
        """Загружает OHLCV для всех тикеров через yfinance, возвращает плоский DataFrame."""
        # Скачиваем все тикеры одним запросом (быстрее и удобнее)
        data = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',   # <- важный параметр: возвращает MultiIndex (Ticker, Price)
            progress=False,
            auto_adjust=False    # оставляем оригинальные OHLCV
        )

        # Если скачан только один тикер, yfinance возвращает обычный DataFrame (не MultiIndex)
        if len(self.tickers) == 1:
            df = data.copy()
            df.columns = [f"{self.tickers[0]}_{col}" for col in df.columns]
        else:
            # Для нескольких тикеров — MultiIndex, преобразуем в плоские колонки
            df_list = []
            for ticker in self.tickers:
                ticker_df = data[ticker].copy()
                ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
                df_list.append(ticker_df)
            df = pd.concat(df_list, axis=1, join='inner')

        # Убедимся, что индекс — это datetime
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Проверка: все ли ожидаемые колонки присутствуют
        expected_cols = [f"{ticker}_{field}" for ticker in self.tickers for field in ['Open','High','Low','Close','Volume']]
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns after download: {missing}")

        self.prices = df
        self.price_array = df.values.astype(np.float32)
        self.dates = df.index

    def _calculate_features(self):
        n = len(self.dates)
        n_assets = self.n_assets
        n_features = 8
        features = np.zeros((n, n_features, n_assets), dtype=np.float32)

        for i, ticker in enumerate(self.tickers):
            close = self.prices[f"{ticker}_Close"].values.astype(float)
            high = self.prices[f"{ticker}_High"].values.astype(float)
            low = self.prices[f"{ticker}_Low"].values.astype(float)
            volume = self.prices[f"{ticker}_Volume"].values.astype(float)

            # Защита от нулевых и отрицательных цен
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

            # Заполняем массив
            features[:, 0, i] = np.nan_to_num(ret1, nan=0.0, posinf=1.0, neginf=-1.0)
            features[:, 1, i] = np.nan_to_num(ret5, nan=0.0, posinf=1.0, neginf=-1.0)
            features[:, 2, i] = np.nan_to_num(rsi / 100.0, nan=0.5, posinf=1.0, neginf=0.0)
            features[:, 3, i] = np.nan_to_num(bb_pct_b, nan=0.5, posinf=1.0, neginf=0.0)
            features[:, 4, i] = np.nan_to_num(atr_rel, nan=0.0, posinf=1.0, neginf=0.0)
            features[:, 5, i] = np.nan_to_num(vol_ratio, nan=1.0, posinf=10.0, neginf=0.0)
            features[:, 6, i] = np.nan_to_num(daily_pos, nan=0.5, posinf=1.0, neginf=0.0)
            features[:, 7, i] = np.nan_to_num(mom5, nan=0.0, posinf=1.0, neginf=-1.0)

            # Клиппинг, чтобы не было гигантских выбросов
            features[:, :, i] = np.clip(features[:, :, i], -5.0, 5.0)

        # Z‑нормализация по скользящему окну
        norm_features = np.zeros_like(features)
        for t in range(self.lookback, n):
            window = features[t-self.lookback:t, :, :]
            mean = np.mean(window, axis=0, keepdims=True)
            std = np.std(window, axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)   # избегаем деления на ноль
            norm_features[t] = (features[t] - mean) / std
        norm_features[:self.lookback] = 0.0
        self.features_array = norm_features.astype(np.float32)

    @staticmethod
    def _sma(arr, window):
        ret = np.cumsum(arr, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        ret[:window-1] = np.nan
        return ret / window

    @staticmethod
    def _rolling_std(arr, window):
        return pd.Series(arr).rolling(window).std().fillna(0).values

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Начинаем с lookback дня (чтобы было достаточно истории)
        self.current_step = self.lookback
        self.capital = self.initial_capital
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)  # количество акций
        self.cost_basis = np.zeros(self.n_assets, dtype=np.float32)  # средняя цена покупки (если нужно)
        self.portfolio_history = []
        self.reward_history = []
        self.done = False

        # Состояние: (lookback, features, assets)
        state = self._get_obs()
        return state, {}

    def _get_obs(self):
        """Возвращает последние lookback дней признаков."""
        start = self.current_step - self.lookback
        end = self.current_step
        # features_array: (time, feature, asset)
        obs = self.features_array[start:end]  # (lookback, F, A)
        return obs.astype(np.float32)

    def _get_current_prices(self) -> np.ndarray:
        """Возвращает цены закрытия для текущего шага."""
        idx = min(self.current_step, len(self.dates) - 1)
        close_cols = [f"{ticker}_Close" for ticker in self.tickers]
        return self.prices[close_cols].iloc[idx].values.astype(np.float32)

    def _apply_action(self, direction: np.ndarray, size: np.ndarray):
        current_prices = self._get_current_prices()
        current_holdings = self.holdings.copy()
        current_capital = self.capital
        portfolio_value = current_capital + np.sum(current_holdings * current_prices)

        # 1. Сигнал с порогом
        signal = direction * size
        signal[np.abs(signal) < self.min_signal] = 0.0

        # 2. Желаемые веса (доли портфеля) – ограничены max_position_pct
        target_weights = np.clip(signal, -self.max_position_pct, self.max_position_pct)
        target_values = target_weights * portfolio_value

        # 3. Текущие стоимости
        current_values = current_holdings * current_prices

        # 4. Необходимое изменение стоимости (∆V)
        delta_value = target_values - current_values

        # 5. Если шорт запрещён, нельзя продать больше, чем имеем
        if not self.use_short:
            delta_value = np.where(delta_value < 0, np.maximum(delta_value, -current_values), delta_value)

        # 6. Переводим ∆V в количество акций
        shares_to_trade = delta_value / (current_prices + 1e-8)

        # 7. Ограничиваем покупку доступным капиталом (только для положительных shares)
        buy_value = np.sum(np.maximum(shares_to_trade, 0) * current_prices)
        if buy_value > current_capital:
            # Пропорционально ужимаем все покупки
            scale = (current_capital / buy_value) * 0.99  # оставляем запас на комиссии
            shares_to_trade[shares_to_trade > 0] *= scale

        # 8. Денежный поток (отрицательный для покупок)
        cash_flow = -np.sum(shares_to_trade * current_prices)

        # 9. Издержки
        trade_amount = np.abs(shares_to_trade * current_prices)
        # Дневной объём (последний известный)
        daily_volumes = self.prices[[f"{ticker}_Volume" for ticker in self.tickers]].iloc[self.current_step].values
        daily_volume_usd = daily_volumes * current_prices + 1e-8
        slippage_ratio = self.slippage_factor + 0.1 * trade_amount / daily_volume_usd
        slippage_cost = np.sum(trade_amount * slippage_ratio)
        commission = self.transaction_cost * np.sum(trade_amount)
        total_costs = commission + slippage_cost

        # 10. Обновляем капитал
        new_capital = current_capital + cash_flow - total_costs
        if new_capital < 0:
            new_capital = 0.0

        new_holdings = current_holdings + shares_to_trade
        if not self.use_short:
            new_holdings = np.maximum(new_holdings, 0.0)
        executed_shares = shares_to_trade.copy()
        return new_holdings, new_capital, total_costs, commission, slippage_cost, new_holdings, executed_shares

    def _calculate_reward(self, old_portfolio_value: float, new_portfolio_value: float,
                          total_costs: float, new_holdings: np.ndarray) -> float:
        """
        Награда = относительное изменение портфеля - штраф за издержки - штраф за просадку/волатильность.
        """
        # 1. Базовая доходность (log-доходность портфеля)
        if old_portfolio_value <= 0:
            return -1.0  # банкротство
        ret = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        # 2. Вычитаем издержки (как долю от портфеля)
        cost_penalty = total_costs / old_portfolio_value
        reward = ret - cost_penalty

        # 3. Дополнительные штрафы: за высокую концентрацию (HHI индекса Херфиндаля позиций)
        weights = np.abs(new_holdings * self._get_current_prices()) / (new_portfolio_value + 1e-8)
        hhi = np.sum(weights ** 2)
        concentration_penalty = 0.01 * (hhi - 1/self.n_assets)  # если HHI выше равномерного

        # 4. Штраф за волатильность портфеля (можно добавлять скользящую волатильность, но тут упрощённо)
        # Сохраняем историю доходностей для расчёта волатильности (дополнительно в методе step)
        # Упростим: штраф, если новое значение портфеля слишком сильно отклонилось от предыдущего тренда
        if len(self.portfolio_history) > 1:
            prev_ret = (self.portfolio_history[-1] - self.portfolio_history[-2]) / self.portfolio_history[-2]
            if abs(ret - prev_ret) > 0.05:  # резкое изменение
                reward -= 0.005

        reward -= concentration_penalty
        # Масштабируем награду для стабильности обучения
        reward *= self.reward_scaling
        return np.clip(reward, -10.0, 10.0)  # клиппинг для устойчивости

    def step(self, action: np.ndarray):
        direction = action[:self.n_assets]
        size = action[self.n_assets:]
        # --- 1. Состояние до любых изменений ---
        current_prices = self._get_current_prices()
        old_portfolio_value = self.capital + np.sum(self.holdings * current_prices)

        # --- 2. Исполняем сделки по старым ценам ---
        new_holdings, new_capital, total_costs, comm, slippage, new_holdings, executed_shares = self._apply_action(
            direction, size
        )
        self.holdings = new_holdings
        self.capital = new_capital

        # --- 3. Переход к следующему дню ---
        self.current_step += 1
        if self.current_step >= len(self.dates):
            self.done = True

        # --- 4. Оценка портфеля по новым ценам ---
        new_prices = self._get_current_prices()
        new_portfolio_value = self.capital + np.sum(self.holdings * new_prices)

        # --- 5. Награда = доходность портфеля (издержки уже внутри капитала) ---
        if old_portfolio_value <= 0:
            reward = -1.0
        else:
            reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value

        # --- 6. Штраф за чрезмерную концентрацию (опционально) ---
        if new_portfolio_value > 0:
            weights = np.abs(self.holdings * new_prices) / (new_portfolio_value + 1e-8)
            concentration = np.sum(weights ** 2) - 1.0 / self.n_assets
            reward -= 0.01 * max(0.0, concentration)

        # --- 7. Масштабирование и клиппинг ---
        reward *= self.reward_scaling
        reward = np.clip(reward, -10.0, 10.0)

        # --- 8. Сохранение истории ---
        self.portfolio_history.append(new_portfolio_value)
        self.reward_history.append(reward)

        # --- 9. Флаги завершения ---
        if self.capital <= 0 or new_portfolio_value <= 0:
            self.done = True
        truncated = (self.current_step >= self.max_steps + self.lookback) or self.done

        # --- 10. Следующее наблюдение ---
        next_obs = self._get_obs()

        info = {
            'portfolio_value': new_portfolio_value,
            'capital': self.capital,
            'holdings': self.holdings.copy(),
            'total_cost': total_costs,
            'commission': comm,
            'slippage': slippage,
            'step': self.current_step,
            'executed_shares': executed_shares
        }
        return next_obs, reward, self.done, truncated, info
