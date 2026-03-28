import pandas as pd
import numpy as np

class RegimeDetector:
    def __init__(self, benchmark_col='SPY', sma_window=200, vol_window=20):
        self.benchmark_col = benchmark_col
        self.sma_window = sma_window
        self.vol_window = vol_window
        self.daily_regimes = None

    def detect_regime(self, daily_prices):
        """
        일별 데이터를 기반으로 매일의 마켓 레짐(장세)을 판별합니다.
        반환값: 'Bull', 'Sideways', 'Crash'
        """
        print(f"시장 레짐 (Regime) 분석 중... (이동평균: {self.sma_window}일, 단기변동성: {self.vol_window}일)")
        bm_price = daily_prices[self.benchmark_col]
        
        # 200일 이동평균선
        sma = bm_price.rolling(window=self.sma_window).mean()
        
        # 20일 단기 일간 수익률 변동성 (연환산)
        daily_returns = bm_price.pct_change()
        rolling_vol = daily_returns.rolling(window=self.vol_window).std() * np.sqrt(252)
        
        # 1년(252일) 역사적 평균 변동성 (임계값 계산용)
        historical_vol_mean = rolling_vol.rolling(window=252).mean()
        
        regimes = pd.Series(index=bm_price.index, dtype='object')
        
        # 레짐 판단 조건
        for date in bm_price.index:
            price = bm_price.loc[date]
            ma = sma.loc[date]
            vol = rolling_vol.loc[date]
            vol_mean = historical_vol_mean.loc[date]
            
            if pd.isna(ma) or pd.isna(vol_mean):
                regimes.loc[date] = 'Unknown'
                continue
                
            if price > ma:
                # 200일선 위면 무조건 상승장 (Bull)
                regimes.loc[date] = 'Bull'
            else:
                # 200일선 아래인데 변동성이 평균의 1.5배 이상 폭발하면 패닉장 (Crash)
                if vol > (vol_mean * 1.5):
                    regimes.loc[date] = 'Crash'
                else:
                    # 200일선 아래지만 변동성이 평이하면 횡보/약세장 (Sideways)
                    regimes.loc[date] = 'Sideways'
                    
        self.daily_regimes = regimes
        return regimes

    def get_monthly_regimes(self):
        """
        매월 말 기준(리밸런싱 시점)의 레짐 상태를 반환합니다.
        """
        if self.daily_regimes is None:
            raise ValueError("먼저 detect_regime()을 실행해야 합니다.")
        
        return self.daily_regimes.resample('ME').last()
