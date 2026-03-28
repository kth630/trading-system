import pandas as pd
from strategy.sub_strategies import SubStrategies

class DynamicRouter:
    def __init__(self, monthly_regimes, monthly_prices, lookback_months=12):
        self.monthly_regimes = monthly_regimes
        self.monthly_prices = monthly_prices
        self.lookback_months = lookback_months
        self.weights = None

    def generate_final_weights(self):
        print("레짐 스위칭 기반 동적 자산 배분 비중(Weights) 라우팅 중...")
        # 12개월 모멘텀 계산 (듀얼모멘텀 용도)
        monthly_momentum = self.monthly_prices.pct_change(periods=self.lookback_months)
        
        assets = self.monthly_prices.columns
        self.weights = pd.DataFrame(0.0, index=self.monthly_prices.index, columns=assets)
        
        # 월별 레짐 정보를 트리거로 삼아 하위 전략 분기
        for date in self.monthly_prices.index:
            if date not in self.monthly_regimes.index:
                continue
                
            regime = self.monthly_regimes.loc[date]
            
            # 초기 데이터 미형성 구간 보호 (현금)
            if pd.isna(monthly_momentum.loc[date, 'SPY']) or regime == 'Unknown':
                self.weights.loc[date] = SubStrategies.run_cash_protection(date, assets, cash_asset='SHV')
                continue
                
            # 레짐(장세)별 맞춤형 모델 적용
            if regime == 'Bull':
                # 상승장 -> 수익 극대화 듀얼 모멘텀 모델
                self.weights.loc[date] = SubStrategies.run_dual_momentum(
                    date, monthly_momentum, target_assets=['SPY', 'EFA'], safe_asset='AGG'
                )
            elif regime == 'Sideways':
                # 횡보장 -> 주식/채권/금 올웨더 모델
                self.weights.loc[date] = SubStrategies.run_all_weather(date, assets)
            elif regime == 'Crash':
                # 폭락장 -> 초단기채(현금) 대피 모델
                self.weights.loc[date] = SubStrategies.run_cash_protection(date, assets, cash_asset='SHV')
                
        return self.weights
