import pandas as pd

class DualMomentum:
    def __init__(self, lookback_months=12):
        """
        :param lookback_months: 모멘텀(수익률)을 계산할 개월 수. 보통 12개월(1년).
        """
        self.lookback_months = lookback_months
        self.weights = None
        self.momentum_scores = None

    def generate_signals(self, monthly_prices, target_assets, safe_asset):
        """
        월별 가격 데이터를 바탕으로 모멘텀을 계산하고 투자 비중(Weights)을 산출합니다.
        :param monthly_prices: 월별 종가 데이터 (DataFrame)
        :param target_assets: 위험 자산(주식) 티커 리스트 e.g. ['SPY', 'EFA']
        :param safe_asset: 안전 자산(채권) 티커 e.g. 'AGG'
        :return: 투자 비중 (DataFrame)
        """
        # 지정된 개월 수(기본 12개월)를 기준으로 모멘텀(수익률) 계산
        self.momentum_scores = monthly_prices.pct_change(periods=self.lookback_months)
        
        # 포트폴리오 비중 저장 프레임 초기화 (기본값 0)
        self.weights = pd.DataFrame(0.0, index=self.momentum_scores.index, columns=monthly_prices.columns)
        
        print(f"듀얼 모멘텀 시그널 및 타겟 비중 계산 중... (Lookback: {self.lookback_months}개월)")
        
        # Dual Momentum 로직 적용
        for date in self.momentum_scores.index:
            spy_mom = self.momentum_scores.loc[date, target_assets[0]] # 미국 주식 (SPY)
            efa_mom = self.momentum_scores.loc[date, target_assets[1]] # 선진국 주식 (EFA)
            
            # 초기 데이터 부족 기간 패스
            if pd.isna(spy_mom):
                continue
                
            # [절대 모멘텀] 기준 자산(SPY)의 1년 수익률이 0 초과인지 확인
            if spy_mom > 0:
                # [상대 모멘텀] 두 주식 중 더 수익률이 높은 것을 선택
                if spy_mom > efa_mom:
                    self.weights.loc[date, target_assets[0]] = 1.0
                else:
                    self.weights.loc[date, target_assets[1]] = 1.0
            else:
                # 주식 시장이 전체적으로 하락장이면 안전 자산(채권)으로 대피
                self.weights.loc[date, safe_asset] = 1.0
                
        return self.weights
