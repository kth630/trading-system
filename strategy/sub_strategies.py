import pandas as pd

class SubStrategies:
    @staticmethod
    def run_dual_momentum(date, monthly_momentum, target_assets=['SPY', 'EFA'], safe_asset='AGG'):
        """상승장 전용: 오리지널 듀얼 모멘텀 (수익 극대화)"""
        weights = pd.Series(0.0, index=monthly_momentum.columns)
        
        spy_mom = monthly_momentum.loc[date, target_assets[0]]
        efa_mom = monthly_momentum.loc[date, target_assets[1]]
        
        if pd.isna(spy_mom):
            return weights
            
        if spy_mom > 0:
            if spy_mom > efa_mom:
                weights[target_assets[0]] = 1.0 # SPY
            else:
                weights[target_assets[1]] = 1.0 # EFA
        else:
            weights[safe_asset] = 1.0 # 혹시 모를 안전장치
            
        return weights

    @staticmethod
    def run_all_weather(date, assets):
        """횡보장 전용: 주식/장기채/금 방어적 정적 배분 (40/40/20)"""
        weights = pd.Series(0.0, index=assets)
        if 'SPY' in assets: weights['SPY'] = 0.4
        if 'TLT' in assets: weights['TLT'] = 0.4
        if 'GLD' in assets: weights['GLD'] = 0.2
        return weights

    @staticmethod
    def run_cash_protection(date, assets, cash_asset='SHV'):
        """패닉장 전용: 100% 현금/초단기채 대피"""
        weights = pd.Series(0.0, index=assets)
        if cash_asset in assets:
            weights[cash_asset] = 1.0
        return weights
