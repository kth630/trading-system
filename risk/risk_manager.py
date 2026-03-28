import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self):
        pass

    @staticmethod
    def calculate_drawdowns(cumulative_returns):
        """
        누적 수익률(Cumulative Returns) 시계열을 받아 현재 낙폭(Drawdown)을 계산합니다.
        """
        # 누적 최고점(Running Maximum) 계산
        running_max = cumulative_returns.cummax()
        # 현재 누적 수익률이 고점 대비 얼마나 빠졌는지 계산 (Drawdown)
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown

    @staticmethod
    def calculate_mdd(cumulative_returns):
        """
        최대 낙폭(Maximum Drawdown)을 계산합니다.
        """
        drawdown = RiskManager.calculate_drawdowns(cumulative_returns)
        mdd = drawdown.min()
        return mdd
        
    @staticmethod
    def calculate_annualized_volatility(returns, periods_per_year=252):
        """
        연환산 변동성(Annualized Volatility)을 계산합니다.
        (월간 데이터로 계산할 경우 periods_per_year=12 사용)
        """
        return returns.std() * np.sqrt(periods_per_year)
