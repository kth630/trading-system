import pandas as pd
import numpy as np
import sys
import os

# 현재 파일 위치보다 상위 폴더를 sys.path에 추가하여 risk 경로를 찾도록 함
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from risk.risk_manager import RiskManager

class Evaluator:
    def __init__(self, monthly_prices, weights):
        self.monthly_prices = monthly_prices
        self.weights = weights
        self.strategy_returns = None
        self.cumulative_strategy = None
        
        self.benchmark_returns = None
        self.cumulative_benchmark = None

    def run_backtest(self, benchmark_col='SPY'):
        print("포트폴리오 백테스트 계산 중...")
        # 실제 투자는 판단한 달의 '다음 달'에 이루어지므로 비중(weights)을 1칸 뒤로(shift) 밉니다.
        adjusted_weights = self.weights.shift(1).dropna()
        
        # 각 자산별 월간 수익률 계산
        monthly_returns = self.monthly_prices.pct_change()
        
        # 전략의 월간 수익률 = 비중 * 해당 자산의 월 수익률의 총합
        self.strategy_returns = (adjusted_weights * monthly_returns).sum(axis=1)
        # 길이를 맞춤
        self.strategy_returns = self.strategy_returns[adjusted_weights.index] 
        
        # 전략 누적 수익률
        self.cumulative_strategy = (1 + self.strategy_returns).cumprod()
        self.cumulative_strategy = self.cumulative_strategy / self.cumulative_strategy.iloc[0]
        
        # 벤치마크 누적 수익률
        self.benchmark_returns = monthly_returns[benchmark_col].dropna()
        self.cumulative_benchmark = (1 + self.benchmark_returns).cumprod()
        
        # 전략과 동일한 시작 시점으로 동기화
        align_start = self.cumulative_strategy.index[0]
        if align_start in self.cumulative_benchmark.index:
            self.cumulative_benchmark = self.cumulative_benchmark.loc[align_start:]
            self.cumulative_benchmark = self.cumulative_benchmark / self.cumulative_benchmark.iloc[0]

        return self.cumulative_strategy, self.cumulative_benchmark

    def get_performance_metrics(self):
        if self.cumulative_strategy is None:
            self.run_backtest()
            
        print("성과 지표(Metrics) 계산 중...")
        
        # 1. Total Return
        total_ret = self.cumulative_strategy.iloc[-1] - 1
        bench_ret = self.cumulative_benchmark.iloc[-1] - 1
        
        # 2. CAGR
        months = len(self.cumulative_strategy)
        years = months / 12.0
        cagr = (self.cumulative_strategy.iloc[-1] ** (1 / years)) - 1
        bench_cagr = (self.cumulative_benchmark.iloc[-1] ** (1 / years)) - 1
        
        # 3. MDD
        mdd = RiskManager.calculate_mdd(self.cumulative_strategy)
        bench_mdd = RiskManager.calculate_mdd(self.cumulative_benchmark)
        
        # 4. 변동성 (월간 수익률 기반이므로 12 곱함)
        volatility = RiskManager.calculate_annualized_volatility(self.strategy_returns, periods_per_year=12)
        bench_vol = RiskManager.calculate_annualized_volatility(self.benchmark_returns.loc[self.strategy_returns.index], periods_per_year=12)
        
        # 5. 샤프 지수 (무위험 수익률은 보수적으로 0으로 설정)
        sharpe = cagr / volatility if volatility != 0 else 0
        bench_sharpe = bench_cagr / bench_vol if bench_vol != 0 else 0
        
        metrics = {
            'Metric': ['총 누적 수익률 (Total Return)', '연평균 수익률 (CAGR)', '최대 낙폭 (MDD)', '연환산 변동성 (Volatility)', '샤프 지수 (Sharpe Ratio)'],
            '듀얼 모멘텀 전략': [f"{total_ret*100:.2f}%", f"{cagr*100:.2f}%", f"{mdd*100:.2f}%", f"{volatility*100:.2f}%", f"{sharpe:.2f}"],
            'S&P 500 (단순 보유)': [f"{bench_ret*100:.2f}%", f"{bench_cagr*100:.2f}%", f"{bench_mdd*100:.2f}%", f"{bench_vol*100:.2f}%", f"{bench_sharpe:.2f}"]
        }
        
        return pd.DataFrame(metrics)
