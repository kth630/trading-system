import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from risk.risk_manager import RiskManager

class Plotter:
    def __init__(self, cumulative_strategy, cumulative_benchmark, monthly_regimes=None):
        self.cumulative_strategy = cumulative_strategy
        self.cumulative_benchmark = cumulative_benchmark
        self.monthly_regimes = monthly_regimes

    def plot_both(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. Equity Curve
        ax1.plot(self.cumulative_strategy.index, self.cumulative_strategy, label='Regime-Switching Strategy', color='red', linewidth=2)
        ax1.plot(self.cumulative_benchmark.index, self.cumulative_benchmark, label='S&P 500 (Buy & Hold)', color='blue', alpha=0.5)
        
        # 레짐 구역마다 배경색(아우라) 다르게 칠하기
        if self.monthly_regimes is not None:
            for i in range(len(self.monthly_regimes) - 1):
                start = self.monthly_regimes.index[i]
                end = self.monthly_regimes.index[i+1]
                regime = self.monthly_regimes.iloc[i]
                
                if regime == 'Bull':
                    color = 'green'     # 안전/성장 구간
                elif regime == 'Sideways':
                    color = 'gold'      # 정체/경계 구간
                elif regime == 'Crash':
                    color = 'darkred'   # 위험/패닉 구간
                else:
                    color = 'white'
                    
                ax1.axvspan(start, end, color=color, alpha=0.15, lw=0)

        ax1.set_title('Regime-Switching Performance (Green: Bull, Gold: Sideways, Red: Crash)', fontsize=15)
        ax1.set_ylabel('Cumulative Return (1 = 100%)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Drawdowns
        strat_dd = RiskManager.calculate_drawdowns(self.cumulative_strategy)
        bench_dd = RiskManager.calculate_drawdowns(self.cumulative_benchmark)
        
        ax2.fill_between(strat_dd.index, strat_dd * 100, 0, color='red', alpha=0.4, label='Strategy Drawdown')
        ax2.plot(bench_dd.index, bench_dd * 100, color='blue', alpha=0.3, label='S&P 500 Drawdown')
        ax2.set_title('Underwater Chart (Drawdowns)', fontsize=13)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(loc='lower left', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
