import os
import sys

# 하위 모듈 임포트용 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.data_loader import DataLoader
from strategy.regime_detector import RegimeDetector
from strategy.dynamic_router import DynamicRouter
from backtest.evaluator import Evaluator
from dashboard.plotter import Plotter

def main():
    print("="*65)
    print(" 🚀 레짐 체제 기반 동적 자산 배분 시스템 실행 (Regime-Switching) ")
    print("="*65)
    
    # 듀얼모멘텀, 올웨더, 현금대피에 통합적으로 필요한 티커 구성
    tickers = ['SPY', 'EFA', 'AGG', 'TLT', 'GLD', 'SHV']
    start_date = '2005-01-01'
    lookback_months = 12
    
    # 1. 데이터 로드 (Data Loader)
    loader = DataLoader(tickers, start_date)
    daily_prices = loader.get_daily_data()
    monthly_prices = loader.get_monthly_data()
    
    # 2. 시장 레짐(장세) 감지기 실행 (Regime Detector)
    detector = RegimeDetector(benchmark_col='SPY', sma_window=200, vol_window=20)
    detector.detect_regime(daily_prices)
    monthly_regimes = detector.get_monthly_regimes()
    
    # 3. 전략 라우터 실행 및 월별 비중 도출 (Dynamic Router)
    router = DynamicRouter(monthly_regimes, monthly_prices, lookback_months=lookback_months)
    weights = router.generate_final_weights()
    
    # 4. 백테스트 수행 및 성과 확인 (Evaluation)
    evaluator = Evaluator(monthly_prices, weights)
    cumulative_strategy, cumulative_benchmark = evaluator.run_backtest(benchmark_col='SPY')
    
    metrics_df = evaluator.get_performance_metrics()
    
    print("\n[백테스트 성과 요약 (Benchmark: S&P 500)]")
    print("-" * 75)
    print(metrics_df.to_string(index=False))
    print("-" * 75)
    
    # 5. 시각화 (Dashboard Plot)
    print("\n차트를 생성하고 있습니다... 창이 뜨면 상단의 '레짐별 구역 색상' 진입 지점들을 관찰해보세요.")
    plotter = Plotter(cumulative_strategy, cumulative_benchmark, monthly_regimes=monthly_regimes)
    plotter.plot_both()
    
    print("\n레짐 스위칭 다중 모델 백테스트 정상 종료.")

if __name__ == "__main__":
    main()
