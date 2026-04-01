"""
기술지표 기반 레짐(Market Regime) 판단 모듈

구조:
    BaseRegimeDetector (ABC)          — 모든 레짐 판단기의 공통 인터페이스
        └─ BollingerBandRegime        — 볼린저 밴드 기반 레짐 판단 (Bandwidth + %b)

향후 확장:
    BaseRegimeDetector
        ├─ BollingerBandRegime
        ├─ RSIRegime
        └─ MacroRegime  ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
#  공통 레짐 상태 정의
# ──────────────────────────────────────────────
class RegimeState(IntEnum):
    """시장 레짐 상태 코드"""
    STRONG_DOWN = -2    # 강한 하락 추세
    BREAKOUT_DOWN = -1  # 하락 돌파장
    RANGE_BOUND = 0     # 박스권 / 평균회귀장
    BREAKOUT_UP = 1     # 상승 돌파장
    STRONG_UP = 2       # 강한 상승 추세


@dataclass
class RegimeResult:
    """레짐 판단 결과를 담는 데이터 클래스

    Attributes:
        regime: 현재 레짐 상태 (RegimeState)
        confidence: 신뢰도 (0.0 ~ 1.0)
        indicators: 판단에 사용된 지표 값들 (pd.DataFrame)
        label: 사람이 읽을 수 있는 레짐 설명
    """
    regime: pd.Series          # 각 시점의 RegimeState
    confidence: pd.Series      # 각 시점의 신뢰도
    indicators: pd.DataFrame   # 판단에 사용된 모든 중간 지표
    label: pd.Series           # 사람이 읽기 쉬운 레짐 라벨


# ──────────────────────────────────────────────
#  추상 베이스 클래스
# ──────────────────────────────────────────────
class BaseRegimeDetector(ABC):
    """기술지표 기반 레짐 판단의 추상 베이스 클래스

    하위 클래스에서 반드시 구현해야 할 메서드:
        - _compute_indicators(): 기술지표 계산
        - _classify_regime(): 지표 → 레짐 상태 매핑
    """

    def __init__(self, name: str = "BaseRegime"):
        self.name = name
        self._last_result: Optional[RegimeResult] = None

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """레짐 판단 실행 (Template Method 패턴)

        Args:
            df: OHLCV 데이터프레임.
                최소 컬럼: 'close' (종가)
                선택 컬럼: 'open', 'high', 'low', 'volume'

        Returns:
            RegimeResult: 레짐 판단 결과

        Raises:
            ValueError: 필수 컬럼 누락 또는 데이터 부족 시
        """
        self._validate_input(df)
        indicators = self._compute_indicators(df)
        regime, confidence, label = self._classify_regime(indicators)

        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            label=label,
        )
        self._last_result = result
        return result

    def _validate_input(self, df: pd.DataFrame) -> None:
        """입력 데이터 유효성 검증"""
        if "close" not in df.columns:
            raise ValueError(
                f"[{self.name}] 'close' 컬럼이 필요합니다. "
                f"현재 컬럼: {list(df.columns)}"
            )
        if len(df) < 2:
            raise ValueError(
                f"[{self.name}] 최소 2행 이상의 데이터가 필요합니다. "
                f"현재: {len(df)}행"
            )

    @abstractmethod
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술지표 계산 — 하위 클래스에서 구현

        Returns:
            pd.DataFrame: 계산된 지표가 추가된 데이터프레임
        """
        ...

    @abstractmethod
    def _classify_regime(
        self, indicators: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """지표 → 레짐 상태 분류 — 하위 클래스에서 구현

        Returns:
            (regime, confidence, label) 튜플
        """
        ...

    @property
    def last_result(self) -> Optional[RegimeResult]:
        """마지막 판단 결과 반환"""
        return self._last_result

    def summary(self) -> pd.DataFrame:
        """레짐별 분포 요약 통계"""
        if self._last_result is None:
            raise RuntimeError("detect()를 먼저 실행하세요.")

        regime = self._last_result.regime
        total = len(regime.dropna())

        summary_rows = []
        for state in RegimeState:
            count = (regime == state).sum()
            summary_rows.append({
                "regime": state.name,
                "code": int(state),
                "count": count,
                "ratio": round(count / total, 4) if total > 0 else 0.0,
            })
        return pd.DataFrame(summary_rows)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ──────────────────────────────────────────────
#  볼린저 밴드 기반 레짐 판단
# ──────────────────────────────────────────────
class BollingerBandRegime(BaseRegimeDetector):
    """볼린저 밴드(Bandwidth + %b) 기반 Market Regime 판단

    레짐 분류:
        State  0 (RANGE_BOUND)  : 박스권 / 평균회귀장
        State  1 (BREAKOUT_UP)  : 상승 돌파장
        State -1 (BREAKOUT_DOWN): 하락 돌파장
        State  2 (STRONG_UP)    : 강한 상승 추세 (지속적 %b ≥ 0.8)
        State -2 (STRONG_DOWN)  : 강한 하락 추세 (지속적 %b ≤ 0.2)

    Parameters:
        bb_period (int): 볼린저 밴드 이동평균 기간. 기본값 20.
        bb_std_mult (float): 표준편차 배수 (k). 기본값 2.0.
        bw_lookback (int): Bandwidth 최저치/평균 비교 기간 (M). 기본값 120.
        squeeze_percentile (float): 스퀴즈 판단 백분위수. 기본값 20.
        pctb_strong_upper (float): 강한 상승 추세 %b 임계값. 기본값 0.8.
        pctb_strong_lower (float): 강한 하락 추세 %b 임계값. 기본값 0.2.
        bw_slope_period (int): Bandwidth 기울기 계산 기간. 기본값 5.
        strong_trend_window (int): 강한 추세 지속 판단 윈도우. 기본값 10.
        strong_trend_ratio (float): 윈도우 내 임계 충족 비율. 기본값 0.7.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std_mult: float = 2.0,
        bw_lookback: int = 120,
        squeeze_percentile: float = 20.0,
        pctb_strong_upper: float = 0.8,
        pctb_strong_lower: float = 0.2,
        bw_slope_period: int = 5,
        strong_trend_window: int = 10,
        strong_trend_ratio: float = 0.7,
        name: str = "BollingerBandRegime",
    ):
        super().__init__(name=name)

        # ── 볼린저 밴드 기본 파라미터 ──
        self.bb_period = bb_period
        self.bb_std_mult = bb_std_mult

        # ── Bandwidth 레짐 파라미터 ──
        self.bw_lookback = bw_lookback
        self.squeeze_percentile = squeeze_percentile

        # ── %b 레짐 파라미터 ──
        self.pctb_strong_upper = pctb_strong_upper
        self.pctb_strong_lower = pctb_strong_lower

        # ── 결합 로직 파라미터 ──
        self.bw_slope_period = bw_slope_period
        self.strong_trend_window = strong_trend_window
        self.strong_trend_ratio = strong_trend_ratio

    def _validate_input(self, df: pd.DataFrame) -> None:
        """볼린저 밴드 계산에 필요한 최소 데이터 확인"""
        super()._validate_input(df)
        min_rows = self.bb_period + self.bw_lookback
        if len(df) < min_rows:
            raise ValueError(
                f"[{self.name}] 최소 {min_rows}행이 필요합니다 "
                f"(bb_period={self.bb_period} + bw_lookback={self.bw_lookback}). "
                f"현재: {len(df)}행"
            )

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """볼린저 밴드 지표 계산: MB, UB, LB, Bandwidth, %b, BW 기울기

        Returns:
            pd.DataFrame: 원본 + 계산된 지표 컬럼이 추가된 데이터프레임
        """
        result = df.copy()
        close = result["close"]

        # ── 볼린저 밴드 3요소 ──
        result["bb_mb"] = close.rolling(window=self.bb_period).mean()
        rolling_std = close.rolling(window=self.bb_period).std(ddof=0)
        result["bb_ub"] = result["bb_mb"] + self.bb_std_mult * rolling_std
        result["bb_lb"] = result["bb_mb"] - self.bb_std_mult * rolling_std

        # ── 1. Bandwidth (변동성 레짐) ──
        #   Bandwidth = (UB - LB) / MB
        result["bandwidth"] = (result["bb_ub"] - result["bb_lb"]) / result["bb_mb"]

        # Bandwidth 장기 통계 (rolling)
        result["bw_rolling_min"] = result["bandwidth"].rolling(
            window=self.bw_lookback, min_periods=self.bb_period
        ).min()
        result["bw_rolling_mean"] = result["bandwidth"].rolling(
            window=self.bw_lookback, min_periods=self.bb_period
        ).mean()
        result["bw_rolling_std"] = result["bandwidth"].rolling(
            window=self.bw_lookback, min_periods=self.bb_period
        ).std()
        result["bw_rolling_percentile"] = result["bandwidth"].rolling(
            window=self.bw_lookback, min_periods=self.bb_period
        ).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=False,
        )

        # Bandwidth 기울기 (1차 미분 근사: 단순 차분)
        #   d(Bandwidth)/dt ≈ BW[t] - BW[t - slope_period]
        result["bw_slope"] = result["bandwidth"].diff(self.bw_slope_period)

        # ── 2. %b (방향성 / 추세 레짐) ──
        #   %b = (Price - LB) / (UB - LB)
        band_width_abs = result["bb_ub"] - result["bb_lb"]
        result["pct_b"] = (close - result["bb_lb"]) / band_width_abs.replace(0, np.nan)

        # %b 이동평균 (강한 추세 지속 판단용)
        result["pct_b_ma"] = result["pct_b"].rolling(
            window=self.strong_trend_window, min_periods=1
        ).mean()

        # ── 스퀴즈 / 벌지 판단 플래그 ──
        result["is_squeeze"] = (
            result["bw_rolling_percentile"] <= self.squeeze_percentile
        )
        result["is_bulge"] = (
            result["bandwidth"]
            > result["bw_rolling_mean"] + 1.0 * result["bw_rolling_std"]
        )

        return result

    def _classify_regime(
        self, ind: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bandwidth + %b 결합 로직으로 레짐 분류

        로직 우선순위:
            1) 강한 추세 판단 (STRONG_UP / STRONG_DOWN)
               → %b가 지속적으로 극단에 머무는 경우
            2) 돌파 판단 (BREAKOUT_UP / BREAKOUT_DOWN)
               → BW 기울기 양수 + %b 밴드 이탈
            3) 그 외 → RANGE_BOUND (박스권)

        Returns:
            (regime, confidence, label) 튜플
        """
        n = len(ind)
        regime = pd.Series(RegimeState.RANGE_BOUND, index=ind.index, dtype=int)
        confidence = pd.Series(0.5, index=ind.index, dtype=float)
        label = pd.Series("박스권/평균회귀장", index=ind.index, dtype=str)

        pct_b = ind["pct_b"]
        bw_slope = ind["bw_slope"]
        bandwidth = ind["bandwidth"]
        bw_mean = ind["bw_rolling_mean"]
        is_squeeze = ind["is_squeeze"]
        is_bulge = ind["is_bulge"]

        # ── ① 강한 추세 판단 (rolling window 기반) ──
        #   최근 N봉 중 임계 비율 이상이 극단에 위치하면 강한 추세
        for i in range(self.strong_trend_window, n):
            window_slice = pct_b.iloc[i - self.strong_trend_window + 1: i + 1]
            valid_count = window_slice.notna().sum()
            if valid_count == 0:
                continue

            upper_ratio = (window_slice >= self.pctb_strong_upper).sum() / valid_count
            lower_ratio = (window_slice <= self.pctb_strong_lower).sum() / valid_count

            if upper_ratio >= self.strong_trend_ratio:
                regime.iloc[i] = RegimeState.STRONG_UP
                confidence.iloc[i] = min(upper_ratio, 1.0)
                label.iloc[i] = "강한 상승 추세"
            elif lower_ratio >= self.strong_trend_ratio:
                regime.iloc[i] = RegimeState.STRONG_DOWN
                confidence.iloc[i] = min(lower_ratio, 1.0)
                label.iloc[i] = "강한 하락 추세"

        # ── ② 돌파 판단 (BREAKOUT) ──
        #   조건: BW 기울기 > 0 (변동성 확대) + %b 밴드 이탈
        #   강한 추세로 이미 분류된 건 덮어쓰지 않음 (강한 추세가 우선)
        breakout_up_mask = (
            (bw_slope > 0)
            & (pct_b > 1.0)
            & (regime == RegimeState.RANGE_BOUND)
        )
        breakout_down_mask = (
            (bw_slope > 0)
            & (pct_b < 0.0)
            & (regime == RegimeState.RANGE_BOUND)
        )

        regime[breakout_up_mask] = RegimeState.BREAKOUT_UP
        confidence[breakout_up_mask] = np.clip(pct_b[breakout_up_mask], 0.6, 1.0)
        label[breakout_up_mask] = "상승 돌파장"

        regime[breakout_down_mask] = RegimeState.BREAKOUT_DOWN
        confidence[breakout_down_mask] = np.clip(
            1.0 - pct_b[breakout_down_mask], 0.6, 1.0
        )
        label[breakout_down_mask] = "하락 돌파장"

        # ── ③ 나머지: 박스권 세분화 ──
        #   스퀴즈 상태에서의 박스권 → 높은 확신
        range_mask = regime == RegimeState.RANGE_BOUND
        squeeze_range = range_mask & is_squeeze
        confidence[squeeze_range] = 0.8  # 스퀴즈 + 박스권 = 확실한 횡보

        # 벌지인데 돌파/추세에 해당하지 않는 경우 → 확신 낮음
        bulge_range = range_mask & is_bulge
        confidence[bulge_range] = 0.3

        # ── NaN 마스크 처리 ──
        nan_mask = pct_b.isna() | bw_slope.isna()
        regime[nan_mask] = RegimeState.RANGE_BOUND
        confidence[nan_mask] = 0.0
        label[nan_mask] = "데이터 부족"

        return regime, confidence, label

    def get_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 레짐만 별도로 반환 (Squeeze vs Bulge)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            pd.DataFrame: bandwidth, is_squeeze, is_bulge, vol_regime 컬럼
        """
        indicators = self._compute_indicators(df)
        vol = indicators[
            ["bandwidth", "bw_rolling_min", "bw_rolling_mean",
             "bw_rolling_percentile", "is_squeeze", "is_bulge"]
        ].copy()

        vol["vol_regime"] = "normal"
        vol.loc[indicators["is_squeeze"], "vol_regime"] = "squeeze"
        vol.loc[indicators["is_bulge"], "vol_regime"] = "bulge"

        return vol

    def get_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """방향성(%b) 레짐만 별도로 반환

        Args:
            df: OHLCV 데이터프레임

        Returns:
            pd.DataFrame: pct_b, pct_b_ma, trend_regime 컬럼
        """
        indicators = self._compute_indicators(df)
        trend = indicators[["pct_b", "pct_b_ma"]].copy()

        trend["trend_regime"] = "range"
        trend.loc[indicators["pct_b"] >= self.pctb_strong_upper, "trend_regime"] = (
            "strong_uptrend"
        )
        trend.loc[indicators["pct_b"] <= self.pctb_strong_lower, "trend_regime"] = (
            "strong_downtrend"
        )

        return trend

    @property
    def params(self) -> dict:
        """현재 파라미터 설정값 반환"""
        return {
            "bb_period": self.bb_period,
            "bb_std_mult": self.bb_std_mult,
            "bw_lookback": self.bw_lookback,
            "squeeze_percentile": self.squeeze_percentile,
            "pctb_strong_upper": self.pctb_strong_upper,
            "pctb_strong_lower": self.pctb_strong_lower,
            "bw_slope_period": self.bw_slope_period,
            "strong_trend_window": self.strong_trend_window,
            "strong_trend_ratio": self.strong_trend_ratio,
        }


# ──────────────────────────────────────────────
#  사용 예시 (모듈 직접 실행 시)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    # SPY 일봉 데이터 다운로드
    ticker = yf.Ticker("SPY")
    hist = ticker.history(period="10y")
    hist.columns = [c.lower() for c in hist.columns]  # 컬럼명 소문자

    # 레짐 탐지기 생성 & 실행
    detector = BollingerBandRegime(
        bb_period=20,
        bb_std_mult=2.0,
        bw_lookback=120,
    )
    result = detector.detect(hist)

    # 결과 출력
    print("=" * 60)
    print(f" {detector.name} — 레짐 분포 요약")
    print("=" * 60)
    print(detector.summary().to_string(index=False))

    print("\n최근 10일 레짐:")
    recent = pd.DataFrame({
        "date": hist.index[-10:],
        "close": hist["close"].iloc[-10:].values,
        "regime": result.regime.iloc[-10:].values,
        "label": result.label.iloc[-10:].values,
        "confidence": result.confidence.iloc[-10:].values,
        "%b": result.indicators["pct_b"].iloc[-10:].values,
        "bandwidth": result.indicators["bandwidth"].iloc[-10:].values,
    })
    print(recent.to_string(index=False))
    print(result.indicators)
    print(result.regime)
    
