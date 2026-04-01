"""
시장 데이터 관리 모듈 — 로컬 캐싱 + 증분 업데이트

전략:
    1. 로컬 캐시(Parquet) 확인
    2. 캐시 없으면 → 전체 다운로드 후 저장
    3. 캐시 있으면 → 마지막 날짜 이후만 API로 받아서 이어붙이기
    4. 강제 새로고침 옵션 지원

저장 경로: data/cache/{ticker}_{interval}.parquet
    예) data/cache/SPY_1d.parquet, data/cache/AAPL_1h.parquet

컬럼 규칙: 모든 컬럼명은 소문자로 정규화
    open, high, low, close, volume (+ dividends, stock splits 등)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── 기본 설정 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"


class MarketDataManager:
    """시장 데이터 캐시 관리자

    사용법:
        dm = MarketDataManager()

        # 기본: 캐시 우선, 없으면 다운로드
        spy = dm.load("SPY", period="10y")

        # 여러 종목 한 번에
        data = dm.load_multiple(["SPY", "QQQ", "IWM"], period="5y")

        # 강제 새로고침
        spy = dm.load("SPY", period="10y", refresh=True)

        # 캐시 정보 확인
        dm.cache_info("SPY")

    Parameters:
        cache_dir: 캐시 저장 디렉토리. 기본값은 data/cache/
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, interval: str = "1d") -> Path:
        """캐시 파일 경로 생성"""
        safe_name = ticker.upper().replace("^", "_").replace("/", "_")
        return self.cache_dir / f"{safe_name}_{interval}.parquet"

    def load(
        self,
        ticker: str,
        period: str = "10y",
        interval: str = "1d",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """데이터 로드 (캐시 우선 → 증분 업데이트)

        Args:
            ticker: 종목 심볼 (예: "SPY", "AAPL", "^GSPC")
            period: 최초 다운로드 시 기간 (예: "1y", "5y", "10y", "max")
            interval: 봉 간격 (예: "1d", "1h", "5m")
            refresh: True이면 캐시 무시, 전체 재다운로드

        Returns:
            pd.DataFrame: OHLCV 데이터 (컬럼: open, high, low, close, volume, ...)
        """
        cache_file = self._cache_path(ticker, interval)

        # ── Case 1: 강제 새로고침 ──
        if refresh:
            logger.info(f"[{ticker}] 강제 새로고침 — 전체 다운로드")
            df = self._download(ticker, period=period, interval=interval)
            self._save_cache(df, cache_file)
            return df

        # ── Case 2: 캐시 없음 → 전체 다운로드 ──
        if not cache_file.exists():
            logger.info(f"[{ticker}] 캐시 없음 — 전체 다운로드 (period={period})")
            df = self._download(ticker, period=period, interval=interval)
            self._save_cache(df, cache_file)
            return df

        # ── Case 3: 캐시 있음 → 증분 업데이트 ──
        cached = self._load_cache(cache_file)
        last_date = cached.index[-1]

        # 당일 또는 최근 장 마감 이후라면 → 캐시 그대로 반환
        now = datetime.now(tz=last_date.tzinfo) if last_date.tzinfo else datetime.now()
        days_behind = (now - last_date).days

        if days_behind <= 1:
            logger.info(
                f"[{ticker}] 캐시 최신 (마지막: {last_date.strftime('%Y-%m-%d')})"
            )
            return cached

        # 증분 다운로드: 마지막 날짜 다음날부터
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(
            f"[{ticker}] 증분 업데이트: {last_date.strftime('%Y-%m-%d')} → 현재"
        )

        try:
            new_data = self._download(
                ticker, start=start_date, interval=interval
            )
        except Exception as e:
            logger.warning(
                f"[{ticker}] 증분 다운로드 실패 ({e}), 캐시 데이터 반환"
            )
            return cached

        if new_data.empty:
            logger.info(f"[{ticker}] 새 데이터 없음, 캐시 그대로 반환")
            return cached

        # 이어붙이기 (중복 날짜 제거)
        combined = pd.concat([cached, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        self._save_cache(combined, cache_file)
        logger.info(
            f"[{ticker}] 업데이트 완료: +{len(new_data)}행 "
            f"(총 {len(combined)}행)"
        )
        return combined

    def load_multiple(
        self,
        tickers: list[str],
        period: str = "10y",
        interval: str = "1d",
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """여러 종목 데이터를 한 번에 로드

        Returns:
            dict: {ticker: DataFrame} 딕셔너리
        """
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.load(
                    ticker, period=period, interval=interval, refresh=refresh
                )
            except Exception as e:
                logger.error(f"[{ticker}] 로드 실패: {e}")
        return result

    def cache_info(self, ticker: str = "", interval: str = "1d") -> pd.DataFrame:
        """캐시 상태 정보 조회

        Args:
            ticker: 특정 종목만 조회. 비워두면 전체 캐시 목록.

        Returns:
            pd.DataFrame: 캐시 파일 정보 (종목, 기간, 행수, 파일크기)
        """
        if ticker:
            files = [self._cache_path(ticker, interval)]
        else:
            files = sorted(self.cache_dir.glob("*.parquet"))

        rows = []
        for f in files:
            if not f.exists():
                continue
            try:
                df = pd.read_parquet(f)
                rows.append({
                    "file": f.name,
                    "ticker": f.stem.rsplit("_", 1)[0],
                    "interval": f.stem.rsplit("_", 1)[1] if "_" in f.stem else "?",
                    "rows": len(df),
                    "start": df.index[0].strftime("%Y-%m-%d"),
                    "end": df.index[-1].strftime("%Y-%m-%d"),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
            except Exception as e:
                rows.append({
                    "file": f.name,
                    "ticker": f.stem,
                    "interval": "?",
                    "rows": 0,
                    "start": "ERROR",
                    "end": str(e),
                    "size_kb": 0,
                })

        if not rows:
            print("캐시 파일 없음")
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def clear_cache(self, ticker: str = "", interval: str = "1d") -> None:
        """캐시 삭제

        Args:
            ticker: 특정 종목만 삭제. 비워두면 전체 캐시 삭제.
        """
        if ticker:
            path = self._cache_path(ticker, interval)
            if path.exists():
                path.unlink()
                logger.info(f"캐시 삭제: {path.name}")
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            logger.info("전체 캐시 삭제 완료")

    # ── 내부 메서드 ──

    def _download(
        self,
        ticker: str,
        period: str = "",
        start: str = "",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """yfinance API 호출 래퍼

        Args:
            ticker: 종목 심볼
            period: 기간 (period 또는 start 중 하나)
            start: 시작 날짜 (YYYY-MM-DD)
            interval: 봉 간격

        Returns:
            pd.DataFrame: OHLCV 데이터 (소문자 컬럼)
        """
        t = yf.Ticker(ticker)

        if start:
            df = t.history(start=start, interval=interval)
        else:
            df = t.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"[{ticker}] 다운로드된 데이터 없음")
            return pd.DataFrame()

        # 컬럼명 소문자 정규화
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # 불필요한 컬럼 정리 (있으면)
        drop_cols = [c for c in ["dividends", "stock_splits"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        logger.info(
            f"[{ticker}] 다운로드 완료: {len(df)}행 "
            f"({df.index[0].strftime('%Y-%m-%d')} "
            f"~ {df.index[-1].strftime('%Y-%m-%d')})"
        )
        return df

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Parquet 형식으로 캐시 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
        size_kb = round(path.stat().st_size / 1024, 1)
        logger.info(f"캐시 저장: {path.name} ({size_kb} KB)")

    def _load_cache(self, path: Path) -> pd.DataFrame:
        """Parquet 캐시 로드"""
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(
            f"캐시 로드: {path.name} ({len(df)}행, "
            f"{df.index[0].strftime('%Y-%m-%d')} "
            f"~ {df.index[-1].strftime('%Y-%m-%d')})"
        )
        return df

    def __repr__(self) -> str:
        n_files = len(list(self.cache_dir.glob("*.parquet")))
        return f"MarketDataManager(cache_dir='{self.cache_dir}', files={n_files})"


# ──────────────────────────────────────────────
#  편의 함수 (모듈 레벨)
# ──────────────────────────────────────────────
_default_manager: Optional[MarketDataManager] = None


def get_manager() -> MarketDataManager:
    """싱글턴 매니저 반환"""
    global _default_manager
    if _default_manager is None:
        _default_manager = MarketDataManager()
    return _default_manager


def load(
    ticker: str,
    period: str = "10y",
    interval: str = "1d",
    refresh: bool = False,
) -> pd.DataFrame:
    """편의 함수: 기본 매니저로 데이터 로드

    사용법:
        from data.data import load
        spy = load("SPY")
        spy = load("SPY", refresh=True)  # 강제 새로고침
    """
    return get_manager().load(ticker, period, interval, refresh)


# ──────────────────────────────────────────────
#  직접 실행 시 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    dm = MarketDataManager()
    print(f"\n{dm}\n")

    # ── 1차 호출: API → 캐시 생성 ──
    print("=" * 50)
    print(" 1차 호출 (캐시 생성)")
    print("=" * 50)
    spy = dm.load("SPY", period="10y")
    print(f"  shape: {spy.shape}")
    print(f"  range: {spy.index[0].strftime('%Y-%m-%d')} ~ "
          f"{spy.index[-1].strftime('%Y-%m-%d')}")
    print(f"  columns: {list(spy.columns)}")
    print(spy.tail(3))

    # ── 2차 호출: 캐시에서 로드 (빠름) ──
    print("\n" + "=" * 50)
    print(" 2차 호출 (캐시에서 로드)")
    print("=" * 50)
    spy2 = dm.load("SPY", period="10y")
    print(f"  shape: {spy2.shape}")

    # ── 캐시 정보 ──
    print("\n" + "=" * 50)
    print(" 캐시 정보")
    print("=" * 50)
    print(dm.cache_info().to_string(index=False))
