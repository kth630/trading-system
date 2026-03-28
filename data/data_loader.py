import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, tickers, start_date, end_date=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.daily_data = None
        self.monthly_data = None

    def fetch_data(self):
        print(f"[{','.join(self.tickers)}] 종목 가격 데이터 다운로드 중...")
        try:
            df = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        except Exception as e:
            df = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
            
        df.dropna(inplace=True)
        self.daily_data = df
        print("데이터 다운로드 완료!")
        return df

    def get_daily_data(self):
        """레짐 감지를 위한 일간 데이터 반환"""
        if self.daily_data is None:
            self.fetch_data()
        return self.daily_data

    def get_monthly_data(self):
        """자산 배분을 위한 월말 데이터 반환 (Resampling)"""
        if self.daily_data is None:
            self.fetch_data()
        
        print("데이터 월별 리샘플링 작업 중...")
        self.monthly_data = self.daily_data.resample('ME').last()
        return self.monthly_data
