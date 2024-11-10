TRADING_YR = 252
window_addend = TRADING_YR // 2
daily_risk_free_rate = 0.000016745000791186

def get_data(tickers: str = None, use_polars: bool = True):
    import yfinance as yf
    if use_polars:
        import polars as pl
    else:
        import pandas as pd

    df = yf.download(tickers,
                     period='max',
                     interval='1d',
                     # start="2020-12-01",
                     # end="2023-12-31",
                     group_by="ticker",
                     back_adjust=True,
                     progress=False)

    df = df.stack(level=0, future_stack=True).reset_index()
    df.columns = [col.lower() for col in df.columns]
    df = df.dropna()
    df = df[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"].dt.date)
    df = df.sort_values(["date"], ascending=[True])
    df = df.reset_index(drop=True)

    if use_polars:
        df = pl.from_pandas(df)

    return df
