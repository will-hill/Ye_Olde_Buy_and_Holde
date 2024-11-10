from datetime import datetime
from os.path import exists
import yfinance as yf
from tqdm import tqdm
import pandas as pd

TRADING_YR = 252


def update_yfinance_data(ticker: str = None, use_cache: bool = False):
    if not use_cache:
        # get latest data
        df = yf.download(tickers=[ticker], start=None, end=None)
        # append to existing data
        df.to_csv(f"{ticker}.csv", index=True, mode='a', header=not exists(f"{ticker}.csv"))

        # dedupe & save
        df = pd.read_csv(f"{ticker}.csv", index_col="date")
        df = df.sort_values(by='date')
        df = df[~df.index.duplicated(keep='first')]
        df.to_csv(f"{ticker}.csv", index=True, mode='w', header=True)

    df = pd.read_csv(f"{ticker}.csv", index_col="date")

    # transform for use
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)

    return df


def backtest_strategy(df: pd.DataFrame):
    df.loc[:, 'daily_return'] = df['close'].pct_change()
    df.loc[:, 'cumulative_roi'] = (1 + df['daily_return']).cumprod() - 1
    df.loc[:, 'rolling_max'] = df['close'].cummax()
    df.loc[:, 'rolling_max_drawdown'] = ((df['close'] - df['rolling_max']) / df['rolling_max']).cummin()
    df.loc[:, 'cumulative_sharpe'] = df['daily_return'].expanding().mean() / df['daily_return'].expanding().std()
    return df


def run_backtest(ticker: str = None):
    window_size_addend = 252 // 2

    df = update_yfinance_data(ticker=ticker)
    df.index = pd.to_datetime(df.index)
    df = df[['adj close']]
    df = df.rename(columns={'adj close': 'close'})
    df = df.reset_index(drop=False)

    backtest_results = []
    for holding_days in tqdm(range(TRADING_YR, len(df), window_size_addend)):
        if holding_days > len(df):
            continue

        for i in range(holding_days, len(df) + 1):

            window_df = df.iloc[i - holding_days:i].copy()
            window_df = backtest_strategy(window_df)
            backtest_results.append({
                "holding_days": holding_days,
                "start_date": window_df['date'].iloc[0],
                "end_date": window_df['date'].iloc[-1],
                "roi": window_df['cumulative_roi'].iloc[-1],
                "sharpe": window_df['cumulative_sharpe'].iloc[-1],
                "mdd": window_df['rolling_max_drawdown'].min()
            })

    backtest_results_df = pd.DataFrame(backtest_results)
    backtest_results_df.to_csv(f"{ticker}_backtest_results.{datetime.now().strftime('%Y-%m-%dT%H:%M')}.csv", index=False, mode='a', header=not exists(f"{ticker}_backtest_results.csv"))
    return backtest_results_df


if __name__ == "__main__":
    run_backtest(ticker='SPY')
