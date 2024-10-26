from lightweight_charts import Chart
from os.path import exists
import yfinance as yf
from time import sleep
from tqdm import tqdm
import pandas as pd

TRADING_YR = 252
METRICS = {"worst_roi": {"column": "roi", "ascending": True},
           "best_roi": {"column": "roi", "ascending": False},
           "median_roi": {"column": "roi", "ascending": False},
           "worst_sharpe": {"column": "sharpe", "ascending": True},
           "best_sharpe": {"column": "sharpe", "ascending": False},
           "median_sharpe": {"column": "sharpe", "ascending": False},
           "worst_mdd": {"column": "max_drawdown", "ascending": True},
           "best_mdd": {"column": "max_drawdown", "ascending": False},
           "median_mdd": {"column": "max_drawdown", "ascending": False},
           }


def backtest_strategy(df: pd.DataFrame):
    df.loc[:, 'daily_return'] = df['close'].pct_change()
    df.loc[:, 'cumulative_roi'] = (1 + df['daily_return']).cumprod() - 1
    df.loc[:, 'rolling_max'] = df['close'].cummax()
    df.loc[:, 'rolling_max_drawdown'] = ((df['close'] - df['rolling_max']) / df['rolling_max']).cummin()
    df.loc[:, 'cumulative_sharpe'] = df['daily_return'].expanding().mean() / df['daily_return'].expanding().std()
    return df


def run_backtest(ticker: str = None):
    window_size_addend = 252
    max_history = TRADING_YR * 10

    # 1/29/93 - 10/24/24
    df = get_yfinance_data(ticker=ticker, use_cache=True).head(max_history)
    df = df[['adj close']]
    df = df.rename(columns={'adj close': 'close'})
    df = df.reset_index(drop=False)

    backtest_results = []

    for holding_days in tqdm(range(TRADING_YR, len(df), window_size_addend)):

        if holding_days > len(df):
            continue

        # convolve the backtest_strategy with the holding period
        for i in range(holding_days, len(df) + 1):
            window_df = df.iloc[i - holding_days:i].copy()
            window_df = backtest_strategy(window_df)
            backtest_results.append({
                "holding_days": holding_days,
                "start_date": window_df['date'].iloc[0],
                "end_date": window_df['date'].iloc[-1],
                "roi": window_df['cumulative_roi'].iloc[-1],
                "sharpe": window_df['cumulative_sharpe'].iloc[-1],
                "max_drawdown": window_df['rolling_max_drawdown'].min()
            })

    backtest_results_df = pd.DataFrame(backtest_results)
    backtest_results_df.to_csv(f"{ticker}_backtest_results.csv", index=False)
    return backtest_results_df


def get_yfinance_data(ticker: str = None, use_cache=True):
    if exists(f"{ticker}.csv") and use_cache:
        df = pd.read_csv(f"{ticker}.csv", index_col="date")
        df.index = pd.to_datetime(df.index)
        return df

    df = yf.download(tickers=[ticker], start=None, end=None)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()
    df.index.name = 'date'

    df.to_csv(f"{ticker}.csv", index=True)

    df = pd.read_csv(f"{ticker}.csv", index_col="date")
    df.index = pd.to_datetime(df.index)

    return df


def viz_chart(ticker: str, metric: str, bt_results: pd.DataFrame):
    worst_roi = bt_results.sort_values(METRICS[metric]['column'], ascending=METRICS[metric]['ascending'])
    worst_roi_start_date = worst_roi['start_date'].values[0]
    worst_roi_end_date = worst_roi['end_date'].values[0]
    del worst_roi

    worst_roi_df = get_yfinance_data(ticker=ticker, use_cache=True)
    worst_roi_df = worst_roi_df.loc[worst_roi_start_date:worst_roi_end_date]

    worst_roi_df = backtest_strategy(worst_roi_df)

    chart = Chart()

    init_roi = worst_roi_df['cumulative_roi'].iloc[0]
    init_mdd = worst_roi_df['rolling_max_drawdown'].iloc[0]
    init_sharpe = worst_roi_df['cumulative_sharpe'].iloc[0]

    chart.watermark(f'SPY - {" ".join(metric.upper().split("_"))}', color='rgba(180, 180, 240, 0.7)')

    worst_roi_start_date = worst_roi_start_date.split(" ")[0]
    worst_roi_start_date = worst_roi_start_date.split("-")[1] + "/" + worst_roi_start_date.split("-")[2] + "/" + worst_roi_start_date.split("-")[0][2:]
    chart.topbar.textbox('START', f'START : {worst_roi_start_date}')
    chart.topbar.textbox('CURR', f'CURR : {worst_roi_start_date}')
    # number of days
    chart.topbar.textbox('n_days', f'N DAYS : 1')
    chart.topbar.textbox('roi', f'ROI : {round(init_roi, 2)}')
    chart.topbar.textbox('mdd', f'MDD : {round(init_mdd, 2)}')
    chart.topbar.textbox('sharpe', f'SHARPE : {round(init_sharpe, 2)}')
    chart.legend(visible=True, font_size=14)

    chart.set(worst_roi_df.head(1))
    worst_roi_df = worst_roi_df.iloc[1:]

    chart.show(block=False)

    for i, series in worst_roi_df.reset_index().iterrows():
        chart.update(series)

        chart.topbar['CURR'].set(f'CURR : {series["date"].strftime("%m/%d/%y")}')
        chart.topbar['n_days'].set(f'N DAYS : {i + 2}')

        chart.topbar['roi'].set(f'ROI : {round(series["cumulative_roi"], 2)}')
        chart.topbar['mdd'].set(f'MDD : {round(series["rolling_max_drawdown"], 2)}')
        chart.topbar['sharpe'].set(f'SHARPE : {round(series["cumulative_sharpe"], 2)}')

        sleep(0.1)

    chart.show(block=True)


def run():
    ticker = 'SPY'
    # backtest_results_df = run_backtest(ticker=ticker)
    backtest_results_df = pd.read_csv(f"{ticker}_backtest_results.csv")

    metric = "worst_roi"
    viz_chart(ticker=ticker, metric="best_roi", bt_results=backtest_results_df)


if __name__ == "__main__":
    run()
