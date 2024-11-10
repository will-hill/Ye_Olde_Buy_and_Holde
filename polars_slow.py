from line_profiler_pycharm import profile
import polars as pl
from tqdm import tqdm
from equities_util import get_data, TRADING_YR, window_addend, daily_risk_free_rate



def create_backtest_windows_indices(df):
    investment_window_sizes = list(range(TRADING_YR, len(df), window_addend))
    backtest_windows_indices = []

    for investment_window_size in investment_window_sizes:

        # loop over day ranges in entire history
        for i in range(0, len(df) - investment_window_size + 1):
            start_idx = i
            end_idx = i + investment_window_size
            backtest_windows_indices.append((start_idx, end_idx, investment_window_size))

    return backtest_windows_indices


@profile
def run_backtests(df, backtest_windows_indices):
    backtest_results = []

    for slice_params in tqdm(backtest_windows_indices):
        start_idx = slice_params[0]
        end_idx = slice_params[1]
        holding_period = slice_params[2]

        last_row = (
            df.slice(start_idx, end_idx)
            .with_columns(

                # cumulative returns
                ((pl.col("close").pct_change() + 1).cum_prod() - 1).alias("cumulative_returns"),

                # max drawdown
                (
                        (pl.col('close') - pl.col('close').cum_max()) / pl.col('close').cum_max()
                ).cum_min().alias("max_drawdown"),

                # Sharpe ratio
                (
                        (pl.col("close").pct_change() - daily_risk_free_rate).cumulative_eval(pl.element().mean()) /
                        (pl.col("close").pct_change()).cumulative_eval(pl.element().std())
                ).alias("sharpe")
            )
        ).select(pl.last("date", "cumulative_returns", "max_drawdown", "sharpe")).to_dicts()[0]

        last_row["holding_days"] = holding_period
        backtest_results.append(last_row)
    return backtest_results


def run():
    df = get_data(tickers='SPY')
    backtest_windows_indices = create_backtest_windows_indices(df)
    backtest_results = run_backtests(df, backtest_windows_indices)
    import pandas as pd
    pd.DataFrame(backtest_results).to_csv("backtest_results.csv", index=False)


if __name__ == "__main__":
    run()
