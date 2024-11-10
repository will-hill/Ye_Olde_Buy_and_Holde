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


def run_backtests(df, backtest_windows_indices):
    backtest_results = []

    for slice_params in tqdm(backtest_windows_indices):
        start_idx = slice_params[0]
        end_idx = slice_params[1]
        holding_period = slice_params[2]

        window_df = df.iloc[start_idx: end_idx].copy()
        # roi
        window_df["cumulative_returns"] = (1 + window_df['close'].pct_change()).cumprod() - 1
        # mdd
        window_df['rolling_max_drawdown'] = ((window_df['close'] - (window_df['close'].cummax())) / window_df['close'].cummax()).cummin()
        # sharpe
        window_df['cumulative_sharpe'] = (window_df['close'].pct_change() - daily_risk_free_rate).expanding().mean() / window_df['close'].pct_change().expanding().std()

        # last row as dict
        result = window_df.tail(1).to_dict()
        result['holding_days'] = holding_period

        backtest_results.append(result)

    return backtest_results


def run():
    df = get_data(tickers='SPY')
    backtest_windows_indices = create_backtest_windows_indices(df)
    backtest_results = run_backtests(df, backtest_windows_indices)
    import pandas as pd
    pd.DataFrame(backtest_results).to_csv("backtest_results.csv", index=False)


if __name__ == "__main__":
    run()
