import polars as pl
from tqdm import tqdm
import yfinance as yf


def get_stocks_data(tickers: [] = None):
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
    df = df[["date", "ticker", "close"]]
    df = df.sort_values(["ticker", "date"], ascending=[True, True])
    return df


def run_backtests(df, risk_free_rate: float = None):
    df = (
        pl.from_pandas(df)
        .with_columns(
            pl.col("date").dt.date(),
            pl.col("close").pct_change().alias("pct_change"))
        .with_columns(
            (pl.col("pct_change") + 1).alias("ret_pls_one")
        ))

    for i in tqdm(range(0, len(df))):
        df = df.join(
            df.tail(len(df) - i).with_columns(
                (pl.col("ret_pls_one").cum_prod() - 1).alias(f"cumulative_roi_{i}"),
                (pl.col("close") - (pl.col("close").cum_max())).alias(f"rolling_max_drawdown_{i}"),
                ((pl.col("pct_change") - risk_free_rate).cumulative_eval(pl.element().mean())
                 / (pl.col("pct_change") - risk_free_rate).cumulative_eval(pl.element().std())
                 ).alias(f"sharpe_ratio_{i}")
            ).select(["date", f"cumulative_roi_{i}", f"rolling_max_drawdown_{i}", f"sharpe_ratio_{i}"])
            , on=["date"], how="left")

    df.write_csv("df_scored.csv")
    return df


def run():
    df = get_stocks_data(tickers=['SPY', 'QQQ', 'IWM', 'TQQQ'])
    bt_df = run_backtests(df=df, risk_free_rate=0.02)


if __name__ == "__main__":
    run()
