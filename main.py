from lightweight_charts import Chart
from time import sleep
import pandas as pd

from backtester import backtest_strategy, TRADING_YR, update_yfinance_data

METRICS = {"worst_roi": {"column": "roi", "ascending": True},
           "best_roi": {"column": "roi", "ascending": False},
           "mid_roi": {"column": "roi", "ascending": False},
           "worst_sharpe": {"column": "sharpe", "ascending": True},
           "best_sharpe": {"column": "sharpe", "ascending": False},
           "mid_sharpe": {"column": "sharpe", "ascending": False},
           "worst_mdd": {"column": "mdd", "ascending": True},
           "best_mdd": {"column": "mdd", "ascending": False},
           "mid_mdd": {"column": "mdd", "ascending": False},
           }
INIT_METRIC = "worst_roi"


def clear_chart(chart: Chart):
    try:
        chart.clear_markers()
    except:
        print("HERE?")
    try:
        if hasattr(chart, 'box_obj'):
            chart.box_obj.delete()
            del chart.box_obj
    except:
        print("HERE?")


def get_results(bt_results: pd.DataFrame, metric: str, ticker: str):
    bt_results_sorted = bt_results.sort_values(METRICS[metric]['column'],
                                               ascending=METRICS[metric]['ascending'])
    metric_row = bt_results_sorted.iloc[0]
    if metric.split("_")[0] == 'mid':
        mean_val = bt_results[METRICS[metric]['column']].mean()
        idx_min = (bt_results[METRICS[metric]['column']] - mean_val).abs().idxmin()
        metric_row = bt_results.loc[idx_min]

    start_date = metric_row['start_date']
    end_date = metric_row['end_date']
    del metric_row

    df = update_yfinance_data(ticker=ticker, use_cache=True)
    df = df.loc[start_date:end_date]
    df = df.reset_index(drop=False)

    close_col = df.pop("close")
    df = df.rename(columns={'adj close': 'close'})
    df = backtest_strategy(df)

    df = df.rename(columns={'close': 'adj close'})
    df['close'] = close_col

    start_date = start_date.split(" ")[0]
    start_date = start_date.split("-")[1] + "/" + start_date.split("-")[2] + "/" + start_date.split("-")[0][2:]

    return df, start_date, bt_results_sorted


def reinit_chart(chart: Chart, metric: str, start_date, results_df: pd.DataFrame):  # , init_roi, init_mdd, init_sharpe):
    clear_chart(chart)

    init_roi = results_df['cumulative_roi'].iloc[0]
    init_mdd = results_df['rolling_max_drawdown'].iloc[0]
    init_sharpe = results_df['cumulative_sharpe'].iloc[0]

    chart.watermark(f'{chart.ticker.upper()}', color='rgba(180, 180, 240, 0.7)')

    chart.topbar['START_END'].set(f'{start_date} - {start_date}')
    chart.topbar['roi'].set(f'ROI {round(init_roi, 2)}')
    chart.topbar['mdd'].set(f'MDD {round(init_mdd, 2)}')
    chart.topbar['sharpe'].set(f'σ {round(init_sharpe, 2)}')

    chart.legend(visible=True, font_size=14)

    chart.set(results_df.head(1))
    # chart.fit()
    buy_marker = chart.marker(time=results_df.iloc[0]['date'], position='above', shape='arrow_down', color='rgba(0, 255, 0, 0.5)', text='BUY')
    results_df = results_df.iloc[1:]

    return chart, results_df


def zoom_out_event(chart: Chart):
    clear_chart(chart)

    # GET ALL DATA
    df = update_yfinance_data(ticker=chart.ticker, use_cache=True)
    df = df.reset_index(drop=False)
    close_col = df.pop("close")
    df = df.rename(columns={'adj close': 'close'})
    df = backtest_strategy(df)
    df = df.rename(columns={'close': 'adj close'})
    df['close'] = close_col
    chart.set(df=df, keep_drawings=False)
    chart.fit()
    results_df = chart.results_df
    box_obj = chart.box(results_df.iloc[0]['date'], results_df.iloc[0]['open'], results_df.iloc[-1]['date'], results_df.iloc[-1]['close'], color='white', width=4)
    chart.box_obj = box_obj


def init_chart(chart: Chart, ticker: str, metric: str, start_date, results_df: pd.DataFrame):  # , init_roi, init_mdd, init_sharpe):

    init_roi = results_df['cumulative_roi'].iloc[0]
    init_mdd = results_df['rolling_max_drawdown'].iloc[0]
    init_sharpe = results_df['cumulative_sharpe'].iloc[0]

    chart.ticker = ticker.upper()
    chart.topbar.button(name='search', button_text='search', func=search_click_event)
    chart.topbar.button(name='zoom_out', button_text='zoom out', func=zoom_out_event)
    chart.topbar.switcher(name='metric_switcher', options=('ROI', 'MDD', 'SHARPE'), default=INIT_METRIC.split("_")[1].upper(), func=metric_choice_event, align='left')
    chart.topbar.switcher(name='scenario_switcher', options=('WORST', 'MID', 'BEST'), default=INIT_METRIC.split("_")[0].upper(), func=metric_choice_event, align='left')
    chart.topbar.textbox(name="max_yrs_lbl", initial_text="max yrs:")
    chart.topbar.switcher(name='max_yrs', options=(1, 2, 3, 5, 10, 15), default='10', func=metric_choice_event, align='left')

    chart.topbar.textbox(name="start_yr_lbl", initial_text="start:")
    bt_results = pd.read_csv(f"{ticker}_backtest_results.csv")
    year_options = pd.to_datetime(bt_results.start_date).dt.year.unique()
    chart.topbar.switcher(name='start_yr', options=(1993, 2010, 2015, 2020, 2023, 2024), default=min(year_options), func=metric_choice_event, align='left')
    chart.watermark(f'{ticker.upper()}', color='rgba(180, 180, 240, 0.7)')

    chart.topbar.textbox('START_END', f'{start_date} - {start_date}')

    buy_marker = chart.marker(time=start_date, position='above', shape='arrow_down', color='rgba(0, 255, 0, 0.5)', text='BUY')

    chart.topbar.textbox('years', '9')

    chart.topbar.textbox('roi', f'ROI {round(init_roi, 2)}')
    chart.topbar.textbox('mdd', f'MDD {round(init_mdd, 2)}')
    chart.topbar.textbox('sharpe', f'σ {round(init_sharpe, 2)}')

    chart.legend(visible=True, font_size=14)

    chart.set(results_df.head(1))
    # chart.fit()
    results_df = results_df.iloc[1:]

    chart.show(block=False)

    return chart, results_df


def animate_chart(ticker: str, results_df: pd.DataFrame, metric: str, chart: Chart):
    for i, series in results_df.reset_index().iterrows():
        chart.update(series)

        start = chart.topbar['START_END'].value.split(" - ")[0]
        chart.topbar['START_END'].set(f'{start} - {series["date"].strftime("%m/%d/%y")}')

        curr_years = (i + 1) / TRADING_YR
        if curr_years == int(curr_years):
            chart.topbar['years'].set(f'YRS {int(curr_years)}')
        else:
            chart.topbar['years'].set(f'YRS {round(curr_years, 2)}')

        roi = round(series["cumulative_roi"], 2)
        mdd = round(series["rolling_max_drawdown"], 2)
        sharpe = round(series["cumulative_sharpe"], 2)
        chart.topbar['roi'].set(f'ROI {int(roi * 100)}%')
        chart.topbar['mdd'].set(f'MDD {int(mdd * 100)}%')
        chart.topbar['sharpe'].set(f'σ {sharpe}')
        # chart.fit()
        sleep(0.00)
    color = 'rgba(0, 255, 0, 0.5)'
    if roi < 0:
        color = 'red'
    end_marker = chart.marker(time=series['date'], position='above', shape='arrow_down', color=color, text=f'ROI \n {int(roi * 100)}%')


def search_click_event(chart: Chart):
    # remove marker
    chart.clear_markers()
    metric = chart.topbar['scenario_switcher'].value.lower() + "_" + chart.topbar['metric_switcher'].value.lower()
    max_yrs = int(chart.topbar['max_yrs'].value)
    start_yr = int(chart.topbar['start_yr'].value)

    ######## RESULTS

    bt_results = pd.read_csv(f"{chart.ticker}_backtest_results.csv")

    # convert start_date to datetime, remove rows where the year in start date is less than start_yr
    bt_results = bt_results[pd.to_datetime(bt_results['start_date']).dt.year >= start_yr]

    # filter results by max_yrs
    max_days = int(max_yrs) * TRADING_YR
    bt_results = bt_results[bt_results['holding_days'] <= max_days]
    results_df, start_date, bt_results_sorted = get_results(bt_results=bt_results, metric=metric, ticker=chart.ticker)
    chart.results_df = results_df

    chart, results_df = reinit_chart(chart=chart,
                                     metric=metric,
                                     start_date=start_date,
                                     results_df=results_df)

    ######## ANIMATE
    animate_chart(ticker=chart.ticker, results_df=results_df, metric=metric, chart=chart)


def metric_choice_event(chart: Chart):
    clear_chart(chart)

    start_yr = int(chart.topbar['start_yr'].value)
    ######## UPDATE OPTIONS
    curr_yr = pd.to_datetime('today').year
    years_passed_from_year_choice = curr_yr - start_yr + 1
    # update max year choices
    # chart.topbar['max_yrs'].options = tuple(range(1, years_passed_from_year_choice + 1))
    # update max year default
    # chart.topbar['max_yrs'].default = str(years_passed_from_year_choice)


def add_table(chart: Chart, bt_results_sorted: pd.DataFrame):
    def on_row_click(row):
        row['PL'] = round(row['PL'] + 1, 2)
        row.background_color('PL', 'green' if row['PL'] > 0 else 'red')
        backtest_results_table.footer[1] = row['Ticker']

    backtest_results_table = chart.create_table(width=0.3,
                                                height=0.2,
                                                headings=('yrs', 'start', 'end', 'roi', 'sharpe', 'mdd'),
                                                widths=(0.17, 0.17, 0.17, 0.17, 0.16, 0.16),
                                                alignments=('center', 'center', 'right', 'right', 'right', 'center'),
                                                position='right',
                                                func=on_row_click)

    period = bt_results_sorted.pop('holding_days')
    period = period // 252  # convert period to years integer divide by 252 trading days

    start = bt_results_sorted.pop('start_date')
    start = pd.to_datetime(start).dt.date
    end = bt_results_sorted.pop('end_date')
    end = pd.to_datetime(end).dt.date

    # add period, start, end back to beginning of results_df
    bt_results_sorted.insert(0, 'yrs', period)
    bt_results_sorted.insert(1, 'start', start)
    bt_results_sorted.insert(2, 'end', end)

    for i in range(bt_results_sorted.shape[0]):
        if i > 80:
            break
        backtest_results_table.new_row(*(bt_results_sorted.round(2).iloc[i].values.tolist()))
    backtest_results_table.footer(2)
    backtest_results_table.footer[0] = 'Selected:'


def viz_chart(ticker: str, metric: str):
    # INST CHART 16:9
    chart = Chart(width=1230, height=692,
                  inner_width=1.0, inner_height=1.0,
                  x=0, y=0,
                  title='',
                  screen=None,
                  on_top=False,
                  toolbox=True,
                  scale_candles_only=False,
                  position='left')  # maximize=True, debug=False,

    # BT RESULTS
    bt_results = pd.read_csv(f"{ticker}_backtest_results.csv")
    results_df, start_date, bt_results_sorted = get_results(bt_results=bt_results, metric=metric, ticker=ticker)
    chart.results_df = results_df

    # INIT CHART
    chart, results_df = init_chart(chart=chart, ticker=ticker, metric=metric, start_date=start_date, results_df=results_df)

    # ADD TABLE
    # add_table(chart=chart, bt_results_sorted=bt_results_sorted)

    # ANIMATE CHART
    animate_chart(ticker=ticker, results_df=results_df, metric=metric, chart=chart)

    chart.show(block=True)


if __name__ == "__main__":
    viz_chart(ticker='SPY', metric=INIT_METRIC)
