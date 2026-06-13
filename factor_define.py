"""
MFML 项目因子定义模块

每个函数完全独立，内部计算所需所有统计量
"""

import polars as pl
from vnpy.trader.constant import FactorType
FUNDAMENTAL = FactorType.FUNDAMENTAL
PRICE_AND_VOLUME = FactorType.PRICE_AND_VOLUME


def get_time_bucket_expr(time_col):
    """将时间映射到半小时桶"""
    return (
        pl.when((time_col >= pl.time(9, 30)) & (time_col < pl.time(10, 0))).then(1)
        .when((time_col >= pl.time(10, 0)) & (time_col < pl.time(10, 30))).then(2)
        .when((time_col >= pl.time(10, 30)) & (time_col < pl.time(11, 0))).then(3)
        .when((time_col >= pl.time(11, 0)) & (time_col < pl.time(11, 30))).then(4)
        .when((time_col >= pl.time(13, 0)) & (time_col < pl.time(13, 30))).then(5)
        .when((time_col >= pl.time(13, 30)) & (time_col < pl.time(14, 0))).then(6)
        .when((time_col >= pl.time(14, 0)) & (time_col < pl.time(14, 30))).then(7)
        .when((time_col >= pl.time(14, 30)) & (time_col <= pl.time(15, 0))).then(8)
        .otherwise(None)
        .alias('time_bucket')
    )


# ============================================================================
# 因子1: late_skew_ret (尾盘收益率偏度)
# ============================================================================
def calc_late_skew_ret(df_minute):
    """
    尾盘(14:30-15:00)分钟收益率的偏度
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close 的 DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    # 内部计算分钟收益率和时间
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    df = df.with_columns([
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', 'date'])) - 1).alias('min_ret')
    ])
    df = df.with_columns([get_time_bucket_expr(pl.col('time'))])
    
    late_data = df.filter(pl.col('time_bucket') == 8)
    result = late_data.group_by(['vt_symbol', 'date']).agg([
        pl.col('min_ret').mean().alias('mean_ret'),
        pl.col('min_ret').std().alias('std_ret'),
        ((pl.col('min_ret') - pl.col('min_ret').mean())**3).mean().alias('moment3')
    ]).with_columns([
        (pl.col('moment3') / (pl.col('std_ret')**3 + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 因子2: down_vol_perc (下行收益率波动占比)
# ============================================================================
def calc_down_vol_perc(df_minute):
    """
    下行分钟收益率的方差占总方差的比例
    
    Args:
        df_minute: 包含 vt_symbol, datetime, close 的 DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    # 内部计算收益率和日期
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', 'date'])) - 1).alias('min_ret')
    ])

    # 每日统计量
    daily_stats = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('min_ret').std().alias('daily_std_ret')
    ])
    
    down_vol_data = df.filter(pl.col('min_ret') < 0)
    down_vol = down_vol_data.group_by(['vt_symbol', 'date']).agg([
        ((pl.col('min_ret') - pl.col('min_ret').mean())**2).mean().alias('down_var')
    ])
    
    result = daily_stats.join(down_vol, on=['vt_symbol', 'date'], how='left')
    result = result.with_columns([
        (pl.col('down_var') / (pl.col('daily_std_ret')**2 + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 因子3: corr_ret_lastret (前后两分钟收益率相关性)
# ============================================================================
def calc_corr_ret_lastret(df_minute):
    """
    当前分钟收益率与前一分钟收益率的相关性
    
    Args:
        df_minute: 包含 vt_symbol, datetime, close 的 DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    # 内部计算收益率和日期
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', 'date'])) - 1).alias('min_ret')
    ])

    df_lag = df.with_columns([
        pl.col('min_ret').shift(1).over(['vt_symbol', 'date']).alias('min_ret_lag1')
    ]).filter(pl.col('min_ret_lag1').is_not_null())

    result = df_lag.group_by(['vt_symbol', 'date']).agg([
        pl.corr('min_ret', 'min_ret_lag1').alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 因子4: corr_close_nextopen (前一分钟收盘价与后一分钟开盘价相关性)
# ============================================================================
def calc_corr_close_nextopen(df_minute):
    """
    前一分钟收盘价与当前分钟开盘价的相关性
    
    Args:
        df_minute: 包含 vt_symbol, datetime, close, open 的 DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    
    df_close_lag = df.with_columns([
        pl.col('close').shift(1).over(['vt_symbol', 'date']).alias('close_lag1')
    ]).filter(pl.col('close_lag1').is_not_null())
    
    result = df_close_lag.group_by(['vt_symbol', 'date']).agg([
        pl.corr('close_lag1', 'open').alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 因子5-10: volume_perc2-7 (各半小时成交量占比)
# ============================================================================
def calc_volume_perc(df_minute, bucket):
    """
    指定时间桶的成交量占全天成交量的比例
    
    Args:
        df_minute: 包含 datetime, volume 的 DataFrame
        bucket: 时间桶编号 (2-7)
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    # 内部计算日期和时间桶
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    df = df.with_columns([get_time_bucket_expr(pl.col('time'))])
    
    # 每日总成交量
    daily_volume = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').sum().alias('total_volume')
    ])
    
    bucket_vol = df.filter(pl.col('time_bucket') == bucket).group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').sum().alias('bucket_volume')
    ])
    
    result = daily_volume.join(bucket_vol, on=['vt_symbol', 'date'], how='left')
    result = result.with_columns([
        (pl.col('bucket_volume') / (pl.col('total_volume') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_volume_perc2(df_minute):
    """10:00-10:30 成交量占比"""
    df = df_minute
    return calc_volume_perc(df, 2)


def calc_volume_perc3(df_minute):
    """10:30-11:00 成交量占比"""
    df = df_minute
    return calc_volume_perc(df, 3)


def calc_volume_perc4(df_minute):
    """11:00-11:30 成交量占比"""
    df = df_minute
    return calc_volume_perc(df, 4)


def calc_volume_perc5(df_minute):
    """13:00-13:30 成交量占比"""
    df = df_minute
    return calc_volume_perc(df, 5)


def calc_volume_perc6(df_minute):
    """13:30-14:00 成交量占比"""
    df = df_minute
    return calc_volume_perc(df, 6)


def calc_volume_perc7(df_minute):
    """14:00-14:30 成交量占比"""
    df = df_minute
    return calc_volume_perc(df, 7)


# ============================================================================
# 因子11: early_corr_volume_ret (早盘成交量与收益率相关性)
# ============================================================================
def calc_early_corr_volume_ret(df_minute):
    """
    早盘(9:30-11:00)成交量与收益率的相关性
    
    Args:
        df_minute: 包含 datetime, close, volume 的 DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    # 内部计算收益率、日期、时间桶
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    df = df.with_columns([
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', 'date'])) - 1).alias('min_ret')
    ])
    df = df.with_columns([get_time_bucket_expr(pl.col('time'))])
    
    early_data = df.filter(pl.col('time_bucket').is_in([1, 2]))
    result = early_data.group_by(['vt_symbol', 'date']).agg([
        pl.corr('min_ret', 'volume').alias('data')
    ]).with_columns([pl.col('data')])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 因子12: corr_volume_amplitude (成交量与振幅相关性)
# ============================================================================
def calc_corr_volume_amplitude(df_minute):
    """
    分钟成交量与振幅的相关性
    
    Args:
        df_minute: 包含 datetime, high, low, volume 的 DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    # 内部计算日期和振幅
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('high') - pl.col('low')) / pl.col('low')).alias('amplitude')
    ])
    
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.corr('amplitude', 'volume').alias('data')
    ]).with_columns([pl.col('data')])
    return result.select(['vt_symbol', 'date', 'data'])




# ============================================================================
# 主题1: 动量 (Momentum)
# ============================================================================

def calc_rstr(df_daily, window_long=252, window_short=21, shift=1, start=None):
    """
    12-1个月经典动量 (RSTR)
    公式: RSTR_i = prod_{t=1}^{T-window_short} (1+r_{i,t}) - 1, T=window_long
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame (后复权)
        start: 开始日期，用于截断数据
        window_long: 长期窗口，默认252
        window_short: 短期窗口(剔除的最近交易日)，默认21
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # 计算日收益率
    df = df.with_columns([
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    # window_long日log累积收益 减去 window_short日log累积收益
    df = df.with_columns([
        (pl.col('ret') + 1).log().rolling_sum(window_size=window_long).over('vt_symbol').alias('log_cumret_long'),
        (pl.col('ret') + 1).log().rolling_sum(window_size=window_short).over('vt_symbol').alias('log_cumret_short')
    ])
    result = df.with_columns([
        ((pl.col('log_cumret_long') - pl.col('log_cumret_short')).exp() - 1).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_mmt_last30(df_minute):
    """
    日内分段动量 - 尾盘30分钟动量
    公式: mmt_last30 = P_close / P_close-30min - 1
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    # 获取每日收盘前30分钟的收盘价 (14:30:00 = 14:30-14:31这根K线的close)
    close_30min = df.filter(pl.col('time') == pl.time(14, 30)).select([
        'vt_symbol', 'date', pl.col('close').alias('close_30')
    ])
    # 获取每日收盘价（最后一根分钟K线的close，对应14:59:00=14:59-15:00的数据）
    close_daily = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('close').last().alias('close_end')
    ])
    # 合并计算
    result = close_30min.join(close_daily, on=['vt_symbol', 'date'], how='inner')
    result = result.with_columns([
        ((pl.col('close_end') / pl.col('close_30')) - 1).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_mmt_qrs(df_minute):
    """
    QRS动量 - 日内趋势强度与稳定性
    公式: QRS = beta * R^2 (过去50根分钟K线最高价对最低价回归)
    
    Args:
        df_minute: 包含 datetime, vt_symbol, high, low 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.sort(['vt_symbol', 'datetime'])
    
    # 50根滚动窗口的均值
    df = df.with_columns([
        pl.col('low').rolling_mean(window_size=50, min_periods=30).over(['vt_symbol', 'date']).alias('low_mean'),
        pl.col('high').rolling_mean(window_size=50, min_periods=30).over(['vt_symbol', 'date']).alias('high_mean')
    ])
    # 协方差和方差项
    df = df.with_columns([
        ((pl.col('low') - pl.col('low_mean')) * (pl.col('high') - pl.col('high_mean'))).alias('cov_term'),
        (pl.col('low') - pl.col('low_mean')).pow(2).alias('var_low_term'),
        (pl.col('high') - pl.col('high_mean')).pow(2).alias('var_high_term')
    ])
    # 50根滚动协方差和方差
    df = df.with_columns([
        pl.col('cov_term').rolling_mean(window_size=50, min_periods=30).over(['vt_symbol', 'date']).alias('cov'),
        pl.col('var_low_term').rolling_mean(window_size=50, min_periods=30).over(['vt_symbol', 'date']).alias('var_low'),
        pl.col('var_high_term').rolling_mean(window_size=50, min_periods=30).over(['vt_symbol', 'date']).alias('var_high')
    ])
    # beta = cov(low, high) / var(low)
    # R^2 = cov^2 / (var(low) * var(high))
    df = df.with_columns([
        (pl.col('cov') / (pl.col('var_low') + 1e-10)).alias('beta'),
        (pl.col('cov').pow(2) / (pl.col('var_low') * pl.col('var_high') + 1e-10)).alias('r2')
    ])
    df = df.with_columns([
        (pl.col('beta') * pl.col('r2')).alias('data')
    ])
    # 取每日最后一根K线的QRS值作为日频因子
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('data').last().alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_mmt_top20VolumeRet(df_minute):
    """
    成交量加权动量 - 高成交量时段收益率
    公式: 取成交量前20%分钟K线的平均收益率
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', pl.col('datetime').dt.date()])) - 1).alias('min_ret')
    ])
    # 每日计算成交量80%分位数
    vol_threshold = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').quantile(0.8).alias('vol_80')
    ])
    df = df.join(vol_threshold, on=['vt_symbol', 'date'])
    # 筛选高成交量时段
    high_vol_data = df.filter(pl.col('volume') >= pl.col('vol_80'))
    # 计算这些时段的平均收益率
    result = high_vol_data.group_by(['vt_symbol', 'date']).agg([
        pl.col('min_ret').mean().alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])





# ============================================================================
# 主题2: 反转 (Reversal)
# ============================================================================

def calc_streverse(df_daily, shift=21, start=None):
    """
    1个月短期反转
    公式: streverse = -(P_t / P_t-window - 1)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame (后复权)
        start: 开始日期，用于截断数据
        shift: 窗口大小，默认21
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').shift(shift).over('vt_symbol').alias('close_lag')
    ])
    result = df.with_columns([
        (-((pl.col('close') / pl.col('close_lag')) - 1)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_intraday_reversal(df_minute):
    """
    日内反转 - 10点至收盘的反转
    公式: intraday_rev = -(P_close / P_10:00 - 1)
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    # 获取10:00价格 (10:00:00 = 10:00-10:01这根K线的close)
    price_10am = df.filter(pl.col('time') == pl.time(10, 0)).select([
        'vt_symbol', 'date', pl.col('close').alias('close_10am')
    ])
    # 获取收盘价（每日最后一根分钟K线的close）
    close_price = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('close').last().alias('close_end')
    ])
    result = price_10am.join(close_price, on=['vt_symbol', 'date'], how='inner')
    result = result.with_columns([
        (-((pl.col('close_end') / pl.col('close_10am')) - 1)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_overnight_intraday_rev(df_daily, shift = 1, start=None):
    """
    隔夜-日内收益率拆分反转
    公式: ONIR_rev = -sgn(ONR_t) * IDR_t
    
    Args:
        df_daily: 包含 datetime, vt_symbol, open, close 的日频DataFrame
        start: 开始日期，用于截断数据
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('close').shift(shift).over('vt_symbol').alias('close_lag1')
    ])
    # 计算隔夜收益率和日内收益率
    df = df.with_columns([
        ((pl.col('open') / pl.col('close_lag1')) - 1).alias('onr'),
        ((pl.col('close') / pl.col('open')) - 1).alias('idr')
    ])
    # sgn(ONR) * IDR，取负
    result = df.with_columns([
        (-(pl.when(pl.col('onr') > 0).then(1).when(pl.col('onr') < 0).then(-1).otherwise(0) * pl.col('idr'))).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_volume_weighted_rev(df_daily, window=20, shift=1, start=None):
    """
    成交量加权反转 - 低成交量日收益率的反转
    公式: VWR_i = -(sum_{低量日} ret / N_low) * (N_low / window) = -sum_{低量日} ret / window
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    # window日成交量中位数
    df = df.with_columns([
        pl.col('volume').rolling_median(window_size=window).over('vt_symbol').alias('vol_median')
    ])
    # 标记低成交量日，非低量日记为0
    df = df.with_columns([
        pl.when(pl.col('volume') <= pl.col('vol_median')).then(pl.col('ret')).otherwise(0).alias('low_vol_ret')
    ])
    # window日内低成交量日收益率之和 / window，取负
    result = df.with_columns([
        (-pl.col('low_vol_ret').rolling_sum(window_size=window).over('vt_symbol') / pl.lit(window).cast(pl.Float64)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])





# ============================================================================
# 主题3: 波动率 (Volatility)
# ============================================================================

def calc_rvol_21d(df_daily, window=21, shift=1, start=None):
    """
    21日已实现波动率 (年化)
    公式: rvol = sqrt(252/(N-1) * sum((r - r_mean)^2)), N=window
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    # window日样本标准差 * sqrt(252) 年化
    result = df.with_columns([
        (pl.col('ret').rolling_std(window_size=window).over('vt_symbol') * pl.lit(252).sqrt()).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])




def calc_vol_upVol_ratio(df_daily, window=21, shift=1, start=None):
    """
    上行波动率占比
    公式: upVol / (upVol + downVol)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    # 使用window日滚动窗口计算
    df = df.with_columns([
        pl.when(pl.col('ret') > 0).then(pl.col('ret')).otherwise(0).alias('up_ret'),
        pl.when(pl.col('ret') < 0).then(pl.col('ret')).otherwise(0).alias('down_ret')
    ])
    df = df.with_columns([
        pl.col('up_ret').pow(2).rolling_mean(window_size=window).over('vt_symbol').sqrt().alias('up_vol'),
        pl.col('down_ret').pow(2).rolling_mean(window_size=window).over('vt_symbol').sqrt().alias('down_vol')
    ])
    result = df.with_columns([
        (pl.col('up_vol') / (pl.col('up_vol') + pl.col('down_vol') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_vol_downRatio(df_daily, window=21, shift=1, start=None):
    """
    下行波动占比
    公式: 下行方差 / 总方差
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    df = df.with_columns([
        pl.when(pl.col('ret') < 0).then(pl.col('ret').pow(2)).otherwise(0).alias('down_var'),
        pl.col('ret').pow(2).alias('total_var')
    ])
    result = df.with_columns([
        (pl.col('down_var').rolling_mean(window_size=window).over('vt_symbol') / 
         (pl.col('total_var').rolling_mean(window_size=window).over('vt_symbol') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_cmra(df_daily, window_short=21, window_long=252, shift=1, start=None):
    """
    收益率极差 (Cumulative Return Range)
    公式: ln(1 + max R) - ln(1 + min R)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window_short: 短期累积窗口，默认21
        window_long: 长期窗口(用于max/min)，默认252
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    # 计算window_short日累计收益率
    df = df.with_columns([
        # 先计算 (1+ret) 的累积乘积
        ((pl.col('ret') + 1).cum_prod().over('vt_symbol')).alias('cum_prod_ret')
    ]).with_columns([
        # 用当前累积乘积除以window_short天前的累积乘积得到滚动乘积
        ((pl.col('cum_prod_ret') / pl.col('cum_prod_ret').shift(window_short).over('vt_symbol')) - 1).alias('cum_ret_short')
    ])
    # 使用window_long日的最大最小累计收益
    result = df.with_columns([
        pl.col('cum_ret_short').rolling_max(window_size=window_long).over('vt_symbol').alias('max_ret'),
        pl.col('cum_ret_short').rolling_min(window_size=window_long).over('vt_symbol').alias('min_ret')
    ]).with_columns([
        ((1 + pl.col('max_ret')).log() - (1 + pl.col('min_ret')).log()).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题4: 换手率 (Turnover)
# ============================================================================

def calc_abn_turnover(df_daily, window_short=20, window_long=250, start=None):
    """
    异常换手率
    公式: 近window_short日换手率均值 / 近window_long日换手率均值
    注意: turnover字段为总成交金额(元)，需除以volume得到换手率
    
    Args:
        df_daily: 包含 datetime, vt_symbol, turnover, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window_short: 短期窗口，默认20
        window_long: 长期窗口，默认250
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        (pl.col('turnover') / (pl.col('volume') + 1e-10)).alias('turnover_rate')
    ])
    result = df.with_columns([
        pl.col('turnover_rate').rolling_mean(window_size=window_short).over('vt_symbol').alias('turnover_rate_short'),
        pl.col('turnover_rate').rolling_mean(window_size=window_long).over('vt_symbol').alias('turnover_rate_long')
    ]).with_columns([
        (pl.col('turnover_rate_short') / (pl.col('turnover_rate_long') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_turnover_vol(df_daily, window=21, start=None):
    """
    换手率波动率
    公式: window日换手率的标准差
    注意: turnover字段为总成交金额(元)，需除以volume得到换手率
    
    Args:
        df_daily: 包含 datetime, vt_symbol, turnover, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        (pl.col('turnover') / (pl.col('volume') + 1e-10)).alias('turnover_rate')
    ])
    result = df.with_columns([
        pl.col('turnover_rate').rolling_std(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_gtr(df_daily, window=60, shift=1, start=None):
    """
    换手率变化率稳定性 (GTR)
    公式: |mean(delta_turnover_rate)| / std(delta_turnover_rate)
    注意: turnover字段为总成交金额(元)，需除以volume得到换手率
    
    Args:
        df_daily: 包含 datetime, vt_symbol, turnover, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认60
        shift: 换手率偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        (pl.col('turnover') / (pl.col('volume') + 1e-10)).alias('turnover_rate')
    ])
    df = df.with_columns([
        (pl.col('turnover_rate') - pl.col('turnover_rate').shift(shift).over('vt_symbol')).alias('delta_turnover_rate')
    ])
    result = df.with_columns([
        pl.col('delta_turnover_rate').rolling_mean(window_size=window).over('vt_symbol').alias('delta_mean'),
        pl.col('delta_turnover_rate').rolling_std(window_size=window).over('vt_symbol').alias('delta_std')
    ]).with_columns([
        (pl.col('delta_mean').abs() / (pl.col('delta_std') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_turnover_ret_interact(df_daily, window=20, shift=1, start=None):
    """
    换手率与收益的交互
    公式: mean((turnover_rate - mean(turnover_rate)) * ret)
    注意: turnover字段为总成交金额(元)，需除以volume得到换手率
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close, turnover, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret'),
        (pl.col('turnover') / (pl.col('volume') + 1e-10)).alias('turnover_rate')
    ])
    df = df.with_columns([
        pl.col('turnover_rate').rolling_mean(window_size=window).over('vt_symbol').alias('turnover_rate_mean')
    ])
    df = df.with_columns([
        ((pl.col('turnover_rate') - pl.col('turnover_rate_mean')) * pl.col('ret')).alias('interact')
    ])
    result = df.with_columns([
        pl.col('interact').rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_abn_turnover_accel(df_daily, window_short=20, window_long=250, window_smooth=5, lag=5, start=None):
    """
    异常换手率加速度
    公式: (abn_turnover_t - abn_turnover_t-lag) / abn_turnover_t-lag
    注意: turnover字段为总成交金额(元)，需除以volume得到换手率
    
    Args:
        df_daily: 包含 datetime, vt_symbol, turnover, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window_short: 短期窗口，默认20
        window_long: 长期窗口，默认250
        window_smooth: 平滑窗口，默认5
        lag: 滞后天数，默认5
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        (pl.col('turnover') / (pl.col('volume') + 1e-10)).alias('turnover_rate')
    ])
    # 先计算异常换手率
    df = df.with_columns([
        pl.col('turnover_rate').rolling_mean(window_size=window_short).over('vt_symbol').alias('turnover_rate_short'),
        pl.col('turnover_rate').rolling_mean(window_size=window_long).over('vt_symbol').alias('turnover_rate_long')
    ])
    df = df.with_columns([
        (pl.col('turnover_rate_short') / (pl.col('turnover_rate_long') + 1e-10)).alias('abn_turnover')
    ])
    # window_smooth日均值平滑
    df = df.with_columns([
        pl.col('abn_turnover').rolling_mean(window_size=window_smooth).over('vt_symbol').alias('abn_turnover_smooth')
    ])
    # 计算加速度
    df = df.with_columns([
        pl.col('abn_turnover_smooth').shift(lag).over('vt_symbol').alias('abn_turnover_lag')
    ])
    result = df.with_columns([
        ((pl.col('abn_turnover_smooth') - pl.col('abn_turnover_lag')) / (pl.col('abn_turnover_lag') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题5: 流动性 (Liquidity)
# ============================================================================

def calc_liq_amihud(df_daily, window=20, shift=1, start=None):
    """
    Amihud非流动性
    公式: mean(|ret| / (V * 10^9)), V为日成交金额
    注意: turnover字段即为日成交金额(元)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close, turnover 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    # turnover字段即为日成交金额(元)
    df = df.with_columns([
        (pl.col('ret').abs() / (pl.col('turnover') + 1e-10) * 1e9).alias('illiq')
    ])
    result = df.with_columns([
        pl.col('illiq').rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_liq_closevol(df_minute):
    """
    尾盘成交量占比
    公式: 收盘前30分钟成交量 / 全天成交量
    
    Args:
        df_minute: 包含 datetime, vt_symbol, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    # 每日总成交量
    daily_volume = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').sum().alias('total_volume')
    ])
    # 尾盘30分钟成交量 (14:30-15:00)
    late_volume = df.filter(pl.col('time') >= pl.time(14, 30)).group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').sum().alias('late_volume')
    ])
    result = daily_volume.join(late_volume, on=['vt_symbol', 'date'], how='left')
    result = result.with_columns([
        (pl.col('late_volume') / (pl.col('total_volume') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_zero_trades_ratio(df_daily, window=126, start=None):
    """
    零交易天数占比
    公式: 零成交量天数 / 总天数
    
    Args:
        df_daily: 包含 datetime, vt_symbol, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认126
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.when(pl.col('volume') <= 0).then(1).otherwise(0).alias('is_zero_trade')
    ])
    result = df.with_columns([
        pl.col('is_zero_trade').rolling_sum(window_size=window).over('vt_symbol').alias('zero_count')
    ]).with_columns([
        (pl.col('zero_count') / pl.lit(window).cast(pl.Float64)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])



def calc_range_adjusted_amihud(df_daily, window=21, shift=1, start=None):
    """
    振幅调整Amihud
    公式: mean(range / (V * 10^9)), range = (high - low) / close_lag
    注意: turnover字段即为日成交金额(元)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, high, low, close, turnover 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
        shift: 前一日收盘价偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('close').shift(shift).over('vt_symbol').alias('close_lag1')
    ])
    df = df.with_columns([
        ((pl.col('high') - pl.col('low')) / pl.col('close_lag1')).alias('range_ratio')
    ])
    # turnover字段即为日成交金额(元)
    df = df.with_columns([
        (pl.col('range_ratio') / (pl.col('turnover') + 1e-10) * 1e9).alias('ra_illiq')
    ])
    result = df.with_columns([
        pl.col('ra_illiq').rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题6: 成交量 (Volume)
# ============================================================================

def calc_vr_20d(df_daily, window=20, start=None):
    """
    量比
    公式: 当日成交量 / 近window日均量
    
    Args:
        df_daily: 包含 datetime, vt_symbol, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    result = df.with_columns([
        pl.col('volume').rolling_mean(window_size=window).over('vt_symbol').alias('vol_mean')
    ]).with_columns([
        (pl.col('volume') / (pl.col('vol_mean') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_vol_stability(df_daily, window=60, shift=1, start=None):
    """
    成交量稳定性
    公式: |mean(delta_volume)| / std(delta_volume)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认60
        shift: 成交量偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        (pl.col('volume') - pl.col('volume').shift(shift).over('vt_symbol')).alias('delta_vol')
    ])
    result = df.with_columns([
        pl.col('delta_vol').rolling_mean(window_size=window).over('vt_symbol').alias('delta_mean'),
        pl.col('delta_vol').rolling_std(window_size=window).over('vt_symbol').alias('delta_std')
    ]).with_columns([
        (pl.col('delta_mean').abs() / (pl.col('delta_std') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_vol_skew(df_minute):
    """
    成交量分布偏度 (分钟级)
    公式: sum((t_k - t_mean)^3 * V_k) / [sum((t_k - t_mean)^2 * V_k)]^(3/2)
    注意: 使用日内K线索号(0,1,2...)作为时间t，避免休市时段影响
    
    Args:
        df_minute: 包含 datetime, vt_symbol, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.sort(['vt_symbol', 'datetime'])
    # 日内K线索号 (0, 1, 2, ...)，忽略休市时间间隔
    df = df.with_columns([
        pl.int_range(0, pl.count()).over(['vt_symbol', 'date']).alias('bar_idx')
    ])
    # 成交量加权平均K线索号
    vw_time = df.group_by(['vt_symbol', 'date']).agg([
        (pl.col('bar_idx') * pl.col('volume')).sum().alias('vw_idx_sum'),
        pl.col('volume').sum().alias('total_vol')
    ]).with_columns([
        (pl.col('vw_idx_sum') / (pl.col('total_vol') + 1e-10)).alias('vw_mean_idx')
    ])
    df = df.join(vw_time, on=['vt_symbol', 'date'])
    # 计算偏度
    df = df.with_columns([
        ((pl.col('bar_idx') - pl.col('vw_mean_idx')).pow(3) * pl.col('volume')).alias('skew_num'),
        ((pl.col('bar_idx') - pl.col('vw_mean_idx')).pow(2) * pl.col('volume')).alias('var_num')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('skew_num').sum().alias('skew_sum'),
        pl.col('var_num').sum().alias('var_sum')
    ]).with_columns([
        (pl.col('skew_sum') / (pl.col('var_sum').pow(1.5) + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_price_range_vol_ratio(df_minute):
    """
    价格区间成交量占比
    公式: (涨>2% + 跌>2%)成交量 / 总成交量
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', pl.col('datetime').dt.date()])) - 1).alias('min_ret')
    ])
    # 标记极端价格区间
    df = df.with_columns([
        pl.when((pl.col('min_ret').abs() > 0.02)).then(pl.col('volume')).otherwise(0).alias('extreme_vol')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('extreme_vol').sum().alias('extreme_vol_sum'),
        pl.col('volume').sum().alias('total_vol')
    ]).with_columns([
        (pl.col('extreme_vol_sum') / (pl.col('total_vol') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_pv_sync_residual(df_daily, window=20, start=None):
    """
    量价同步性残差
    公式: pv_sync_residual_i = V_i,t / V_m,t - mean(V_i / V_m)_window
    注意: V_m,t 为当天全市场所有股票成交量之和
    
    Args:
        df_daily: 包含 datetime, vt_symbol, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # 计算每日全市场成交量（当天所有股票volume之和）
    daily_market_vol = df.group_by('date').agg([
        pl.col('volume').sum().alias('market_volume')
    ])
    df = df.join(daily_market_vol, on='date')
    # 个股成交量占比
    df = df.with_columns([
        (pl.col('volume') / (pl.col('market_volume') + 1e-10)).alias('vol_ratio')
    ])
    # window日均值
    df = df.with_columns([
        pl.col('vol_ratio').rolling_mean(window_size=window).over('vt_symbol').alias('vol_ratio_mean')
    ])
    result = df.with_columns([
        (pl.col('vol_ratio') - pl.col('vol_ratio_mean')).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题7: 偏度与峰度 (Skewness & Kurtosis)
# ============================================================================

def calc_rskew(df_daily, window=21, shift=1, start=None):
    """
    已实现偏度
    公式: mean((r - r_mean)^3) / std(r)^3
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    df = df.with_columns([
        pl.col('ret').rolling_mean(window_size=window).over('vt_symbol').alias('ret_mean')
    ])
    df = df.with_columns([
        (pl.col('ret') - pl.col('ret_mean')).alias('ret_demean')
    ])
    result = df.with_columns([
        pl.col('ret_demean').pow(3).rolling_mean(window_size=window).over('vt_symbol').alias('m3'),
        pl.col('ret_demean').pow(2).rolling_mean(window_size=window).over('vt_symbol').alias('m2')
    ]).with_columns([
        (pl.col('m3') / (pl.col('m2').pow(1.5) + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])




def calc_rkurt(df_daily, window=21, shift=1, start=None):
    """
    已实现峰度
    公式: mean((r - r_mean)^4) / std(r)^4
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    df = df.with_columns([
        pl.col('ret').rolling_mean(window_size=window).over('vt_symbol').alias('ret_mean')
    ])
    df = df.with_columns([
        (pl.col('ret') - pl.col('ret_mean')).alias('ret_demean')
    ])
    result = df.with_columns([
        pl.col('ret_demean').pow(4).rolling_mean(window_size=window).over('vt_symbol').alias('m4'),
        pl.col('ret_demean').pow(2).rolling_mean(window_size=window).over('vt_symbol').alias('m2')
    ]).with_columns([
        (pl.col('m4') / (pl.col('m2').pow(2) + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_rmax(df_daily, window=21, shift=1, start=None):
    """
    最大日收益率
    公式: max(r_t-1, r_t-2, ..., r_t-window)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    result = df.with_columns([
        pl.col('ret').rolling_max(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_extreme_time(df_minute):
    """
    日内价格极值发生时间
    公式: (t_high - t_low) / T, T=240根K线
    注意: 使用日内K线索号(0,1,2...239)，9:30:00=0, 14:59:00=239
    
    Args:
        df_minute: 包含 datetime, vt_symbol, high, low 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.sort(['vt_symbol', 'datetime'])
    # 日内K线索号 (0, 1, 2, ...)，9:30:00=0, 14:59:00=239
    df = df.with_columns([
        pl.int_range(0, pl.count()).over(['vt_symbol', 'date']).alias('bar_idx')
    ])
    # 获取每日最高价出现的K线索号
    high_idx = df.filter(
        pl.col('high') == pl.col('high').max().over(['vt_symbol', 'date'])
    ).group_by(['vt_symbol', 'date']).agg([
        pl.col('bar_idx').first().alias('high_idx')
    ])
    # 获取每日最低价出现的K线索号
    low_idx = df.filter(
        pl.col('low') == pl.col('low').min().over(['vt_symbol', 'date'])
    ).group_by(['vt_symbol', 'date']).agg([
        pl.col('bar_idx').first().alias('low_idx')
    ])
    result = high_idx.join(low_idx, on=['vt_symbol', 'date'])
    result = result.with_columns([
        ((pl.col('high_idx') - pl.col('low_idx')) / 240.0).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题8: 相关性 (Correlation)
# ============================================================================

def calc_corr_pv(df_minute):
    """
    日内价量相关性
    公式: corr(分钟收盘价, 分钟成交量)
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.corr('close', 'volume').alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_corr_pvl(df_minute):
    """
    领先量价相关性 (成交量领先1分钟)
    公式: corr(分钟收盘价, 前一分钟成交量)
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('volume').shift(1).over(['vt_symbol', pl.col('datetime').dt.date()]).alias('volume_lag1')
    ]).filter(pl.col('volume_lag1').is_not_null())
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.corr('close', 'volume_lag1').alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_corr_prv(df_minute):
    """
    分钟收益与同步成交量相关性
    公式: corr(分钟收益率, 分钟成交量)
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', pl.col('datetime').dt.date()])) - 1).alias('min_ret')
    ]).filter(pl.col('min_ret').is_not_null())
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.corr('min_ret', 'volume').alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_corr_trend(df_minute, window_near=10, window_std=20, lag=10, start=None):
    """
    量价相关系数趋势
    公式: (近window_near日corr均值 - 远window_near日corr均值) / window_std日std
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
        start: 开始日期，用于截断数据
        window_near: 近期窗口，默认10
        window_std: 标准差窗口，默认20
        lag: 远期偏移，默认10
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    # 先计算每日的价量相关性
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    daily_corr = df.group_by(['vt_symbol', 'date']).agg([
        pl.corr('close', 'volume').alias('corr_pv')
    ]).sort(['vt_symbol', 'date'])
    # 计算近期均值、标准差
    daily_corr = daily_corr.with_columns([
        pl.col('corr_pv').rolling_mean(window_size=window_near).over('vt_symbol').alias('corr_near'),
        pl.col('corr_pv').rolling_std(window_size=window_std).over('vt_symbol').alias('corr_std')
    ])
    # 远期均值 = shift(lag)后再rolling_mean(window_near)
    daily_corr = daily_corr.with_columns([
        pl.col('corr_pv').shift(lag).over('vt_symbol').alias('corr_pv_shift')
    ])
    daily_corr = daily_corr.with_columns([
        pl.col('corr_pv_shift').rolling_mean(window_size=window_near).over('vt_symbol').alias('corr_far')
    ])
    result = daily_corr.with_columns([
        ((pl.col('corr_near') - pl.col('corr_far')) / (pl.col('corr_std') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_overnight_pv_mis(df_minute, shift=1, start=None):
    """
    隔夜价量错配
    公式: overnight_pv_mis = corr_pv_{t-shift} * ONR_t
    注意: ONR_t = day_open_t / day_close_{t-shift} - 1，从分钟数据中提取日open/日close
    
    Args:
        df_minute: 包含 datetime, vt_symbol, open, close, volume 的分钟频DataFrame
        start: 开始日期，用于截断数据
        shift: corr_pv与day_close的偏移天数，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # 计算每日corr_pv（分钟close与分钟volume的相关性）
    daily_corr = df.group_by(['vt_symbol', 'date']).agg([
        pl.corr('close', 'volume').alias('corr_pv')
    ]).sort(['vt_symbol', 'date'])
    # 提取每日open（第一根分钟K线的open）和close（最后一根分钟K线的close）
    daily_prices = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('open').first().alias('day_open'),
        pl.col('close').last().alias('day_close')
    ]).sort(['vt_symbol', 'date'])
    # 计算ONR
    daily_prices = daily_prices.with_columns([
        pl.col('day_close').shift(shift).over('vt_symbol').alias('day_close_lag1')
    ])
    daily_prices = daily_prices.with_columns([
        ((pl.col('day_open') / pl.col('day_close_lag1')) - 1).alias('onr')
    ])
    # 合并并shift corr_pv shift天
    result = daily_corr.join(daily_prices, on=['vt_symbol', 'date']).sort(['vt_symbol', 'date'])
    result = result.with_columns([
        pl.col('corr_pv').shift(shift).over('vt_symbol').alias('corr_pv_lag1')
    ])
    result = result.with_columns([
        (pl.col('corr_pv_lag1') * pl.col('onr')).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题9: 筹码分布 (Chip Distribution) - 简化版
# ============================================================================

def calc_doc_std(df_daily, window=60, start=None):
    """
    筹码分布标准差 - 简化版 (使用价格分布代替真实筹码)
    公式: 基于日内价格波动的标准差作为代理
    
    Args:
        df_daily: 包含 datetime, vt_symbol, high, low, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认60
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('daily_range')
    ])
    result = df.with_columns([
        pl.col('daily_range').rolling_std(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_doc_skew(df_daily, window=21, start=None):
    """
    筹码分布偏度 - 简化版
    公式: 使用日内价格位置偏度作为代理
    
    Args:
        df_daily: 包含 datetime, vt_symbol, open, high, low, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('close_position')
    ])
    # 计算close_position的window日偏度近似
    df = df.with_columns([
        pl.col('close_position').rolling_mean(window_size=window).over('vt_symbol').alias('pos_mean')
    ])
    df = df.with_columns([
        (pl.col('close_position') - pl.col('pos_mean')).alias('pos_demean')
    ])
    result = df.with_columns([
        pl.col('pos_demean').pow(3).rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_doc_vol_pdf90(df_daily, window=60, start=None):
    """
    90%分位筹码占比 - 简化版
    公式: 使用当前价格在过去N日价格区间中的位置作为代理
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认60
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').rolling_max(window_size=window).over('vt_symbol').alias('high_rolling'),
        pl.col('close').rolling_min(window_size=window).over('vt_symbol').alias('low_rolling')
    ])
    result = df.with_columns([
        ((pl.col('close') - pl.col('low_rolling')) / (pl.col('high_rolling') - pl.col('low_rolling') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_doc_profit_ratio(df_daily, window_cost=60, window=21, start=None):
    """
    筹码获利比率 - 简化版
    公式: 当前价格高于成本均线价的交易日占比
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window_cost: 成本均线窗口，默认60
        window: 计算占比的窗口，默认21
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # 使用window_cost日均线作为成本基准
    df = df.with_columns([
        pl.col('close').rolling_mean(window_size=window_cost).over('vt_symbol').alias('cost_basis')
    ])
    df = df.with_columns([
        pl.when(pl.col('close') > pl.col('cost_basis')).then(1).otherwise(0).alias('is_profit')
    ])
    result = df.with_columns([
        pl.col('is_profit').rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_doc_concentration(df_minute):
    """
    筹码集中度 - 简化版
    公式: 日内成交量分布集中度
    
    Args:
        df_minute: 包含 datetime, vt_symbol, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # 计算成交量在最大值附近的集中度
    df = df.with_columns([
        pl.col('volume').max().over(['vt_symbol', 'date']).alias('max_vol')
    ])
    df = df.with_columns([
        pl.when(pl.col('volume') > pl.col('max_vol') * 0.8).then(pl.col('volume')).otherwise(0).alias('peak_vol')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('peak_vol').sum().alias('peak_vol_sum'),
        pl.col('volume').sum().alias('total_vol')
    ]).with_columns([
        (pl.col('peak_vol_sum') / (pl.col('total_vol') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题10: 拥挤度 (Crowding) - 简化版
# ============================================================================

def calc_crowd_fft(df_minute):
    """
    傅里叶变换机构拥挤度 - 简化版
    公式: 使用成交量变异系数作为代理
    
    Args:
        df_minute: 包含 datetime, vt_symbol, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').std().alias('vol_std'),
        pl.col('volume').mean().alias('vol_mean')
    ]).with_columns([
        (pl.col('vol_std') / (pl.col('vol_mean') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])





def calc_crowd_turnover(df_daily, window_short=20, quantile=0.8, start=None):
    """
    换手率拥挤度
    公式: 近window_short日换手率超过历史quantile分位的天数占比
    
    Args:
        df_daily: 包含 datetime, vt_symbol, turnover 的日频DataFrame
        start: 开始日期，用于截断数据
        window_short: 短期窗口，默认20
        quantile: 分位数，默认0.8
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('turnover').quantile(quantile).over('vt_symbol').alias('turnover_threshold')
    ])
    df = df.with_columns([
        pl.when(pl.col('turnover') > pl.col('turnover_threshold')).then(1).otherwise(0).alias('high_turnover_flag')
    ])
    result = df.with_columns([
        pl.col('high_turnover_flag').rolling_mean(window_size=window_short).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_crowd_vol(df_daily, window_short=10, window_long=60, start=None):
    """
    波动率拥挤度
    公式: Z-score(window_short日波动率 / window_long日波动率)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window_short: 短期窗口，默认10
        window_long: 长期窗口，默认60
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(1).over('vt_symbol')) - 1).alias('ret')
    ])
    df = df.with_columns([
        pl.col('ret').rolling_std(window_size=window_short).over('vt_symbol').alias('vol_short'),
        pl.col('ret').rolling_std(window_size=window_long).over('vt_symbol').alias('vol_long')
    ])
    df = df.with_columns([
        (pl.col('vol_short') / (pl.col('vol_long') + 1e-10)).alias('vol_ratio')
    ])
    result = df.with_columns([
        pl.col('vol_ratio').rolling_mean(window_size=window_long).over('vt_symbol').alias('ratio_mean'),
        pl.col('vol_ratio').rolling_std(window_size=window_long).over('vt_symbol').alias('ratio_std')
    ]).with_columns([
        ((pl.col('vol_ratio') - pl.col('ratio_mean')) / (pl.col('ratio_std') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题11: 资金流与能量 (Money Flow) - 使用分钟数据近似
# ============================================================================

def calc_trade_CBuyRatio(df_minute):
    """
    主动买入成交占比 - 简化版
    公式: 使用分钟上涨成交量占比作为主动买入代理
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').shift(1).over(['vt_symbol', 'date']).alias('close_lag1')
    ])
    df = df.with_columns([
        pl.when(pl.col('close') >= pl.col('close_lag1')).then(pl.col('volume')).otherwise(0).alias('up_volume')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('up_volume').sum().alias('up_vol_sum'),
        pl.col('volume').sum().alias('total_vol')
    ]).with_columns([
        (pl.col('up_vol_sum') / (pl.col('total_vol') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_trade_headRatio_std(df_minute, window=21, start=None):
    """
    开盘成交占比波动率
    公式: 9:30-10:00成交量占比的window日标准差

    Args:
        df_minute: 包含 datetime, vt_symbol, volume 的分钟频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认21

    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    daily_volume = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').sum().alias('total_volume')
    ]).sort(['vt_symbol', 'date'])
    # 开盘30分钟成交量
    morning_volume = df.filter(pl.col('time') <= pl.time(10, 0)).group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').sum().alias('morning_volume')
    ]).sort(['vt_symbol', 'date'])
    result = daily_volume.join(morning_volume, on=['vt_symbol', 'date'], how='left').sort(['vt_symbol', 'date'])
    result = result.with_columns([
        (pl.col('morning_volume') / (pl.col('total_volume') + 1e-10)).alias('morning_ratio')
    ])
    result = result.with_columns([
        pl.col('morning_ratio').rolling_std(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_trade_netBuyRatio(df_minute):
    """
    净主动买入占比 - 简化版
    公式: (上涨成交量 - 下跌成交量) / 总成交量
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').shift(1).over(['vt_symbol', 'date']).alias('close_lag1')
    ])
    df = df.with_columns([
        pl.when(pl.col('close') >= pl.col('close_lag1')).then(pl.col('volume')).otherwise(-pl.col('volume')).alias('signed_volume')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('signed_volume').sum().alias('net_volume'),
        pl.col('volume').sum().alias('total_vol')
    ]).with_columns([
        (pl.col('net_volume') / (pl.col('total_vol') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_mf_residual(df_daily, window=60, start=None):
    """
    日度资金流残差 - 简化版
    公式: 个股成交量偏离市场均值的部分
    
    Args:
        df_daily: 包含 datetime, vt_symbol, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认60
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # 简化: 使用成交量的window日均值偏离作为代理
    result = df.with_columns([
        pl.col('volume').rolling_mean(window_size=window).over('vt_symbol').alias('vol_mean'),
        pl.col('volume').rolling_std(window_size=window).over('vt_symbol').alias('vol_std')
    ]).with_columns([
        ((pl.col('volume') - pl.col('vol_mean')) / (pl.col('vol_std') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_trade_top20retRatio(df_minute):
    """
    高收益K线成交量占比
    公式: 收益率前20%的分钟K线成交量 / 总成交量
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(1).over(['vt_symbol', pl.col('datetime').dt.date()])) - 1).alias('min_ret')
    ])
    # 计算每日收益率80%分位数
    ret_threshold = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('min_ret').quantile(0.8).alias('ret_80')
    ])
    df = df.join(ret_threshold, on=['vt_symbol', 'date'])
    df = df.with_columns([
        pl.when(pl.col('min_ret') >= pl.col('ret_80')).then(pl.col('volume')).otherwise(0).alias('top_ret_vol')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('top_ret_vol').sum().alias('top_vol_sum'),
        pl.col('volume').sum().alias('total_vol')
    ]).with_columns([
        (pl.col('top_vol_sum') / (pl.col('total_vol') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题12: 价格形态 (Price Pattern)
# ============================================================================

def calc_high_position_volume(df_daily, window_price=60, window_vol=20, window_result=21, threshold_price=0.95, threshold_vol=1.5, start=None):
    """
    高位放量
    公式: 价格接近window_price日高点且成交量>threshold_vol倍window_vol日均量的标准化值，回溯window_result日
    
    Args:
        df_daily: 包含 datetime, vt_symbol, high, close, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window_price: 价格高点窗口，默认60
        window_vol: 成交量均值窗口，默认20
        window_result: 结果平滑窗口，默认21
        threshold_price: 价格阈值，默认0.95
        threshold_vol: 成交量倍数阈值，默认1.5
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('high').rolling_max(window_size=window_price).over('vt_symbol').alias('high_rolling')
    ])
    df = df.with_columns([
        pl.col('volume').rolling_mean(window_size=window_vol).over('vt_symbol').alias('vol_mean'),
        pl.col('volume').rolling_std(window_size=window_vol).over('vt_symbol').alias('vol_std')
    ])
    # 条件: 价格>threshold_price*window_price日高点 且 成交量>threshold_vol*window_vol日均量
    df = df.with_columns([
        ((pl.col('close') / (pl.col('high_rolling') + 1e-10) > threshold_price) & 
         (pl.col('volume') > pl.col('vol_mean') * threshold_vol)).cast(pl.Int32).alias('condition')
    ])
    df = df.with_columns([
        ((pl.col('volume') - pl.col('vol_mean')) / (pl.col('vol_std') + 1e-10)).alias('vol_zscore')
    ])
    df = df.with_columns([
        (pl.col('condition').cast(pl.Float64) * pl.col('vol_zscore')).alias('daily_data')
    ])
    result = df.with_columns([
        pl.col('daily_data').rolling_mean(window_size=window_result).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_low_position_volume(df_daily, window_price=60, window_vol=20, window_result=21, threshold_price=1.05, threshold_vol=1.5, start=None):
    """
    低位放量
    公式: 价格接近window_price日低点且成交量>threshold_vol倍window_vol日均量的标准化值，回溯window_result日
    
    Args:
        df_daily: 包含 datetime, vt_symbol, low, close, volume 的日频DataFrame
        start: 开始日期，用于截断数据
        window_price: 价格低点窗口，默认60
        window_vol: 成交量均值窗口，默认20
        window_result: 结果平滑窗口，默认21
        threshold_price: 价格阈值，默认1.05
        threshold_vol: 成交量倍数阈值，默认1.5
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('low').rolling_min(window_size=window_price).over('vt_symbol').alias('low_rolling')
    ])
    df = df.with_columns([
        pl.col('volume').rolling_mean(window_size=window_vol).over('vt_symbol').alias('vol_mean'),
        pl.col('volume').rolling_std(window_size=window_vol).over('vt_symbol').alias('vol_std')
    ])
    df = df.with_columns([
        ((pl.col('close') / (pl.col('low_rolling') + 1e-10) < threshold_price) & 
         (pl.col('volume') > pl.col('vol_mean') * threshold_vol)).cast(pl.Int32).alias('condition')
    ])
    df = df.with_columns([
        ((pl.col('volume') - pl.col('vol_mean')) / (pl.col('vol_std') + 1e-10)).alias('vol_zscore')
    ])
    df = df.with_columns([
        (pl.col('condition').cast(pl.Float64) * pl.col('vol_zscore')).alias('daily_data')
    ])
    result = df.with_columns([
        pl.col('daily_data').rolling_mean(window_size=window_result).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_candle_body_ratio(df_daily, window=5, start=None):
    """
    日线实体占比
    公式: |close - open| / (high - low)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, open, high, low, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 平滑窗口，默认5
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') - pl.col('open')).abs() / (pl.col('high') - pl.col('low') + 1e-10)).alias('body_ratio')
    ])
    result = df.with_columns([
        pl.col('body_ratio').rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_gap_ratio(df_daily, window=20, shift=1, start=None):
    """
    缺口率
    公式: (open - close_lag1) / close_lag1
    
    Args:
        df_daily: 包含 datetime, vt_symbol, open, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 平滑窗口，默认20
        shift: 前一日收盘价偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('close').shift(shift).over('vt_symbol').alias('close_lag1')
    ])
    df = df.with_columns([
        ((pl.col('open') - pl.col('close_lag1')) / (pl.col('close_lag1') + 1e-10)).alias('gap')
    ])
    result = df.with_columns([
        pl.col('gap').rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_shadow_ratio(df_daily, window=5, start=None):
    """
    影线占比
    公式: upper_shadow - lower_shadow
    
    Args:
        df_daily: 包含 datetime, vt_symbol, open, high, low, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 平滑窗口，默认5
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('high') - pl.max_horizontal(['open', 'close'])) / (pl.col('high') - pl.col('low') + 1e-10)).alias('upper_shadow'),
        ((pl.min_horizontal(['open', 'close']) - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('lower_shadow')
    ])
    df = df.with_columns([
        (pl.col('upper_shadow') - pl.col('lower_shadow')).alias('shadow_diff')
    ])
    result = df.with_columns([
        pl.col('shadow_diff').rolling_mean(window_size=window).over('vt_symbol').alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题13: 技术指标 (Technical Indicators)
# ============================================================================

def calc_rsi(df_daily, period=14, shift=1, start=None):
    """
    RSI相对强弱指数
    公式: RSI = 100 - 100/(1 + RS), RS = avg(up)/avg(down)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        period: RSI周期, 默认14
        shift: 收益率计算偏移，默认1
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('close') / pl.col('close').shift(shift).over('vt_symbol')) - 1).alias('ret')
    ])
    df = df.with_columns([
        pl.when(pl.col('ret') > 0).then(pl.col('ret')).otherwise(0).alias('up'),
        pl.when(pl.col('ret') < 0).then(-pl.col('ret')).otherwise(0).alias('down')
    ])
    df = df.with_columns([
        pl.col('up').rolling_mean(window_size=period).over('vt_symbol').alias('avg_up'),
        pl.col('down').rolling_mean(window_size=period).over('vt_symbol').alias('avg_down')
    ])
    result = df.with_columns([
        (100 - 100 / (1 + pl.col('avg_up') / (pl.col('avg_down') + 1e-10))).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_macd_hist(df_daily, fast=12, slow=26, signal=9, start=None):
    """
    MACD离差值 (柱)
    公式: MACD_hist = (EMA_fast - EMA_slow - DEA) / close
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        fast: 快线周期，默认12
        slow: 慢线周期，默认26
        signal: 信号线周期，默认9
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # EMA计算
    df = df.with_columns([
        pl.col('close').ewm_mean(span=fast, adjust=False).over('vt_symbol').alias('ema_fast'),
        pl.col('close').ewm_mean(span=slow, adjust=False).over('vt_symbol').alias('ema_slow')
    ])
    df = df.with_columns([
        (pl.col('ema_fast') - pl.col('ema_slow')).alias('dif')
    ])
    df = df.with_columns([
        pl.col('dif').ewm_mean(span=signal, adjust=False).over('vt_symbol').alias('dea')
    ])
    result = df.with_columns([
        ((pl.col('dif') - pl.col('dea')) / pl.col('close')).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_boll_width(df_daily, window=20, start=None):
    """
    布林带宽度百分比
    公式: (close - MA) / (2 * std)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').rolling_mean(window_size=window).over('vt_symbol').alias('ma'),
        pl.col('close').rolling_std(window_size=window).over('vt_symbol').alias('std')
    ])
    result = df.with_columns([
        ((pl.col('close') - pl.col('ma')) / (2 * pl.col('std') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_ma_divergence(df_daily, window_short=5, window_long=60, start=None):
    """
    均线偏离度
    公式: (MA_short - MA_long) / MA_long
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window_short: 短期均线窗口，默认5
        window_long: 长期均线窗口，默认60
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').rolling_mean(window_size=window_short).over('vt_symbol').alias('ma_short'),
        pl.col('close').rolling_mean(window_size=window_long).over('vt_symbol').alias('ma_long')
    ])
    result = df.with_columns([
        ((pl.col('ma_short') - pl.col('ma_long')) / (pl.col('ma_long') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_boll_bandwidth(df_daily, window=20, start=None):
    """
    布林带宽
    公式: 4 * std / MA
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').rolling_mean(window_size=window).over('vt_symbol').alias('ma'),
        pl.col('close').rolling_std(window_size=window).over('vt_symbol').alias('std')
    ])
    result = df.with_columns([
        (4 * pl.col('std') / (pl.col('ma') + 1e-10)).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题14: 振幅/价差 (Amplitude / Spread)
# ============================================================================

def calc_daily_range(df_daily):
    """
    日振幅
    公式: (high - low) / open
    
    Args:
        df_daily: 包含 datetime, vt_symbol, open, high, low 的日频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('high') - pl.col('low')) / (pl.col('open') + 1e-10)).alias('data')
    ])
    return df.select(['vt_symbol', 'date', 'data'])


def calc_ideal_range_cut(df_minute):
    """
    分钟理想振幅
    公式: 剔除开盘后25%和收盘前25%K线后的中间50%K线的价格波动
    注意: 使用K线索号而非价格排序
    
    Args:
        df_minute: 包含 datetime, vt_symbol, high, low 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.sort(['vt_symbol', 'datetime'])
    # 日内K线索号 (0, 1, 2, ...)
    df = df.with_columns([
        pl.int_range(0, pl.count()).over(['vt_symbol', 'date']).alias('bar_idx')
    ])
    # 每日总K线数
    daily_count = df.group_by(['vt_symbol', 'date']).agg([
        pl.count().alias('total_bars')
    ])
    df = df.join(daily_count, on=['vt_symbol', 'date'])
    # 剔除前25%和后25%，取中间50%
    mid_data = df.filter(
        (pl.col('bar_idx') >= pl.col('total_bars') * 0.25) &
        (pl.col('bar_idx') < pl.col('total_bars') * 0.75)
    )
    # 中间50%K线的价格波动 = max(high) - min(low)
    result = mid_data.group_by(['vt_symbol', 'date']).agg([
        (pl.col('high').max() - pl.col('low').min()).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_effective_spread(df_daily, window=20, start=None):
    """
    有效价差 (Roll Spread)
    公式: 2 * sqrt(|cov(dP_t, dP_t-1)|)
    
    Args:
        df_daily: 包含 datetime, vt_symbol, close 的日频DataFrame
        start: 开始日期，用于截断数据
        window: 窗口大小，默认20
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('close').diff().over('vt_symbol').alias('dp')
    ])
    df = df.with_columns([
        pl.col('dp').shift(1).over('vt_symbol').alias('dp_lag1')
    ])
    # 计算协方差近似 (dp * dp_lag1的均值)
    df = df.with_columns([
        (pl.col('dp') * pl.col('dp_lag1')).alias('cov_term')
    ])
    result = df.with_columns([
        pl.col('cov_term').rolling_mean(window_size=window).over('vt_symbol').alias('cov_mean')
    ]).with_columns([
        (2 * (pl.col('cov_mean').abs() + 1e-10).sqrt()).alias('data')
    ])
    # 截断数据，只保留start之后的数据
    result = result.filter(pl.col('date') >= start)
    return result.select(['vt_symbol', 'date', 'data'])


def calc_adjusted_range(df_daily):
    """
    修正振幅
    公式: ((high - low) - |close - open|) / open
    即: 影线振幅 / 开盘价
    
    Args:
        df_daily: 包含 datetime, vt_symbol, open, high, low, close 的日频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_daily
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        ((pl.col('high') - pl.col('low')) / (pl.col('open') + 1e-10)).alias('total_range'),
        ((pl.col('close') - pl.col('open')).abs() / (pl.col('open') + 1e-10)).alias('body_range')
    ])
    result = df.with_columns([
        (pl.col('total_range') - pl.col('body_range')).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_open_range_ratio(df_minute):
    """
    开盘振幅比
    公式: (10:00前振幅) / (全天振幅)
    
    Args:
        df_minute: 包含 datetime, vt_symbol, high, low 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date'),
        pl.col('datetime').dt.time().alias('time')
    ])
    # 全天振幅
    daily_range = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('high').max().alias('daily_high'),
        pl.col('low').min().alias('daily_low')
    ]).with_columns([
        (pl.col('daily_high') - pl.col('daily_low')).alias('daily_range')
    ])
    # 10:00前振幅
    morning_data = df.filter(pl.col('time') <= pl.time(10, 0))
    morning_range = morning_data.group_by(['vt_symbol', 'date']).agg([
        pl.col('high').max().alias('morning_high'),
        pl.col('low').min().alias('morning_low')
    ]).with_columns([
        (pl.col('morning_high') - pl.col('morning_low')).alias('morning_range')
    ])
    result = daily_range.join(morning_range, on=['vt_symbol', 'date'], how='left')
    result = result.with_columns([
        (pl.col('morning_range') / (pl.col('daily_range') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 主题15: 日内微观结构 (Intraday Microstructure)
# ============================================================================
def calc_avg_trade_size(df_minute):
    """
    逐笔成交平均单笔金额 - 简化版 (分钟数据)
    公式: 当日总成交金额 / 分钟数
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        (pl.col('close') * pl.col('volume')).sum().alias('total_amount'),
        pl.count().alias('num_bars')
    ]).with_columns([
        (pl.col('total_amount') / (pl.col('num_bars') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_big_order_net(df_minute):
    """
    大单净流入强度 - 简化版
    公式: 大成交量分钟的净流入占比
    
    Args:
        df_minute: 包含 datetime, vt_symbol, close, volume 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    # 计算成交量80%分位数作为大单阈值
    vol_threshold = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('volume').quantile(0.8).alias('vol_80')
    ])
    df = df.join(vol_threshold, on=['vt_symbol', 'date'])
    # 标记大单
    df = df.with_columns([
        pl.col('close').shift(1).over(['vt_symbol', 'date']).alias('close_lag1')
    ])
    df = df.with_columns([
        pl.when(pl.col('volume') >= pl.col('vol_80')).then(
            pl.when(pl.col('close') >= pl.col('close_lag1')).then(pl.col('volume')).otherwise(-pl.col('volume'))
        ).otherwise(0).alias('big_signed_vol')
    ])
    result = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('big_signed_vol').sum().alias('big_net_vol'),
        pl.col('volume').sum().alias('total_vol')
    ]).with_columns([
        (pl.col('big_net_vol') / (pl.col('total_vol') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


def calc_intraday_price_efficiency(df_minute):
    """
    日内价格效率
    公式: |close - open| / sum(|price_t - price_t-1|)
    
    Args:
        df_minute: 包含 datetime, vt_symbol, open, close 的分钟频DataFrame
    
    Returns:
        DataFrame with columns: [vt_symbol, date, data]
    """
    df = df_minute
    df = df.with_columns([
        pl.col('datetime').dt.date().alias('date')
    ])
    df = df.with_columns([
        pl.col('close').shift(1).over(['vt_symbol', 'date']).alias('close_lag1')
    ])
    df = df.with_columns([
        (pl.col('close') - pl.col('close_lag1')).abs().alias('price_change')
    ])
    # 每日汇总
    daily = df.group_by(['vt_symbol', 'date']).agg([
        pl.col('close').last().alias('day_close'),
        pl.col('open').first().alias('day_open'),
        pl.col('price_change').sum().alias('total_path')
    ])
    result = daily.with_columns([
        ((pl.col('day_close') - pl.col('day_open')).abs() / (pl.col('total_path') + 1e-10)).alias('data')
    ])
    return result.select(['vt_symbol', 'date', 'data'])


# ============================================================================
# 因子注册表 - 统一注册表
# ============================================================================
FACTOR_REGISTRY = {
    # 分钟级因子 (日内计算)
    'late_skew_ret': { "func": calc_late_skew_ret, "type": PRICE_AND_VOLUME},
    'down_vol_perc': { "func": calc_down_vol_perc, "type": PRICE_AND_VOLUME},
    'corr_ret_lastret': { "func": calc_corr_ret_lastret, "type": PRICE_AND_VOLUME},
    'corr_close_nextopen': { "func": calc_corr_close_nextopen, "type": PRICE_AND_VOLUME},
    'volume_perc2': { "func": calc_volume_perc2, "type": PRICE_AND_VOLUME},
    'volume_perc3': { "func": calc_volume_perc3, "type": PRICE_AND_VOLUME},
    'volume_perc4': { "func": calc_volume_perc4, "type": PRICE_AND_VOLUME},
    'volume_perc5': { "func": calc_volume_perc5, "type": PRICE_AND_VOLUME},
    'volume_perc6': { "func": calc_volume_perc6, "type": PRICE_AND_VOLUME},
    'volume_perc7': { "func": calc_volume_perc7, "type": PRICE_AND_VOLUME},
    'early_corr_volume_ret': { "func": calc_early_corr_volume_ret, "type": PRICE_AND_VOLUME},
    'corr_volume_amplitude': { "func": calc_corr_volume_amplitude, "type": PRICE_AND_VOLUME},

    # 动量 - 分线部分
    'mmt_last30': { "func": calc_mmt_last30, "type": PRICE_AND_VOLUME},
    'mmt_qrs': { "func": calc_mmt_qrs, "type": PRICE_AND_VOLUME},
    'mmt_top20VolumeRet': { "func": calc_mmt_top20VolumeRet, "type": PRICE_AND_VOLUME},
    
    # 反转 - 分线部分
    'intraday_reversal': { "func": calc_intraday_reversal, "type": PRICE_AND_VOLUME},
    
    # 流动性 - 分线部分
    'liq_closevol': { "func": calc_liq_closevol, "type": PRICE_AND_VOLUME},
    
    # 成交量 - 分线部分
    'vol_skew': { "func": calc_vol_skew, "type": PRICE_AND_VOLUME},
    'price_range_vol_ratio': { "func": calc_price_range_vol_ratio, "type": PRICE_AND_VOLUME},
    
    # 偏度与峰度 - 分线部分
    'extreme_time': { "func": calc_extreme_time, "type": PRICE_AND_VOLUME},

    # 相关性 - 分线部分
    'corr_pv': { "func": calc_corr_pv, "type": PRICE_AND_VOLUME},
    'corr_pvl': { "func": calc_corr_pvl, "type": PRICE_AND_VOLUME},
    'corr_prv': { "func": calc_corr_prv, "type": PRICE_AND_VOLUME},
    'corr_trend': { "func": calc_corr_trend, "type": PRICE_AND_VOLUME},
    'overnight_pv_mis': { "func": calc_overnight_pv_mis, "type": PRICE_AND_VOLUME},
    
    # 资金流与能量 - 分线部分
    'trade_top20retRatio': { "func": calc_trade_top20retRatio, "type": PRICE_AND_VOLUME},
    
    # 振幅/价差 - 分线部分
    'ideal_range_cut': { "func": calc_ideal_range_cut, "type": PRICE_AND_VOLUME},
    'open_range_ratio': { "func": calc_open_range_ratio, "type": PRICE_AND_VOLUME},
    
    # 日内微观结构
    'intraday_price_efficiency': { "func": calc_intraday_price_efficiency, "type": PRICE_AND_VOLUME},
    
    # 筹码分布 - 分线部分
    
    # 拥挤度 - 分线部分

    # 日线因子
    # 动量
    'rstr': { "func": calc_rstr, "type": PRICE_AND_VOLUME},
    
    # 反转
    'streverse_1m': { "func": calc_streverse, "type": PRICE_AND_VOLUME},
    'streverse_2m': { "func": calc_streverse, "type": PRICE_AND_VOLUME},
    'overnight_intraday_rev': { "func": calc_overnight_intraday_rev, "type": PRICE_AND_VOLUME},
    'volume_weighted_rev': { "func": calc_volume_weighted_rev, "type": PRICE_AND_VOLUME},
    
    # 波动率
    'rvol_21d': { "func": calc_rvol_21d, "type": PRICE_AND_VOLUME},
    'vol_upVol_ratio': { "func": calc_vol_upVol_ratio, "type": PRICE_AND_VOLUME},
    'vol_downRatio': { "func": calc_vol_downRatio, "type": PRICE_AND_VOLUME},
    'cmra': { "func": calc_cmra, "type": PRICE_AND_VOLUME},
    
    # 换手率
    'abn_turnover': { "func": calc_abn_turnover, "type": PRICE_AND_VOLUME},
    'turnover_vol': { "func": calc_turnover_vol, "type": PRICE_AND_VOLUME},
    'gtr': { "func": calc_gtr, "type": PRICE_AND_VOLUME},
    'turnover_ret_interact': { "func": calc_turnover_ret_interact, "type": PRICE_AND_VOLUME},
    'abn_turnover_accel': { "func": calc_abn_turnover_accel, "type": PRICE_AND_VOLUME},
    
    # 流动性
    'liq_amihud': { "func": calc_liq_amihud, "type": PRICE_AND_VOLUME},
    'zero_trades_ratio': { "func": calc_zero_trades_ratio, "type": PRICE_AND_VOLUME},
    'range_adjusted_amihud': { "func": calc_range_adjusted_amihud, "type": PRICE_AND_VOLUME},
    
    # 成交量
    'vr_20d': { "func": calc_vr_20d, "type": PRICE_AND_VOLUME},
    'vol_stability': { "func": calc_vol_stability, "type": PRICE_AND_VOLUME},
    'pv_sync_residual': { "func": calc_pv_sync_residual, "type": PRICE_AND_VOLUME},
    
    # 偏度与峰度
    'rskew': { "func": calc_rskew, "type": PRICE_AND_VOLUME},
    'rkurt': { "func": calc_rkurt, "type": PRICE_AND_VOLUME},
    'rmax': { "func": calc_rmax, "type": PRICE_AND_VOLUME},
    
    # 价格形态
    'high_position_volume': { "func": calc_high_position_volume, "type": PRICE_AND_VOLUME},
    'low_position_volume': { "func": calc_low_position_volume, "type": PRICE_AND_VOLUME},
    'candle_body_ratio': { "func": calc_candle_body_ratio, "type": PRICE_AND_VOLUME},
    'gap_ratio': { "func": calc_gap_ratio, "type": PRICE_AND_VOLUME},
    'shadow_ratio': { "func": calc_shadow_ratio, "type": PRICE_AND_VOLUME},
    
    # 技术指标
    'rsi': { "func": calc_rsi, "type": PRICE_AND_VOLUME},
    'macd_hist': { "func": calc_macd_hist, "type": PRICE_AND_VOLUME},
    'boll_width': { "func": calc_boll_width, "type": PRICE_AND_VOLUME},
    'ma_divergence': { "func": calc_ma_divergence, "type": PRICE_AND_VOLUME},
    'boll_bandwidth': { "func": calc_boll_bandwidth, "type": PRICE_AND_VOLUME},
    
    # 振幅/价差
    'daily_range': { "func": calc_daily_range, "type": PRICE_AND_VOLUME},
    'effective_spread': { "func": calc_effective_spread, "type": PRICE_AND_VOLUME},
    'adjusted_range': { "func": calc_adjusted_range, "type": PRICE_AND_VOLUME},
    
    # 筹码分布
    
    # 拥挤度
    
    # 资金流
}

FACTOR_NAMES = list(FACTOR_REGISTRY.keys())

# ============================================================================
# 参数注册表 - 新结构
# 格式: {
#   factor_name: {
#       "constant_params": {...},
#       "df_params": {
#           "df_daily": {"param1": value1, "max_window": max_val},
#           "df_minute": {"param2": value2, "max_window": max_val}
#       }
#   }
# }
# ============================================================================
PARAMS_REGISTRY = {
    # 分钟级因子 - 日内计算，无需跨日滚动
    'late_skew_ret': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'down_vol_perc': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'corr_ret_lastret': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'corr_close_nextopen': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'volume_perc2': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'volume_perc3': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'volume_perc4': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'volume_perc5': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'volume_perc6': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'volume_perc7': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'early_corr_volume_ret': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'corr_volume_amplitude': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'mmt_last30': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'mmt_qrs': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'mmt_top20VolumeRet': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'intraday_reversal': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'liq_closevol': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'vol_skew': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'price_range_vol_ratio': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'extreme_time': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'corr_pv': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'corr_pvl': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'corr_prv': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'trade_top20retRatio': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'ideal_range_cut': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'open_range_ratio': {"constant_params": {}, "df_params": {"df_minute": {}}},
    'intraday_price_efficiency': {"constant_params": {}, "df_params": {"df_minute": {}}},

    # 分钟级因子 - 跨日滚动
    'corr_trend': {"constant_params": {}, "df_params": {"df_minute": {"window_near": 10, "window_std": 20, "lag": 10, "max_window": 20}}},
    'overnight_pv_mis': {"constant_params": {}, "df_params": {"df_minute": {"shift": 1, "max_window": 1}}},

    # 日线因子
    'rstr': {"constant_params": {}, "df_params": {"df_daily": {"window_long": 252, "window_short": 21, "shift": 1, "max_window": 252}}},
    'streverse_1m': {"constant_params": {}, "df_params": {"df_daily": {"shift": 21, "max_window": 21}}},
    'streverse_2m': {"constant_params": {}, "df_params": {"df_daily": {"shift": 42, "max_window": 42}}},
    'overnight_intraday_rev': {"constant_params": {}, "df_params": {"df_daily": {"shift": 1, "max_window": 1}}},
    'volume_weighted_rev': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "shift": 1, "max_window": 20}}},
    'rvol_21d': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "shift": 1, "max_window": 21}}},
    'vol_upVol_ratio': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "shift": 1, "max_window": 21}}},
    'vol_downRatio': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "shift": 1, "max_window": 21}}},
    'cmra': {"constant_params": {}, "df_params": {"df_daily": {"window_short": 21, "window_long": 252, "shift": 1, "max_window": 273}}},
    'abn_turnover': {"constant_params": {}, "df_params": {"df_daily": {"window_short": 20, "window_long": 250, "max_window": 250}}},
    'turnover_vol': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "max_window": 21}}},
    'gtr': {"constant_params": {}, "df_params": {"df_daily": {"window": 60, "shift": 1, "max_window": 60}}},
    'turnover_ret_interact': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "shift": 1, "max_window": 40}}},
    'abn_turnover_accel': {"constant_params": {}, "df_params": {"df_daily": {"window_short": 20, "window_long": 250, "window_smooth": 5, "lag": 5, "max_window": 260}}},
    'liq_amihud': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "shift": 1, "max_window": 20}}},
    'zero_trades_ratio': {"constant_params": {}, "df_params": {"df_daily": {"window": 126, "max_window": 126}}},
    'range_adjusted_amihud': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "shift": 1, "max_window": 21}}},
    'vr_20d': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "max_window": 20}}},
    'vol_stability': {"constant_params": {}, "df_params": {"df_daily": {"window": 60, "shift": 1, "max_window": 60}}},
    'pv_sync_residual': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "max_window": 20}}},
    'rskew': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "shift": 1, "max_window": 42}}},
    'rkurt': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "shift": 1, "max_window": 42}}},
    'rmax': {"constant_params": {}, "df_params": {"df_daily": {"window": 21, "shift": 1, "max_window": 21}}},
    'high_position_volume': {"constant_params": {"threshold_price": 0.95, "threshold_vol": 1.5}, "df_params": {"df_daily": {"window_price": 60, "window_vol": 20, "window_result": 21, "max_window": 80}}},
    'low_position_volume': {"constant_params": {"threshold_price": 1.05, "threshold_vol": 1.5}, "df_params": {"df_daily": {"window_price": 60, "window_vol": 20, "window_result": 21, "max_window": 80}}},
    'candle_body_ratio': {"constant_params": {}, "df_params": {"df_daily": {"window": 5, "max_window": 5}}},
    'gap_ratio': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "shift": 1, "max_window": 20}}},
    'shadow_ratio': {"constant_params": {}, "df_params": {"df_daily": {"window": 5, "max_window": 5}}},
    'rsi': {"constant_params": {}, "df_params": {"df_daily": {"period": 14, "shift": 1, "max_window": 14}}},
    'macd_hist': {"constant_params": {}, "df_params": {"df_daily": {"fast": 12, "slow": 26, "signal": 9, "max_window": 26}}},
    'boll_width': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "max_window": 20}}},
    'ma_divergence': {"constant_params": {}, "df_params": {"df_daily": {"window_short": 5, "window_long": 60, "max_window": 60}}},
    'boll_bandwidth': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "max_window": 20}}},
    'daily_range': {"constant_params": {}, "df_params": {"df_daily": {}}},
    'effective_spread': {"constant_params": {}, "df_params": {"df_daily": {"window": 20, "max_window": 20}}},
    'adjusted_range': {"constant_params": {}, "df_params": {"df_daily": {}}},
}

# ============================================================================
# 兼容性导出（旧接口，后续将逐步移除）
# ============================================================================
FACTOR_REGISTRY1 = {k: v for k, v in FACTOR_REGISTRY.items() if k in [
    'late_skew_ret', 'down_vol_perc', 'corr_ret_lastret', 'corr_close_nextopen',
    'volume_perc2', 'volume_perc3', 'volume_perc4', 'volume_perc5', 'volume_perc6', 'volume_perc7',
    'early_corr_volume_ret', 'corr_volume_amplitude', 'mmt_last30', 'mmt_qrs', 'mmt_top20VolumeRet',
    'intraday_reversal', 'liq_closevol', 'vol_skew', 'price_range_vol_ratio', 'extreme_time',
    'corr_pv', 'corr_pvl', 'corr_prv', 'corr_trend', 'overnight_pv_mis',
    'trade_top20retRatio', 'ideal_range_cut', 'open_range_ratio', 'intraday_price_efficiency'
]}
FACTOR_REGISTRY2 = {k: v for k, v in FACTOR_REGISTRY.items() if k in [
    'rstr', 'streverse_1m', 'streverse_2m', 'overnight_intraday_rev', 'volume_weighted_rev',
    'rvol_21d', 'vol_upVol_ratio', 'vol_downRatio', 'cmra',
    'abn_turnover', 'turnover_vol', 'gtr', 'turnover_ret_interact', 'abn_turnover_accel',
    'liq_amihud', 'zero_trades_ratio', 'range_adjusted_amihud',
    'vr_20d', 'vol_stability', 'pv_sync_residual',
    'rskew', 'rkurt', 'rmax',
    'high_position_volume', 'low_position_volume', 'candle_body_ratio', 'gap_ratio', 'shadow_ratio',
    'rsi', 'macd_hist', 'boll_width', 'ma_divergence', 'boll_bandwidth',
    'daily_range', 'effective_spread', 'adjusted_range'
]}
FACTOR_REGISTRY3 = {}

FACTOR_NAMES1 = list(FACTOR_REGISTRY1.keys())
FACTOR_NAMES2 = list(FACTOR_REGISTRY2.keys())
FACTOR_NAMES3 = list(FACTOR_REGISTRY3.keys())

PARAMS_REGISTRY1 = {k: v for k, v in PARAMS_REGISTRY.items() if k in FACTOR_NAMES1}
PARAMS_REGISTRY2 = {k: v for k, v in PARAMS_REGISTRY.items() if k in FACTOR_NAMES2}
PARAMS_REGISTRY3 = {}
