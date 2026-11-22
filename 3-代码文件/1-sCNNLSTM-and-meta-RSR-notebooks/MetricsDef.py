from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from sklearn.cross_decomposition import PLSRegression
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error
from scipy.stats import t
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.formula.api as smf
np.set_printoptions(suppress=True, precision=4)
pd.set_option('display.float_format', '{:.4f}'.format)
import math
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别.

#define the clark and west test(2007)
def clark_west_test(actual, pred_A, pred_B, lag=None):
    """
    Clark and West (2007) test for nested model comparison.
    
    Parameters:
    e_A (array-like): Forecast errors from the larger model (Model A)
    e_B (array-like): Forecast errors from the smaller nested model (Model B)
    pred_A (array-like): Predictions from Model A
    pred_B (array-like): Predictions from Model B
    lag (int): Number of lags for Newey-West adjustment. If None, automatic selection.
    
    Returns:
    dict: Test statistic, p-value, Newey-West standard error
    """
    # Convert inputs to numpy arrays
    pred_A = np.asarray(pred_A)
    pred_B = np.asarray(pred_B)
    actual = np.asarray(actual)
    e_A = np.asarray(actual - pred_A)
    e_B = np.asarray(actual - pred_B)
    
    # Validate input lengths
    if not (len(e_A) == len(e_B) == len(pred_A) == len(pred_B)):
        raise ValueError("All input arrays must have the same length.")
    
    # Calculate adjusted difference (Clark & West adjustment)
    adj_diff = e_B**2 - (e_A**2 - (pred_A - pred_B)**2)
    
    # Calculate mean difference
    mean_diff = np.mean(adj_diff)

    # Compute Newey-West standard errors
    n = len(adj_diff)
    if lag is None:
        # Automatic lag selection (Newey-West default)
        lag = int(np.floor(4 * (n / 100) ** (2/9)))
    
    # Use statsmodels to compute HAC standard errors
    nw_results = sm.OLS(adj_diff, np.ones(n)).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    se = nw_results.bse[0]
    
    # Compute test statistic (asymptotically normal)
    test_stat = mean_diff / se
    
    # One-sided p-value (H0: Model A does not outperform Model B; H1: Model A outperforms Model B)
    p_value = 1 - norm.cdf(test_stat)

    if p_value <= 0.01:
        significance = '***'
    elif p_value <= 0.05:
        significance = '**'
    elif p_value <= 0.10:
        significance = '*'
    else:
        significance = ''
    
    return test_stat, p_value, significance

def R2OOS(r_real, r_hat, r_bar):
    denominator_res=(r_real-r_bar)**2
    denominator=np.mean(denominator_res)
    numerator_res=(r_real-r_hat)**2
    numerator=np.mean(numerator_res)
    r2oos=1-numerator/denominator
    return r2oos

def R2OOS_by_USREC(y_true, y_pred, y_bench_pred, USREC):
    """
    计算经济衰退期和扩张期的R²OOS。
    
    参数:
    y_true: 真实值数组
    y_pred: 模型预测值数组
    y_bench_pred: 基准模型预测值数组
    USREC: NBER经济衰退指示器（0=扩张期，1=衰退期）
    
    返回:
    tuple: (衰退期R²OOS, 扩张期R²OOS)
    """
    USREC = USREC.iloc[:, 0]
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    y_bench_pred = pd.Series(y_bench_pred).reset_index(drop=True)
    # 分割数据为衰退期和扩张期
    rec_mask = USREC == 1
    exp_mask = USREC == 0

    # 计算衰退期的R²OOS
    r2_rec = R2OOS(y_true[rec_mask], y_pred[rec_mask], y_bench_pred[rec_mask])

    # 计算扩张期的R²OOS
    r2_exp = R2OOS(y_true[exp_mask], y_pred[exp_mask], y_bench_pred[exp_mask])

    return r2_rec, r2_exp

def clark_west_test_by_USREC(y_true, y_pred, y_bench_pred, USREC):
    USREC = USREC.iloc[:, 0]
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    y_bench_pred = pd.Series(y_bench_pred).reset_index(drop=True)
    # 分割数据为衰退期和扩张期
    rec_mask = USREC == 1
    exp_mask = USREC == 0

    # 计算衰退期的R²OOS
    test_stat_rec, p_value_rec, significance_rec = clark_west_test(y_true[rec_mask], y_pred[rec_mask], y_bench_pred[rec_mask])

    # 计算扩张期的R²OOS
    test_stat_exp, p_value_exp, significance_exp = clark_west_test(y_true[exp_mask], y_pred[exp_mask], y_bench_pred[exp_mask])

    return test_stat_rec, p_value_rec, significance_rec, test_stat_exp, p_value_exp, significance_exp

def stat_sig(p_value):
    if p_value <= 0.01/2:
        significance = '***'
    elif p_value <= 0.05/2:
        significance = '**'
    elif p_value <= 0.10/2:
        significance = '*'
    else:
        significance = ''
    return significance

#计算Hurst指数
def s(inputdata):
    # 输入numpy数组
    n = inputdata.shape[0]
    t = 0
    for i in np.arange(n):
        if i <= (n - 1):
            for j in np.arange(i + 1, n):
                if inputdata[j] > inputdata[i]:
                    t = t + 1
                elif inputdata[j] < inputdata[i]:
                    t = t - 1
                else:
                    t = t
    return t

def beta(inputdata):
    n = inputdata.shape[0]
    t = []
    for i in np.arange(n):
        if i <= (n - 1):
            for j in np.arange(i + 1, n):
                t.append((inputdata[j] - inputdata[i]) / ((j - i) * 1.0))
    return np.median(t)

def Hurst(x):
    # x为numpy数组
    n = x.shape[0]
    t = np.zeros(n - 1)  # t为时间序列的差分
    for i in range(n - 1):
        t[i] = x[i + 1] - x[i]
    mt = np.zeros(n - 1)  # mt为均值序列,i为索引,i+1表示序列从1开始
    for i in range(n - 1):
        mt[i] = np.sum(t[0:i + 1]) / (i + 1)

    # Step3累积离差和极差,r为极差
    r = []
    for i in np.arange(1, n):  # i为tao
        cha = []
        for j in np.arange(1, i + 1):
            if i == 1:
                cha.append(t[j - 1] - mt[i - 1])
            if i > 1:
                if j == 1:
                    cha.append(t[j - 1] - mt[i - 1])
                if j > 1:
                    cha.append(cha[j - 2] + t[j - 1] - mt[i - 1])
        r.append(np.max(cha) - np.min(cha))
    s = []
    for i in np.arange(1, n):
        ss = []
        for j in np.arange(1, i + 1):
            ss.append((t[j - 1] - mt[i - 1]) ** 2)
        s.append(np.sqrt(np.sum(ss) / i))
    r = np.array(r)
    s = np.array(s)
    xdata = np.log(np.arange(2, n))
    ydata = np.log(r[1:] / s[1:])

    h, b = np.polyfit(xdata, ydata, 1)
    return h

def MSE(y,y_hat):
    a = y_hat - y
    b = a ** 2
    c = sum(b)
    d = c / len(y)
    return d
def RMSE(y,y_hat):
    a = y_hat - y
    b = a ** 2
    c = sum(b)
    d = c / len(y)
    e = math.sqrt(d)
    return e
def MAE(y,y_hat):
    a = y_hat - y
    b = abs(a)
    c = sum(b)
    d = c / len(y)
    return d
def SMAPE(y,y_hat):
    a = y_hat - y
    b = abs(a)
    c = (abs(y_hat)+abs(y))/2
    d = b / c
    e = sum(d)
    f = e / len(y)
    return f

def R2(y,y_hat,y_bar):
    b = y - y_hat
    c = sum(b**2)
    d = y - y_bar
    e = sum(d**2)
    f = c / e
    g = 1 - f
    return g

def R_2(y,y_hat):
    a = sum(y) / len(y)
    b = y_hat - y
    c = sum(b**2)
    d = y - a
    e = sum(d**2)
    f = c / e
    g = 1 - f
    return g

def RMSPE(y,y_hat):
    a = (y - y_hat) / y
    b = sum(a**2)
    c = b / len(y)
    d = math.sqrt(c)
    return d

def sharpe_ratio(model_forecasts, true_returns, baseline, rf_rates, risk_aversion, 
                transaction_cost=0, oos_len=48, window=120, return_recursive=False):
    """
    新增 return_recursive 参数控制是否返回递归序列
    返回类型：
    - 默认返回年化ΔSR
    - return_recursive=True时返回 (年化ΔSR, 递归ΔSR序列)
    """
    true_returns_simple = np.exp(true_returns) - 1
    model_forecasts_simple = np.exp(model_forecasts) - 1
    baseline_simple = np.exp(baseline) - 1
    T = oos_len

    def compute_excess_returns(forecasts, tc):
        """返回模型/基准的超额收益列表"""
        excess_returns = []
        prev_weight = 0
        for i in range(T):
            # 计算波动率窗口（动态前推）
            returns_window = true_returns_simple[-(T-i+window):-(T-i)]
            variance = np.var(returns_window, ddof=1)
            
            # 计算权重
            w = forecasts.iloc[-(T-i)] / (risk_aversion * variance)
            w = np.clip(w, 0, 1.5)
            
            # 交易成本
            tc_cost = tc * abs(w - prev_weight)
            
            # 计算超额收益
            market_return = true_returns_simple.iloc[-(T-i)]
            excess = w * (market_return - rf_rates.iloc[-(T-i)]) - tc_cost
            excess_returns.append(excess)
            
            prev_weight = w
        return excess_returns

    # 获取模型和基准的超额收益序列
    model_excess = compute_excess_returns(model_forecasts_simple, transaction_cost)
    bench_excess = compute_excess_returns(baseline_simple, transaction_cost)
    
    # 计算整体ΔSR
    sr_model = np.mean(model_excess)/np.std(model_excess, ddof=1)
    sr_bench = np.mean(bench_excess)/np.std(bench_excess, ddof=1)
    delta_sr = (sr_model - sr_bench) * np.sqrt(12)  # 年化
    
    if return_recursive:
        # 生成递归ΔSR序列
        delta_sr_list = []
        cum_m, cum_b = [], []
        for m, b in zip(model_excess, bench_excess):
            cum_m.append(m)
            cum_b.append(b)
            with warnings.catch_warnings():  # 忽略空值警告
                warnings.simplefilter("ignore")
                sr_m = np.mean(cum_m)/np.std(cum_m, ddof=1) if len(cum_m)>1 else np.nan
                sr_b = np.mean(cum_b)/np.std(cum_b, ddof=1) if len(cum_b)>1 else np.nan
            delta_sr_list.append((sr_m - sr_b) * np.sqrt(12))
        
        # 生成时间索引（使用预测序列的最后T个时间点）
        index = model_forecasts.index[-T+12:] #using 12 month as burn-in period
        return delta_sr, pd.Series(delta_sr_list[-T+12:], index=index) #using 12 month as burn-in period
    else:
        return delta_sr
   
def cer(model_forecasts, true_returns, baseline, rf_rates, risk_aversion, transaction_cost=0, 
        oos_len=48, window=120, return_recursive=False):
    """
    新增 return_recursive 参数控制是否返回递归序列
    返回类型：
    - 默认返回年化ΔCER（单位：基点）
    - return_recursive=True时返回 (年化ΔCER, 递归ΔCER序列)
    """
    true_returns_simple = np.exp(true_returns) - 1
    model_forecasts_simple = np.exp(model_forecasts) - 1
    baseline_simple = np.exp(baseline) - 1
    T = oos_len
    
    def compute_portfolio_returns(forecasts, tc):
        """返回组合收益序列（月度）"""
        portfolio_returns = []
        prev_weight = 0
        for i in range(T):
            # 动态前推窗口计算方差
            returns_window = true_returns_simple[-(T - i + window): -(T - i)]
            variance = np.var(returns_window, ddof=1)
            
            # 计算权重并截断
            w = forecasts.iloc[-(T - i)] / (risk_aversion * variance)
            w = np.clip(w, 0, 1.5)
            
            # 交易成本
            tc_cost = tc * abs(w - prev_weight)
            
            # 计算组合收益
            market_return = true_returns_simple.iloc[-(T - i)]
            portfolio_return = w * market_return + (1 - w) * rf_rates.iloc[-(T - i)] - tc_cost
            portfolio_returns.append(portfolio_return)
            
            prev_weight = w  # 更新权重
        return portfolio_returns
    
    # 获取模型和基准的组合收益序列
    model_returns = compute_portfolio_returns(model_forecasts_simple, transaction_cost)
    bench_returns = compute_portfolio_returns(baseline_simple, transaction_cost)
    
    # 计算整体ΔCER（年化基点）
    def compute_cer(returns):
        """计算单期的CER（需至少2个观测值计算方差）"""
        if len(returns) < 2:
            return np.nan
        mu = np.mean(returns)
        sigma_sq = np.var(returns, ddof=1)
        return mu - (risk_aversion / 2) * sigma_sq
    
    cer_model = compute_cer(model_returns)
    cer_bench = compute_cer(bench_returns)
    delta_cer = (cer_model - cer_bench) * 1200  # 年化并转换为基点
    
    if return_recursive:
        # 生成递归ΔCER序列
        delta_cer_list = []
        for t in range(13, T+1): #using 12 month as burn-in period
            cer_m = compute_cer(model_returns[:t])
            cer_b = compute_cer(bench_returns[:t])
            delta = (cer_m - cer_b) * 1200 if not np.isnan(cer_m) and not np.isnan(cer_b) else np.nan
            delta_cer_list.append(delta)
        
        # 生成时间索引（使用预测序列的最后T个时间点）
        index = model_forecasts.index[-T+12:]
        return delta_cer, pd.Series(delta_cer_list, index=index)
    else:
        return delta_cer