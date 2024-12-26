import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
import scipy.optimize as sco
from scipy.optimize import minimize

np.set_printoptions(precision=4, suppress=True)

##############Part1.用于模型的数据###############

# 用日对数收益率计算
df = pd.read_csv('../原始数据/daily_ln_index_yield.csv', index_col='time')  # 将日期作为index
df1 = pd.read_csv('../原始数据/daily_ln_index_yield.csv')
# 用日收盘价计算
df_closing = pd.read_csv('../原始数据/daily_closing_price.csv', index_col='time')  # 将日期作为index
df = pd.read_excel('../原始数据/daily_ln_index_yield.xlsx',index_col = 'time') #将日期作为index
df1 = pd.read_excel('../原始数据/daily_ln_index_yield.xlsx')
#用日收盘价计算
df_closing = pd.read_excel('../原始数据/daily_closing_price.xlsx',index_col = 'time') #将日期作为index

# 新增季度序列，按季度进行groupby
df1['Qtr'] = pd.PeriodIndex(pd.to_datetime(df1['time']), freq='Q')  # 生成季度序列
grouped = df1['time'].groupby(df1['Qtr']).tail(1)  # 按季度聚合，找到每季度最后一个交易日

# 取出每季度最后一个交易日对应的索引值
last_traday_index = list([i for i in grouped.index])  # 将最后一个交易日的索引值放入列表中
last_traday_index = last_traday_index[4:]  # 去掉2010.3.31之前的索引


##############Part2.模型衡量指标##############
# stats_closing函数来记录组合累计收益率（使用日收盘价进行计算的指标）
def stats_closing(weights):
    weights = np.array(weights)
    cum_returns = []
    for i in range(7):
        cum_returns.append(math.log(
            closing_price.tail(1).iloc[:, i].values / closing_price.head(1).iloc[:, i].values))  # 用对数收益率表示累计收益率
    port_cum_returns = np.sum(np.array(cum_returns).T * weights)  # 组合累计收益率
    return np.array(port_cum_returns)


# MaxDrawdown函数来记录组合最大回撤率（使用日收盘价进行计算的指标）
def MaxDrawdown_Single(return_list):  # 计算单个序列的最大回撤率
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))
    # np.maximum.accumulate函数可以生成一列当日之前历史最高价值的序列，在当日价值与历史最高值的比例最小时，就是最大回撤结束的终止点
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return math.log(return_list[j] / return_list[i])  # 计算该序列最大回撤率（对数收益率）


def MaxDrawdown(weights):
    weights = np.array(weights)
    maxdrawdown = []
    for i in range(7):
        return_list = closing_price.iloc[:, i]
        maxdrawdown.append(MaxDrawdown_Single(return_list))
    port_maxdrawdown = np.sum(np.array(maxdrawdown).T * weights)  # 组合最大回撤率
    return np.array(port_maxdrawdown)


# stats_ln函数来记录重要的投资组合统计数据（组合年化收益率，组合年化波动率、组合年化夏普比率、组合Calmar比率、组合最大分散化率）（使用日对数收益率进行计算的指标）
def stats_ln(weights):
    weights = np.array(weights)
    port_returns = np.dot(weights.T, log_returns.mean()) * 252  # 组合年化收益率
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))  # 组合的年化波动率
    port_sharpe = (port_returns - 0.03) / port_volatility  # 组合年化夏普比率
    port_calmar = (port_returns / MaxDrawdown(weights))  # 组合Calmar比率
    port_MD = np.dot(weights.T, log_returns.std()) * np.sqrt(252) / port_volatility  # 组合最大分散化率
    return np.array([port_returns, port_volatility, port_sharpe, port_calmar, port_MD])


##############Part3.最优化模型##############

##########风险平价模型###########
# 计算单个资产对总体风险贡献度RC
# 计算单个资产对总体风险贡献度RC
def calculate_risk_contribution(weights):
    weights = np.array(weights)
    V = log_returns.cov()  # 协方差矩阵
    sigma = np.sqrt(np.dot(weights.T, np.dot(V, weights)))  # 资产总风险贡献
    RC = np.multiply(np.dot(feature_vector.T, weights.T),
                     np.dot(feature_vector.T, V) * weights.T) / sigma  # 单个资产对投资组合的风险贡献,np.multiply为矩阵对应位置的元素相乘
    return RC


# 最优化模型，使得对任意的i，j，均有RCi = RCj
def risk_budget_objective(weights):
    weights = np.array(weights)
    V = log_returns.cov()  # 协方差矩阵
    x_t = 7 * [1. / 7]  # 组合中资产预期风险贡献度的目标向量，风险平价模型中各资产预期风险贡献度均相等
    sig_p = np.sqrt(np.dot(weights.T, np.dot(V, weights)))  # 组合波动率
    risk_target = np.asmatrix(np.multiply(sig_p, x_t)) * 10000  # 风险平价模型要达到的目标RC，即将sigma平均分，每个资产的风险贡献度相同
    real_RC = np.asmatrix(calculate_risk_contribution(weights)) * 10000  # 实际的RC
    J = sum(np.square(real_RC - risk_target.T))[0, 0]  # 最小化二者的误差平方和
    return J  # 最优化的目标即是最小化误差平方和

##########多目标优化模型###########
def single_target(weights):
    port_returns = np.dot(weights.T, log_returns.mean()) * 252  # 组合年化收益率
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))  # 组合的年化波动率
    ratio = (port_returns - 0.03) / port_volatility  # 组合年化夏普比率

    if ratio < 0:
        return risk_budget_objective(weights) - 19.1 * ratio
    else:
        return risk_budget_objective(weights) - 19.1 * math.log(-1 + math.exp(ratio))
        #return risk_budget_objective(weights) - 19.1 * math.log(ratio)
# 不求解主成分分析的最优投资权重，而直接求解原模型的最优投资权重

###############Part4.滚动窗口计算最优权重####################
# 创建一个空的dataframe，放每一期滚动窗口对应的风险收益指标
df_pcarp_index = pd.DataFrame(
    columns=['pcarp_port_returns', 'pcarp_port_volatility', 'pcarp_port_maxdrawdown', 'pcarp_port_sharpe',
             'pcarp_port_calmar'])
# 分每一期统计时，不统计累计收益率指标

# 存储滚动窗口每一期的最优权重数据
df_pcarp_weights = pd.DataFrame(columns=['hs_300', 'zz_500', 'sz_bond', 'sw_gold', 'nh_goods', 'us_spx', 'hk_hsi'])

# 存储滚动窗口每一期主成分的特征值数据
df_pcarp_eigenvalue = pd.DataFrame(columns=['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7'])

# 滚动窗口计算投资组合每一期的最优权重，并计算每一期的风险收益指标，将其放入一个大的表格里
# 计算2009年基期的最优权重
log_returns = df.iloc[:244, :]  # 取整个2009年的日对数收益率数据
closing_price = df_closing.iloc[:244, :]  # 取整个2009年的日收盘价数据

# 主成分分析处理
pca = PCA(n_components=7)  # 与变量个数相同的主成分
# pca.fit(log_returns)
pca_log_returns = pca.fit_transform(log_returns)  # fit_transform生成降维后的数据，主成分分析处理后的日对数收益率数据
pca_log_returns = pd.DataFrame(pca_log_returns)
pca_log_returns.columns = log_returns.columns
pca_log_returns.index = log_returns.index  # 将pca_log_returns由numpy.ndarray类型转变为dataframe类型，以便使用pca_log_return.cov()
# pca_closing_price = pca(closing_price,7) #主成分分析处理后的日收盘价数据

feature_vector = pca.components_  # 降维后特征向量

# 生成一维投资组合初始权重向量（长度为7，与资产数量相等）
w0 = np.ones(shape=(7,)) / 7  # 给定初始权重

bnds = tuple((0, 1) for x in range(7))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

opt_pcarp = minimize(single_target,
                     w0,
                     method='SLSQP',
                     bounds=bnds,
                     constraints=cons,
                     options={'disp': False})  # disp指是否显示过程（True则显示）

opt_weights_2009 = np.array([m for m in opt_pcarp['x']])  # 基期的最优权重，应用于第一期原始数据
# opt_weights = np.dot(feature_vector, pca_opt_weights.T).T #反推出原资产的最优投资权重


# 滚动窗口计算
# 让通过2009年基期数据计算出来的最优权重作用于第一期原数据
acc_opt_weights = w0
opt_weights = opt_weights_2009

for i in last_traday_index:
    j = i - 251  # 选择每季度最后一个交易日前252天的数据
    acc_opt_weights = opt_weights * 0.1 + acc_opt_weights * 0.9  # acc_opt_weights为历史累计的权重
    log_returns = np.multiply(df.iloc[j:i + 1, :], acc_opt_weights)  # 调整第n期的原数据
    #(df_closing.iloc[j:i + 1, :].shape)
    #print(acc_opt_weights.shape)
    closing_price = np.multiply(df_closing.iloc[j:i + 1, :], acc_opt_weights)  # 调整第n期的原数据
    # 主成分分析处理
    pca = PCA(n_components=7)  # 与变量个数相同的主成分
    pca_log_returns = pca.fit_transform(log_returns)  # fit_transform生成降维后的数据，主成分分析处理后的日对数收益率数据
    pca_log_returns = pd.DataFrame(pca_log_returns)
    pca_log_returns.columns = log_returns.columns
    pca_log_returns.index = log_returns.index  # 将pca_log_returns由numpy.ndarray类型转变为dataframe类型，以便使用pca_log_return.cov()

    feature_vector = pca.components_  # 降维后特征向量

    df_pcarp_eigenvalue.loc[i, :] = pca.explained_variance_  # 特征向量的方差即为特征值

    # 最优化模型
    opt_pcarp = minimize(single_target,
                         w0,
                         method='SLSQP',
                         bounds=bnds,
                         constraints=cons,
                         options={'disp': False})  # disp指是否显示过程（True则显示）

    opt_weights = np.array([m for m in opt_pcarp['x']])  # 直接求解出原资产的最优投资权重
    # opt_weights = np.dot(feature_vector, pca_opt_weights.T).T #反推出原资产的最优投资权重

    # 存储每一期最优权重数据
    df_pcarp_weights.loc[i, :] = acc_opt_weights  # 放入每一期的最优权重数据

# 计算每一期的风险收益指标（使用最优权重，以及未经最优权重改变的原始数据）
for i in last_traday_index:
    j = i - 251  # 选择每季度最后一个交易日前252天的数据
    log_returns = df.iloc[j:i + 1, :]
    closing_price = df_closing.iloc[j:i + 1, :]
    acc_opt_weights = df_pcarp_weights.loc[i, :]
    # 存储每一期风险收益指标
    # df_pcarp_index.loc[i,'pcarp_port_cum_returns'] = stats_closing(acc_opt_weights) #组合累计收益率
    df_pcarp_index.loc[i, 'pcarp_port_returns'] = stats_ln(acc_opt_weights)[0]  # 组合年化收益率
    df_pcarp_index.loc[i, 'pcarp_port_volatility'] = stats_ln(acc_opt_weights)[1]  # 组合年化波动率
    df_pcarp_index.loc[i, 'pcarp_port_maxdrawdown'] = MaxDrawdown(acc_opt_weights)  # 组合最大回撤率
    df_pcarp_index.loc[i, 'pcarp_port_sharpe'] = stats_ln(acc_opt_weights)[2]  # 组合年化夏普比率
    df_pcarp_index.loc[i, 'pcarp_port_calmar'] = stats_ln(acc_opt_weights)[3]  # 组合Calmar比率

####新增一列time，并将time放在第一列
df_pcarp_weights['time'] = grouped.values[4:, ]
df_pcarp_weights = df_pcarp_weights.iloc[:, [7, 0, 1, 2, 3, 4, 5, 6]]  # 每一期的最优权重
df_pcarp_index['time'] = grouped.values[4:, ]
df_pcarp_index = df_pcarp_index.iloc[:, [5, 0, 1, 2, 3, 4]]  # 每一期的风险收益指标
df_pcarp_eigenvalue['time'] = grouped.values[4:, ]
df_pcarp_eigenvalue = df_pcarp_eigenvalue.iloc[:, [7, 0, 1, 2, 3, 4, 5, 6]]  # 每一期的主成分特征值

####求组合的日收盘价（原始日收盘价*最优权重求和）
# 处理2010.1.1-2010.3.31的数据
df_closing.iloc[244:302, :] = np.multiply(df_closing.iloc[244:302, :], df_pcarp_weights.loc[
    301, ['hs_300', 'zz_500', 'sz_bond', 'sw_gold', 'nh_goods', 'us_spx', 'hk_hsi']])
# 有一列是time，所以不能直接写df_mv_weights.loc[301,:]

# 从2010.3.31开始处理
for i in range(len(last_traday_index)):
    a = last_traday_index[i]
    if a != last_traday_index[-1]:
        b = last_traday_index[i + 1]
        df_closing.iloc[a + 1:b + 1, :] = np.multiply(df_closing.iloc[a + 1:b + 1, :], df_pcarp_weights.loc[
            b, ['hs_300', 'zz_500', 'sz_bond', 'sw_gold', 'nh_goods', 'us_spx', 'hk_hsi']])
    else:
        break

df_closing = df_closing.iloc[244:, :]  # 去掉基期的数据
df_closing['sum_p'] = df_closing.sum(axis=1)  # 求组合的日收盘价

####导出csv文件
df_pcarp_weights.to_csv('../自己跑的/主成分风险平价模型_每期最优权重.csv')
df_pcarp_index.to_csv('../自己跑的/主成分风险平价模型_每期风险收益指标.csv')
df_pcarp_eigenvalue.to_csv('../自己跑的/主成分风险平价模型_每期主成分特征值.csv')
df_closing.to_csv('../自己跑的/主成分风险平价模型_日收盘价.csv')

mean_returns = df_pcarp_index['pcarp_port_returns'].mean()
mean_volatility = df_pcarp_index['pcarp_port_volatility'].mean()
mean_maxdrawdown = df_pcarp_index['pcarp_port_maxdrawdown'].mean()
mean_sharpe = df_pcarp_index['pcarp_port_sharpe'].mean()
mean_calmar = df_pcarp_index['pcarp_port_calmar'].mean()

    # 打印每个指标的均值

print(f"组合年化收益率的均值: {mean_returns}")
print(f"组合年化波动率的均值: {mean_volatility}")
print(f"组合最大回撤率的均值: {mean_maxdrawdown}")
print(f"组合年化夏普比率的均值: {mean_sharpe}")
print(f"组合Calmar比率的均值: {mean_calmar}")



