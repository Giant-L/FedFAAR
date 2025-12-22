import torch
import torch.nn.functional as F
import numpy as np

# 1. 线性混合基准 (Linear Mixing Baseline)
def score_linear_mixing(act_male, act_female, alpha=0.5):
    # act shape: [Batch, Channel]
    # 我们需要在 Batch 维度 (dim=0) 上聚合，保留 Channel 维度
    
    # 激活率
    rate_m = (act_male > 0).float().mean(dim=0)
    rate_f = (act_female > 0).float().mean(dim=0)
    diff_rate = torch.abs(rate_m - rate_f)
    
    # 均值
    mu_m = act_male.mean(dim=0)
    mu_f = act_female.mean(dim=0)
    diff_mu = torch.abs(mu_m - mu_f)
    
    return (1 - alpha) * diff_mu + alpha * diff_rate

# 2. 混合对数几率 (Mixed Log-Odds)
def score_mixed_log_odds(act_male, act_female, eps=1e-6):
    rate_m = (act_male > 0).float().mean(dim=0)
    rate_f = (act_female > 0).float().mean(dim=0)
    
    logit_m = torch.log((rate_m + eps) / (1 - rate_m + eps))
    logit_f = torch.log((rate_f + eps) / (1 - rate_f + eps))
    
    diff_logit = torch.abs(logit_m - logit_f)
    return diff_logit

# 3. Shape + Rate (Wasserstein + Rate)
def score_shape_rate(act_male, act_female, alpha=0.5):
    # 计算 1D Wasserstein
    # 对 Batch 维度 (dim=0) 进行排序，这样每个 Channel 内部的数据是有序的
    # 假设 probe 是平衡的，act_male 和 act_female 长度一致 (都是 1000)
    
    if act_male.shape[0] == act_female.shape[0]:
        m_sorted, _ = torch.sort(act_male, dim=0)
        f_sorted, _ = torch.sort(act_female, dim=0)
        # 在 Batch 维度上求平均，得到每个 Channel 的平均距离 -> Shape: [Channel]
        wass_dist = torch.mean(torch.abs(m_sorted - f_sorted), dim=0)
    else:
        # 如果长度不一致，使用分位数近似
        qs = torch.linspace(0, 1, 100).to(act_male.device)
        m_q = torch.quantile(act_male, qs, dim=0) # [100, Channel]
        f_q = torch.quantile(act_female, qs, dim=0)
        wass_dist = torch.mean(torch.abs(m_q - f_q), dim=0) # [Channel]

    rate_m = (act_male > 0).float().mean(dim=0)
    rate_f = (act_female > 0).float().mean(dim=0)
    diff_rate = torch.abs(rate_m - rate_f)
    
    return (1 - alpha) * wass_dist + alpha * diff_rate

# 4. 复杂公式 (Fisher-like)
def score_complex_fisher(act_male, act_female):
    mu_m, var_m = act_male.mean(dim=0), act_male.var(dim=0)
    mu_f, var_f = act_female.mean(dim=0), act_female.var(dim=0)
    
    numerator = torch.pow(mu_m - mu_f, 2)
    denominator = var_m + var_f + 1e-8
    
    fisher_score = numerator / denominator
    
    rate_diff = torch.abs((act_male > 0).float().mean(dim=0) - (act_female > 0).float().mean(dim=0))
    
    return fisher_score * (1 + rate_diff)

# 5. 零膨胀 JSD (Zero-Inflated JSD)
# 必须对每个 Channel 单独算，避免 Flatten
def score_zero_inflated_jsd(act_male, act_female, bins=20):
    num_channels = act_male.shape[1]
    scores = []
    
    # ⚠️ 为了保证正确性，这里对 Channel 进行循环 (64次循环很快)
    # 因为 vectorize 这种动态非零部分的 hist 比较麻烦
    
    # 全局最大值，用于统一直方图范围
    max_val = max(act_male.max().item(), act_female.max().item())
    
    for c in range(num_channels):
        am = act_male[:, c]
        af = act_female[:, c]
        
        # 零概率
        p0_m = (am <= 0).float().mean()
        p0_f = (af <= 0).float().mean()
        
        # 零部分的距离
        jsd_zero = torch.abs(p0_m - p0_f)
        
        # 连续部分 (非零)
        am_nz = am[am > 0]
        af_nz = af[af > 0]
        
        if len(am_nz) < 5 or len(af_nz) < 5:
            # 如果某一方几乎全是0，直接用零距离
            scores.append(jsd_zero)
            continue
            
        hist_m = torch.histc(am_nz, bins=bins, min=0, max=max_val)
        prob_m = hist_m / (hist_m.sum() + 1e-8)
        
        hist_f = torch.histc(af_nz, bins=bins, min=0, max=max_val)
        prob_f = hist_f / (hist_f.sum() + 1e-8)
        
        # JS Divergence
        m = 0.5 * (prob_m + prob_f)
        kl_m = F.kl_div(torch.log(prob_m + 1e-8), m, reduction='sum')
        kl_f = F.kl_div(torch.log(prob_f + 1e-8), m, reduction='sum')
        jsd_cont = 0.5 * (kl_m + kl_f)
        
        # 组合
        scores.append(0.5 * jsd_zero + 0.5 * jsd_cont)
        
    return torch.tensor(scores).to(act_male.device)

# 调度中心
def get_doubt_score(act_male, act_female, method='shape_rate'):
    if method == 'linear':
        return score_linear_mixing(act_male, act_female)
    elif method == 'log_odds':
        return score_mixed_log_odds(act_male, act_female)
    elif method == 'shape_rate':
        return score_shape_rate(act_male, act_female)
    elif method == 'complex':
        return score_complex_fisher(act_male, act_female)
    elif method == 'jsd':
        return score_zero_inflated_jsd(act_male, act_female)
    else:
        raise ValueError(f"Unknown method: {method}")