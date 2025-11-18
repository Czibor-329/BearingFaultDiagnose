
function features = get_features(signal, fs, fr)
    % 时域特征提取
    RMS = rms(signal);  % RMS
    var_signal = var(signal);  % 方差
    std_signal = std(signal);  % 标准差
    skewness_signal = skewness(signal);  % 偏度
    kurtosis_signal = kurtosis(signal);  % 峭度
    peak_factor = max(signal) / rms(signal);  % 峰值因子
    pulse_factor = max(abs(signal)) / mean(abs(signal));  % 脉冲因子
    margin_factor = max(abs(signal)) / std_signal;  % 裕度因子
    mean_offset = mean(signal);  % 偏移量（均值）
    peak_to_peak_ratio = (max(signal) - min(signal)) / rms(signal);  % 峰峰比

    % 频域特征提取
    [fft_vals, f] = fft_analysis(signal, fs);
    [peak_freq, peak_val] = find_peak_frequency(fft_vals, f);
    
    % BPFO、BPFI、BSF 及其谐波的峰值
    %BPF = cal_fault_f(p, bp, fr);
    %BPFO_peak = find_harmonic_peak(fft_vals, f, BPF);

    %turn_energy_ratio = turn_frequency_energy_ratio(fft_vals, f, fr);
    envelope_spectrum = envelope_spectrum_analysis(signal, fs);

    % 额外频域特征
    total_energy = sum(abs(fft_vals).^2);  % 总能量
    spectral_entropy = spectral_entropy_analysis(fft_vals);  % 频谱熵
    band_energy_ratio = calculate_band_energy_ratio(fft_vals, f, 0, 1000);  % 频带能量比

    % 返回特征数据为一行
    features = [RMS, var_signal, std_signal, skewness_signal, kurtosis_signal, ...
                peak_factor, pulse_factor, margin_factor, mean_offset, peak_to_peak_ratio, ...
                peak_freq, peak_val, envelope_spectrum, ...
                total_energy, spectral_entropy, band_energy_ratio];
end

% 滑动窗口特征提取函数（平均）




% 新增的频域特征函数
function entropy = spectral_entropy_analysis(fft_vals)
    magnitude_spectrum = abs(fft_vals).^2;  % 计算幅度谱的能量
    total_energy = sum(magnitude_spectrum);  % 频谱的总能量
    normalized_spectrum = magnitude_spectrum / total_energy;  % 归一化
    entropy = -sum(normalized_spectrum .* log(normalized_spectrum + eps));  % 计算熵
end

function band_ratio = calculate_band_energy_ratio(fft_vals, f, low_freq, high_freq)
    idx_band = f >= low_freq & f <= high_freq;
    band_energy = sum(abs(fft_vals(idx_band)).^2);  % 该频带的能量
    total_energy = sum(abs(fft_vals).^2);  % 总能量
    band_ratio = band_energy / total_energy;  % 频带能量与总能量的比值
end


% 频谱分析函数
function [fft_vals, f] = fft_analysis(signal, fs)
    N = length(signal);
    fft_vals = abs(fft(signal));  % 计算FFT幅度
    fft_vals = fft_vals(1:floor(N/2));  % 只保留正频率部分
    f = (0:N/2-1) * fs / N;  % 频率轴
end

% 查找主峰频率和幅值
function [peak_freq, peak_val] = find_peak_frequency(fft_vals, f)
    [peak_val, idx] = max(fft_vals);  % 找到最大幅值
    peak_freq = f(idx);  % 对应的频率
end

% 查找指定频率的谐波峰值
function peak = find_harmonic_peak(fft_vals, f, fault_freq)
    idx = find(abs(f - fault_freq) < 1);  % 查找最接近的频率点
    peak = fft_vals(idx);  % 返回该频率点的幅值
end

% 转频边带能量比计算
function [ratio] = turn_frequency_energy_ratio(fft_vals, f, turn_frequency)
    % 计算转频边带的能量比
    idx = find(abs(f - turn_frequency) < 1);
    left_band = fft_vals(idx-1);
    right_band = fft_vals(idx+1);
    ratio = left_band / right_band;
end

% 计算指定频带的能量比例
function ratio = energy_ratio_in_band(fft_vals, f, low, high)
    idx = (f >= low & f <= high);
    band_energy = sum(fft_vals(idx).^2);  % 频带能量
    total_energy = sum(fft_vals.^2);  % 总能量
    ratio = band_energy / total_energy;  % 能量比例
end

% 包络谱分析（假设使用 Hilbert 包络分析）
function [envelope_spectrum] = envelope_spectrum_analysis(signal, fs)
    analytic_signal = hilbert(signal);  % 获取解析信号
    envelope = abs(analytic_signal);  % 获取包络
    [fft_vals, f] = fft_analysis(envelope, fs);
    envelope_spectrum = sum(fft_vals);  % 返回包络谱总能量
end

function [BPFO, BPFI, BSF] = calculate_fault_frequencies(type, fr)
    
    d = 0.3126;  % 滚动体直径（单位：英寸）
    D = 1.537;   % 轴承内径（单位：英寸）
    Nd = 9;      % 滚动体数
    if type == 2
        d = 0.2656;  % 滚动体直径（单位：英寸）
        D = 1.122;   % 轴承内径（单位：英寸）
    end
    
    % 计算外圈故障特征频率 BPFO
    BPFO = fr * (Nd / 2) * (1 - (d / D)); 
    % 计算内圈故障特征频率 BPFI
    BPFI = fr * (Nd / 2) * (1 + (d / D));
    % 计算滚动体故障特征频率 BSF
    BSF = fr * (D / d) * (1 - (d / D)^2);
end

