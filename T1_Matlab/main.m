function extract_and_store_features()
    % 设置文件路径
    source_data_dir = 'C:\Users\khand\OneDrive\code\rail-bearing-fault-diagnosis\data\source_mat';
    target_data_dir = 'C:\Users\khand\OneDrive\code\rail-bearing-fault-diagnosis\data\target_mat';
    target = true;
    if target == false
        dirx = source_data_dir;
    else
        dirx = target_data_dir;
    end
    % 获取所有.mat文件路径
    mat_files = dir(fullfile(dirx, '**', '*.mat'));  % 遍历子目录中的所有.mat文件

    % 初始化结果存储
    all_features = [];

    % 遍历所有.mat文件
    for i = 1:length(mat_files)
        mat_file = mat_files(i);
        file_path = fullfile(mat_file.folder, mat_file.name);
        % 加载.mat文件
        data = load(file_path);
        var_names = fieldnames(data);
        if ~target
            % 提取转频=转速/60
            fr = 0;
            for j = 1:length(var_names)
                var_name = var_names{j};  
                if contains(var_name, 'RPM')  
                    rpm = data.(var_name); 
                    fr = rpm/60;
                    var_names(contains(var_names,'RPM')) = [];
                    break;
                end
            end

            % 提取文件名中的故障信息
            [position, sample_f, bearing_position, fault_size] = extract_fault_info(file_path);

            for j = 1:length(var_names)
                var_name = var_names{j};
                
                if contains(var_name, 'DE')  
                    % 获取时间序列
                    whole_signal = data.(var_name);
                    whole_signal = change_samplerate(whole_signal,sample_f*1000);
                    f = cal_fault_f(position,bearing_position,fr);
                    win_len = 32000;
                    hop_len = win_len/2;
                    segments = slice_signal(whole_signal, win_len, hop_len);

                    for s = 1:numel(segments)
                        seg = segments{s};
                        features = get_features(seg, 32000, fr);
                        % 将故障信息和特征合并成一个观察对象
                        observation = {position, bearing_position, rpm,f};
                        for w = 1:length(features) 
                            var = features(w); 
                            observation = [observation,var]; 
                        end 
                        all_features = [all_features; observation]; 
                    end                  
                end
            end
        else
            %目标
            signal = data.(var_names{1});
            win_len = 16000;
            hop_len = win_len/2;
            segments = slice_signal(signal, win_len, hop_len);
            
            fr = 600;
            for s = 1:numel(segments)
                seg = segments{s}; 
                features = get_features(seg, 32000, fr);
                all_features = [all_features; features];
            end
        end   
    end

    % 将结果保存到一个表格中
    result_table = cell2table(all_features);
    writetable(result_table, 'target_features.csv');  % 保存为CSV文件

    % writematrix(all_features, 'output.csv');
end

function [position,sample_f, bearing_position, fault_size] = extract_fault_info(file_path)
    % 从文件路径中提取故障信息
    % 文件路径格式示例：
    % C:\Users\khand\OneDrive\code\rail-bearing-fault-diagnosis\data\source_mat\12kHz_DE_data\B\0007
    
    % 获取文件夹路径中的各个部分
    parts = strsplit(file_path, '\');  % 分割路径
    parts = parts(9:end);
    
    fault_position = parts{1};
    if contains(fault_position, 'DE')
        position = 'DE';
    elseif contains(fault_position, 'FE')
        position = 'FE';
    elseif contains(fault_position, 'N')
        position = 'N';
    else
        position = -1;
    end

    if contains(fault_position, '12kHz')
        sample_f = 12;
    elseif contains(fault_position, '48kHz')
        sample_f = 48;
    else
        sample_f = -1;
    end

    [~,n_var] = size(parts);
    if n_var > 2
        % 提取轴承位置（B, IR, OR）
        bearing_position = parts{2};
        % 提取故障大小（如 0.007）
        fault_size_str = parts{end-1};  % 例如 '0007'
        fault_size = str2double(fault_size_str) / 1000;  % 假设故障大小是文件夹名称的最后三位数，转化为数值
    else
        bearing_position = -1;
        fault_size = 0;
    end
end

% ===== 切片 =====
function segments = slice_signal(x, win_len, hop_len)
    n = length(x);
    if n < win_len
        segments = {x};
        return;
    end
    starts = 1:hop_len:(n - win_len + 1);
    segments = cell(numel(starts), 1);
    for k = 1:numel(starts)
        idx = starts(k):(starts(k) + win_len - 1);
        segments{k} = x(idx);
    end
end

% ===== 重采样到 32kHz =====
function y_resampled = change_samplerate(x, sample_f)
    % x         : 原始信号（列向量）
    % sample_f  : 原采样频率
    % 输出 y_resampled : 变换到 32kHz 的信号
    
    target_f = 32000;   % 目标采样率
    % 计算整数比率
    [p, q] = rat(target_f / sample_f, 1e-6);
    
    % 使用抗混叠滤波器 + 重采样
    y_resampled = resample(x, p, q);
end

extract_and_store_features()