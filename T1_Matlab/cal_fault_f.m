function f0 = cal_fault_f(position, bearing_position, fr)
% 计算并返回指定损伤位置的特征频率
% 输入:
%   position          : 'DE' | 'FE' | 'N'   （采集/安装位置）
%   bearing_position  : 'IR' | 'OR' | 'B' | '-1'（损伤部位；-1 视为无损伤）
%   fr                : 转轴频率(Hz)，通常由 RPM/60 得到
%
% 输出:
%   f0   : 选中的特征频率（根据 bearing_position 返回 BPFO/BPFI/BSF；若无损伤返回 -1）
%   BPFO : 外圈故障特征频率
%   BPFI : 内圈故障特征频率
%   BSF  : 滚动体故障特征频率
%
% 说明:
%   - DE 使用参数组1，FE 使用参数组2，N 或 -1 直接返回 -1
%   - 所有尺寸单位保持与原始代码一致（英寸），公式与原版相同
%
% 示例:
%   [f0, BPFO, BPFI, BSF] = cal_fault_f('DE','IR', fr);

    % -------- 输入标准化 --------
    if ischar(position);         position = upper(string(position));        end
    if ischar(bearing_position); bearing_position = upper(string(bearing_position)); end
    position = upper(string(position));
    bearing_position = upper(string(bearing_position));

    % -------- N 或无损伤直接返回 --------
    if position == "N" || bearing_position == "-1" || fr <= 0
        f0   = -1;
        return;
    end

    % -------- 按位置选择轴承参数 --------
    % 参数组1（DE）
    d_DE = 0.3126;   % 滚动体直径（英寸）
    D_DE = 1.537;    % 轴承内径（英寸）
    % 参数组2（FE）
    d_FE = 0.2656;   % 滚动体直径（英寸）
    D_FE = 1.122;    % 轴承内径（英寸）

    Nd = 9;          % 滚动体数（两组一致）

    switch position
        case "DE"
            d = d_DE; D = D_DE;
        case "FE"
            d = d_FE; D = D_FE;
        otherwise
            % 其他未知位置按无效处理
            f0   = -1;
            return;
    end

    % -------- 计算三类特征频率 --------
    % 外圈：Ball Pass Frequency Outer race
    BPFO = fr * (Nd/2) * (1 - (d/D));
    % 内圈：Ball Pass Frequency Inner race
    BPFI = fr * (Nd/2) * (1 + (d/D));
    % 滚动体：Ball Spin Frequency
    BSF  = fr * (D/d)   * (1 - (d/D)^2);

    % -------- 根据损伤部位选择返回 --------
    switch bearing_position
        case "OR"
            f0 = BPFO;
        case "IR"
            f0 = BPFI;
        case "B"
            f0 = BSF;
        otherwise
            f0 = -1;  % 未知标签按无效
    end
end
