%求运动物体的range-speed map 
% 清理工作区
clear;
clc;

% ----------------------- 雷达参数设置 -----------------------
c = 3e8;  % 光速 (m/s)

% RF 参数
p.startFreq = 60;       % GHz
p.chirpTime = 360;      % us
p.sweepSlope = 3000 / 57; % MHz/us
p.sweepTime = 57;       % us
p.sweepBandwidth = p.sweepSlope * p.sweepTime; % MHz

% 天线参数
p.nTxAnt = 1;
p.nRxAnt = 4;
p.nAnt = p.nTxAnt * p.nRxAnt;

% ADC 参数
p.sampleRate = 10;      % MHz
p.nSample = 512;        % 每个 chirp 的采样点数
p.nChirp = 16;          % 每帧 chirp 数
p.sampleTime = p.nSample / p.sampleRate; % 每个 chirp 的时间 (us)
p.sampleBandwidth = p.sampleTime * p.sweepSlope; % 带宽 (MHz)
p.rangeRes = c / (2 * p.sampleBandwidth * 1e6); % 距离分辨率 (m)
p.dopRes = c / (2 * p.startFreq * 1e9 * p.nChirp * p.chirpTime * 1e-6 * p.nTxAnt); % 多普勒分辨率 (m/s)

% ----------------------- 数据加载和处理 -----------------------
% 加载数据
% data = load('adcSampleAll_run.mat');
data = load('adcSampleAll.mat');
adcSampleAll = data.adcSampleAll;  % 加载雷达数据

% 重塑数据 [Range, Chirp, Antenna, Frame]
adcSampleAll = reshape(adcSampleAll, p.nSample, p.nChirp, p.nRxAnt, []);

% FFT 参数
nFFT = 512;  % FFT 点数
nsFFT = 16;  % chirp点数
% 选择特定天线数据（例如第4个天线）
selectedAntenna = 4; % 第4个天线
antennaData = adcSampleAll(:, :, selectedAntenna, :); % 提取第4个天线数据


% ----------------------- Range-Speed 图 -----------------------
% 进行 Range FFT
rangeFFT = fft(antennaData, nFFT, 1);  % 在 range 维度上做 FFT
rangeFFT = rangeFFT(1:nFFT/2, :, :, :); % 取正频部分

% 进行 Doppler FFT（删除了 fftshift）
dopplerFFT = fft(rangeFFT, 16, 2); % 在 chirp 维度上做 FFT
dopplerFFT = dopplerFFT(:, :, :, :);  
combinedDoppler = abs(dopplerFFT);

% Min-Max 归一化
minVal = min(combinedDoppler(:));
maxVal = max(combinedDoppler(:));
normalizedDoppler = (combinedDoppler - minVal) / (maxVal - minVal);

% 可视化 Range-Speed 图（第一帧为例）
% 重新计算 Speed Axis (速度轴)
% % 计算 speedAxis (速度轴)
% Adjust the speed axis to ensure 0 speed is at the middle
speedAxis = (-nsFFT/2:nsFFT/2-1) / nsFFT * c / (2 * p.startFreq * 1e9 * p.nChirp * p.chirpTime * 1e-6);
maxSpeed = speedAxis(end);
minSpeed = speedAxis(1);

% disp(['最大测速范围: ', num2str(minSpeed), ' m/s 到 ', num2str(maxSpeed), ' m/s']);
% Shift the axis to place 0 at the middle
% speedAxis = fftshift(speedAxis);
% 最大测速范围

rangeAxis = (0:nFFT/2-1) * p.rangeRes;    % 仅取正频部分的距离轴
disp(['rangeAxis ', num2str(rangeAxis(end)), ' m']);  % 显示range

figure;
% imagesc(rangeAxis, speedAxis, squeeze(normalizedDoppler(:, :, 1, 1))); % 显示第一帧
% xlabel('Range (m)');
% ylabel('Speed (m/s)');
% title('Range-Speed (Normalized)');
% colorbar;

% --------------动画展示-----------------------------------------------------------
% 可视化 Range-Speed 图
for frame = 1:size(normalizedDoppler, 4)
    % 提取当前帧的数据
%     dataToPlot = squeeze(normalizedDoppler(:, 1:nFFT/2, 1, frame)); % 取前半部分数据
    dataToPlot = squeeze(normalizedDoppler(:, :, 1, frame)); % 取前半部分数据
    dataToPlot = fftshift(dataToPlot,2);
    
    % 对数据进行 Min-Max 归一化
    minVal = min(dataToPlot(:));
    maxVal = max(dataToPlot(:));
    normalizedData = (dataToPlot - minVal) / (maxVal - minVal);

    % 绘制 Range-Speed 图
    imagesc(speedAxis, rangeAxis, normalizedData);  % 使用已计算的 rangeAxis 和 speedAxis
    xlabel('Speed (m/s)');
    ylabel('Range (m)');
    title(['Frame ', num2str(frame)]);
    colorbar;  % 显示色条
    pause(0.1); % 暂停 0.1 秒，形成动画效果
end


% ----------------------- Range-Time 图 -----------------------
% 初始化存储 Range-Time 数据
nFrames = size(adcSampleAll, 4); % 总帧数
rangeTimeData = zeros(nFFT/2, nFrames); % 用于存储 Range-Time 数据

% 逐帧处理
for frame = 1:nFrames
    % 提取当前帧数据
    frameData = antennaData(:, :, frame); % [Range, Chirp]
    % 进行 Range FFT
    rangeFFT = fft(frameData, nFFT, 1); % Range FFT
    rangeFFT = abs(rangeFFT(1:nFFT/2, :)); % 取正频部分
    % 对 Chirp 求平均值
    rangeTimeData(:, frame) = mean(rangeFFT, 2);
end

% 可视化 Range-Time 图
% 计算时间轴
% 可视化 Range-Time 图
% 计算每帧的时间
timeAxis = (0:nFrames-1) * p.chirpTime * p.nChirp * 1e-6;  % 计算每帧的时间

disp(['timeAxis ', num2str(timeAxis(end)), ' s']);  % 显示最后一个时间点

% 可视化 Range-Time 图
figure;
imagesc(timeAxis, rangeAxis, rangeTimeData); % 绘制 Range-Time 图
xlabel('Time (s)');  % 横轴标签：时间 (秒)
ylabel('Range (m)');  % 纵轴标签：距离 (米)
title('Range-Time Map');

% 设置横轴的范围，确保横轴只显示到最后一个帧的数据
xlim([0 timeAxis(end)]);  % 横轴范围从0到最后一个时间点

% 设置横轴刻度，使得每帧的时间分界点更加明显
xticks(timeAxis(1:5:end));  % 每隔5个帧显示一个刻度
xticklabels(arrayfun(@(x) sprintf('%.2f', x), timeAxis(1:5:end), 'UniformOutput', false));  % 格式化标签为小数点后两位

colorbar;  % 显示颜色条

