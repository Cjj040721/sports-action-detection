%���˶������range-speed map 
% ��������
clear;
clc;

% ----------------------- �״�������� -----------------------
c = 3e8;  % ���� (m/s)

% RF ����
p.startFreq = 60;       % GHz
p.chirpTime = 360;      % us
p.sweepSlope = 3000 / 57; % MHz/us
p.sweepTime = 57;       % us
p.sweepBandwidth = p.sweepSlope * p.sweepTime; % MHz

% ���߲���
p.nTxAnt = 1;
p.nRxAnt = 4;
p.nAnt = p.nTxAnt * p.nRxAnt;

% ADC ����
p.sampleRate = 10;      % MHz
p.nSample = 512;        % ÿ�� chirp �Ĳ�������
p.nChirp = 16;          % ÿ֡ chirp ��
p.sampleTime = p.nSample / p.sampleRate; % ÿ�� chirp ��ʱ�� (us)
p.sampleBandwidth = p.sampleTime * p.sweepSlope; % ���� (MHz)
p.rangeRes = c / (2 * p.sampleBandwidth * 1e6); % ����ֱ��� (m)
p.dopRes = c / (2 * p.startFreq * 1e9 * p.nChirp * p.chirpTime * 1e-6 * p.nTxAnt); % �����շֱ��� (m/s)

% ----------------------- ���ݼ��غʹ��� -----------------------
% ��������
% data = load('adcSampleAll_run.mat');
data = load('adcSampleAll.mat');
adcSampleAll = data.adcSampleAll;  % �����״�����

% �������� [Range, Chirp, Antenna, Frame]
adcSampleAll = reshape(adcSampleAll, p.nSample, p.nChirp, p.nRxAnt, []);

% FFT ����
nFFT = 512;  % FFT ����
nsFFT = 16;  % chirp����
% ѡ���ض��������ݣ������4�����ߣ�
selectedAntenna = 4; % ��4������
antennaData = adcSampleAll(:, :, selectedAntenna, :); % ��ȡ��4����������


% ----------------------- Range-Speed ͼ -----------------------
% ���� Range FFT
rangeFFT = fft(antennaData, nFFT, 1);  % �� range ά������ FFT
rangeFFT = rangeFFT(1:nFFT/2, :, :, :); % ȡ��Ƶ����

% ���� Doppler FFT��ɾ���� fftshift��
dopplerFFT = fft(rangeFFT, 16, 2); % �� chirp ά������ FFT
dopplerFFT = dopplerFFT(:, :, :, :);  
combinedDoppler = abs(dopplerFFT);

% Min-Max ��һ��
minVal = min(combinedDoppler(:));
maxVal = max(combinedDoppler(:));
normalizedDoppler = (combinedDoppler - minVal) / (maxVal - minVal);

% ���ӻ� Range-Speed ͼ����һ֡Ϊ����
% ���¼��� Speed Axis (�ٶ���)
% % ���� speedAxis (�ٶ���)
% Adjust the speed axis to ensure 0 speed is at the middle
speedAxis = (-nsFFT/2:nsFFT/2-1) / nsFFT * c / (2 * p.startFreq * 1e9 * p.nChirp * p.chirpTime * 1e-6);
maxSpeed = speedAxis(end);
minSpeed = speedAxis(1);

% disp(['�����ٷ�Χ: ', num2str(minSpeed), ' m/s �� ', num2str(maxSpeed), ' m/s']);
% Shift the axis to place 0 at the middle
% speedAxis = fftshift(speedAxis);
% �����ٷ�Χ

rangeAxis = (0:nFFT/2-1) * p.rangeRes;    % ��ȡ��Ƶ���ֵľ�����
disp(['rangeAxis ', num2str(rangeAxis(end)), ' m']);  % ��ʾrange

figure;
% imagesc(rangeAxis, speedAxis, squeeze(normalizedDoppler(:, :, 1, 1))); % ��ʾ��һ֡
% xlabel('Range (m)');
% ylabel('Speed (m/s)');
% title('Range-Speed (Normalized)');
% colorbar;

% --------------����չʾ-----------------------------------------------------------
% ���ӻ� Range-Speed ͼ
for frame = 1:size(normalizedDoppler, 4)
    % ��ȡ��ǰ֡������
%     dataToPlot = squeeze(normalizedDoppler(:, 1:nFFT/2, 1, frame)); % ȡǰ�벿������
    dataToPlot = squeeze(normalizedDoppler(:, :, 1, frame)); % ȡǰ�벿������
    dataToPlot = fftshift(dataToPlot,2);
    
    % �����ݽ��� Min-Max ��һ��
    minVal = min(dataToPlot(:));
    maxVal = max(dataToPlot(:));
    normalizedData = (dataToPlot - minVal) / (maxVal - minVal);

    % ���� Range-Speed ͼ
    imagesc(speedAxis, rangeAxis, normalizedData);  % ʹ���Ѽ���� rangeAxis �� speedAxis
    xlabel('Speed (m/s)');
    ylabel('Range (m)');
    title(['Frame ', num2str(frame)]);
    colorbar;  % ��ʾɫ��
    pause(0.1); % ��ͣ 0.1 �룬�γɶ���Ч��
end


% ----------------------- Range-Time ͼ -----------------------
% ��ʼ���洢 Range-Time ����
nFrames = size(adcSampleAll, 4); % ��֡��
rangeTimeData = zeros(nFFT/2, nFrames); % ���ڴ洢 Range-Time ����

% ��֡����
for frame = 1:nFrames
    % ��ȡ��ǰ֡����
    frameData = antennaData(:, :, frame); % [Range, Chirp]
    % ���� Range FFT
    rangeFFT = fft(frameData, nFFT, 1); % Range FFT
    rangeFFT = abs(rangeFFT(1:nFFT/2, :)); % ȡ��Ƶ����
    % �� Chirp ��ƽ��ֵ
    rangeTimeData(:, frame) = mean(rangeFFT, 2);
end

% ���ӻ� Range-Time ͼ
% ����ʱ����
% ���ӻ� Range-Time ͼ
% ����ÿ֡��ʱ��
timeAxis = (0:nFrames-1) * p.chirpTime * p.nChirp * 1e-6;  % ����ÿ֡��ʱ��

disp(['timeAxis ', num2str(timeAxis(end)), ' s']);  % ��ʾ���һ��ʱ���

% ���ӻ� Range-Time ͼ
figure;
imagesc(timeAxis, rangeAxis, rangeTimeData); % ���� Range-Time ͼ
xlabel('Time (s)');  % �����ǩ��ʱ�� (��)
ylabel('Range (m)');  % �����ǩ������ (��)
title('Range-Time Map');

% ���ú���ķ�Χ��ȷ������ֻ��ʾ�����һ��֡������
xlim([0 timeAxis(end)]);  % ���᷶Χ��0�����һ��ʱ���

% ���ú���̶ȣ�ʹ��ÿ֡��ʱ��ֽ���������
xticks(timeAxis(1:5:end));  % ÿ��5��֡��ʾһ���̶�
xticklabels(arrayfun(@(x) sprintf('%.2f', x), timeAxis(1:5:end), 'UniformOutput', false));  % ��ʽ����ǩΪС�������λ

colorbar;  % ��ʾ��ɫ��

