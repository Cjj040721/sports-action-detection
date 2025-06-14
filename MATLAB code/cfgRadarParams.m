
c = 3e8;

%% RF 参数
p.startFreq = 60;       % GHz 起始频率为 60 GHz。
p.chirpTime = 360;       % us 每个 chirp 的持续时间
p.sweepSlope = 3000/57;	% MHz/us 频率变化率（频率增量与时间的比值）
p.sweepTime = 57;       % us 每个 chirp 的时间
p.sweepBandwidth = p.sweepSlope * p.sweepTime; %扫描带宽

%% 天线参数
p.nTxAnt = 1;
p.nRxAnt = 4;
p.nAnt = p.nTxAnt * p.nRxAnt;

%% adc参数及分辨率
p.sampleRate = 10;   % MHz 采样率
p.nSample = 512;     % 每个 chirp 的采样点数
p.nChirp = 16;       % 每次测量的 chirp 数量
p.sampleTime = p.nSample / p.sampleRate;    % 每个 chirp 的采样时间
p.sampleBandwidth = p.sampleTime * p.sweepSlope;    % 每个 chirp 的采样带宽
p.rangeRes = c / (2 * p.sampleBandwidth * 1e6);     % 距离分辨率
p.dopRes = c / (2 * p.startFreq * 1e9 * p.nChirp * p.chirpTime * 1e-6 * p.nTxAnt);  %多普勒分辨率

%% 信号处理参数
p.rangeFftLen = p.nSample;  % 距离FFT的长度
p.dopFftLen = p.nChirp * p.nTxAnt;  % 多普勒FFT的长度
p.aziFftLen = 128;  % 方位FFT的长度
p.eleFftLen = 128;  % 仰角FFT的长度

p.nRangeBin = p.rangeFftLen / 2;  % 距离分辨率bin数
p.nDopBin = p.nChirp;  % 多普勒分辨率bin数
p.nAziBin = p.aziFftLen;  % 方位bin数
p.nEleBin = p.eleFftLen;  % 仰角bin数

p.dpkTime = 1;  % 峰值时间
p.dpkTres = 6;  % 峰值门限

p.cfarAnt = 1;  % CFAR (Constant False Alarm Rate) 方法参数

p.cfarCfg.dim1Thr = 0;  % CFAR 阈值设置
p.cfarCfg.dim1GuardSize = 0;  % CFAR 护卫区大小
p.cfarCfg.dim1SearchSize = 1;  % CFAR 搜索区大小
p.cfarCfg.dim1A = 1;  % CFAR 参数
p.cfarCfg.dim1B = p.nRangeBin;  % CFAR 参数

p.cfarCfg.dim2Thr = 1;  % CFAR 阈值设置
p.cfarCfg.dim2GuardSize = 0;  % CFAR 护卫区大小
p.cfarCfg.dim2SearchSize = 1;  % CFAR 搜索区大小
p.cfarCfg.dim2A = 1;  % CFAR 参数
p.cfarCfg.dim2B = p.nDopBin;  % CFAR 参数


%% 导入常数
load const.mat
p.const = const;
