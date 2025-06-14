
c = 3e8;

%% RF ����
p.startFreq = 60;       % GHz ��ʼƵ��Ϊ 60 GHz��
p.chirpTime = 360;       % us ÿ�� chirp �ĳ���ʱ��
p.sweepSlope = 3000/57;	% MHz/us Ƶ�ʱ仯�ʣ�Ƶ��������ʱ��ı�ֵ��
p.sweepTime = 57;       % us ÿ�� chirp ��ʱ��
p.sweepBandwidth = p.sweepSlope * p.sweepTime; %ɨ�����

%% ���߲���
p.nTxAnt = 1;
p.nRxAnt = 4;
p.nAnt = p.nTxAnt * p.nRxAnt;

%% adc�������ֱ���
p.sampleRate = 10;   % MHz ������
p.nSample = 512;     % ÿ�� chirp �Ĳ�������
p.nChirp = 16;       % ÿ�β����� chirp ����
p.sampleTime = p.nSample / p.sampleRate;    % ÿ�� chirp �Ĳ���ʱ��
p.sampleBandwidth = p.sampleTime * p.sweepSlope;    % ÿ�� chirp �Ĳ�������
p.rangeRes = c / (2 * p.sampleBandwidth * 1e6);     % ����ֱ���
p.dopRes = c / (2 * p.startFreq * 1e9 * p.nChirp * p.chirpTime * 1e-6 * p.nTxAnt);  %�����շֱ���

%% �źŴ������
p.rangeFftLen = p.nSample;  % ����FFT�ĳ���
p.dopFftLen = p.nChirp * p.nTxAnt;  % ������FFT�ĳ���
p.aziFftLen = 128;  % ��λFFT�ĳ���
p.eleFftLen = 128;  % ����FFT�ĳ���

p.nRangeBin = p.rangeFftLen / 2;  % ����ֱ���bin��
p.nDopBin = p.nChirp;  % �����շֱ���bin��
p.nAziBin = p.aziFftLen;  % ��λbin��
p.nEleBin = p.eleFftLen;  % ����bin��

p.dpkTime = 1;  % ��ֵʱ��
p.dpkTres = 6;  % ��ֵ����

p.cfarAnt = 1;  % CFAR (Constant False Alarm Rate) ��������

p.cfarCfg.dim1Thr = 0;  % CFAR ��ֵ����
p.cfarCfg.dim1GuardSize = 0;  % CFAR ��������С
p.cfarCfg.dim1SearchSize = 1;  % CFAR ��������С
p.cfarCfg.dim1A = 1;  % CFAR ����
p.cfarCfg.dim1B = p.nRangeBin;  % CFAR ����

p.cfarCfg.dim2Thr = 1;  % CFAR ��ֵ����
p.cfarCfg.dim2GuardSize = 0;  % CFAR ��������С
p.cfarCfg.dim2SearchSize = 1;  % CFAR ��������С
p.cfarCfg.dim2A = 1;  % CFAR ����
p.cfarCfg.dim2B = p.nDopBin;  % CFAR ����


%% ���볣��
load const.mat
p.const = const;
