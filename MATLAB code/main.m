
function main(handles)

global RUN_STATE;
RUN_STATE = 1;

cfgRadarParams;
% 每帧数据的期望大小，取决于雷达的采样数量、天线数量和每个信号的脉冲数。
desiredSize = p.nSample * p.nAnt * p.nChirp; 
% 存储从串口读取的字节数据，大小为串口输入缓冲区的大小。
bytesBuffer = zeros(1,handles.hSerialPort.InputBufferSize);
bytesBufferLen = 0;

allFrame = [];
frameCnt = 1;
% 启动计时器，用于测量每帧处理时间。
tic

recDataType = 0;

while RUN_STATE
    fig = 1;
    % 读取数据
    [bytesBuffer, bytesBufferLen, isBufferFull, bytesAvailableFlag] = readUARTtoBuffer(handles.hSerialPort, bytesBuffer, bytesBufferLen);
    % parse bytes to frame
    [newframe, bytesBuffer, bytesBufferLen, numFramesAvailable,validFrame] = parseBytes_TM(bytesBuffer, bytesBufferLen, 'FIFO');
    
    if numFramesAvailable > 0
        switch recDataType
            case 0
                % 获取数据帧中的有效数据部分，跳过前 12 字节的头部信息
                if size(newframe.packet(13:end)) ~= desiredSize * 2
                    fprintf('errSize = %d - %d\r\n',size(newframe.packet(13:end)),desiredSize);
                    continue;
                end
                % 提取uint8类型并进行转换int16类型
                dataBuffer = typecast(uint8(newframe.packet(13:end)),'int16');
                % reshape 函数将一维的 dataBuffer 转换成三维矩阵，大小为 p.nSample x p.nAnt x p.nChirp。
                % adcSample 是一个三维数组，表示雷达的原始采样数据。
                adcSample = reshape(dataBuffer,p.nSample,p.nAnt,p.nChirp); 
                % 重新排列矩阵的维度，修改之后为[采样点数, 脉冲数，天线数]
                adcSample = permute(adcSample,[1 3 2]);
                % 将当前帧的处理结果 adcSample 存储到 adcSampleAll 中。
                % adcSampleAll 是一个四维数组，用于存储所有帧的数据，frameCnt 是当前帧的计数。
                % 通过 frameCnt，程序将每一帧的采样数据存储在 adcSampleAll 的不同位置。
                adcSampleAll(:,:,:,frameCnt) = adcSample;
                figure(fig); fig = fig + 1; plot(adcSample(:,:,1));
        end
        
        if frameCnt == 10
            save adcSampleAll.mat adcSampleAll
            frameCnt = 0;
        end
        toc
        tic
        frameCnt = frameCnt + 1;
    end
    pause(0.01);
end