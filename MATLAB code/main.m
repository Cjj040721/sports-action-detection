
function main(handles)

global RUN_STATE;
RUN_STATE = 1;

cfgRadarParams;
% ÿ֡���ݵ�������С��ȡ�����״�Ĳ�������������������ÿ���źŵ���������
desiredSize = p.nSample * p.nAnt * p.nChirp; 
% �洢�Ӵ��ڶ�ȡ���ֽ����ݣ���СΪ�������뻺�����Ĵ�С��
bytesBuffer = zeros(1,handles.hSerialPort.InputBufferSize);
bytesBufferLen = 0;

allFrame = [];
frameCnt = 1;
% ������ʱ�������ڲ���ÿ֡����ʱ�䡣
tic

recDataType = 0;

while RUN_STATE
    fig = 1;
    % ��ȡ����
    [bytesBuffer, bytesBufferLen, isBufferFull, bytesAvailableFlag] = readUARTtoBuffer(handles.hSerialPort, bytesBuffer, bytesBufferLen);
    % parse bytes to frame
    [newframe, bytesBuffer, bytesBufferLen, numFramesAvailable,validFrame] = parseBytes_TM(bytesBuffer, bytesBufferLen, 'FIFO');
    
    if numFramesAvailable > 0
        switch recDataType
            case 0
                % ��ȡ����֡�е���Ч���ݲ��֣�����ǰ 12 �ֽڵ�ͷ����Ϣ
                if size(newframe.packet(13:end)) ~= desiredSize * 2
                    fprintf('errSize = %d - %d\r\n',size(newframe.packet(13:end)),desiredSize);
                    continue;
                end
                % ��ȡuint8���Ͳ�����ת��int16����
                dataBuffer = typecast(uint8(newframe.packet(13:end)),'int16');
                % reshape ������һά�� dataBuffer ת������ά���󣬴�СΪ p.nSample x p.nAnt x p.nChirp��
                % adcSample ��һ����ά���飬��ʾ�״��ԭʼ�������ݡ�
                adcSample = reshape(dataBuffer,p.nSample,p.nAnt,p.nChirp); 
                % �������о����ά�ȣ��޸�֮��Ϊ[��������, ��������������]
                adcSample = permute(adcSample,[1 3 2]);
                % ����ǰ֡�Ĵ����� adcSample �洢�� adcSampleAll �С�
                % adcSampleAll ��һ����ά���飬���ڴ洢����֡�����ݣ�frameCnt �ǵ�ǰ֡�ļ�����
                % ͨ�� frameCnt������ÿһ֡�Ĳ������ݴ洢�� adcSampleAll �Ĳ�ͬλ�á�
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