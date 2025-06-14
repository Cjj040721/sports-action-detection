% ���� : ������UART��ӡ��HEX��ʽ����
% ���������
%   filePath     - UART�����ļ�·��
%   variableType - �������ͣ�int8_t; int16_t; uint32_t; float; pfloat��
%   dataType     - �������ͣ�Real; ImRe; ReIm; ReImPf��
%   varargin     - �������ݽ���Ϊ�����ά��
% ��� : vectorData - ���ս����ĸ�ʽ�����ݡ���������ʽ
%        matrixData - ���ս����ĸ�ʽ�����ݡ���������ʽ
% �޸�ʱ�䣺2023.03.07

function [vectorData, matrixData] = parseProcessData(filePath, variableType, dataType, varargin)

%% ����ԭʼ����
cellData = textread(filePath, '%s');
uint8Data = hex2dec(char(cellData));

%% ���ݱ������ͽ���ת��
switch variableType
    case 'uint8_t'
        parseData = uint8Data;
    case 'int8_t'
        parseData = typecast(uint8(uint8Data), 'int8');
    case 'uint16_t'
        parseData = typecast(uint8(uint8Data), 'uint16');
    case 'int16_t'
        parseData = typecast(uint8(uint8Data), 'int16');
    case 'uint32_t'
        parseData = typecast(uint8(uint8Data), 'uint32');
    case 'float'
        parseData = typecast(uint8(uint8Data), 'single');
    case 'pfloat'
        parseData = typecast(uint8(uint8Data), 'uint32');
end

parseData = double(parseData);

%% �����������ͽ���ת��
switch dataType
    case 'Real'
        vectorData = parseData;
    case 'ImRe'
        imagData = parseData(1:2:end);
        realData = parseData(2:2:end);
        vectorData = realData + sqrt(-1)*imagData;
    case 'ReIm'
        realData = parseData(1:2:end);
        imagData = parseData(2:2:end);
        vectorData = realData + sqrt(-1)*imagData;
    case 'ReImPf'
        vectorData = pseudoFloatCplx2FloatCplx(parseData);
end

%% ת��Ϊ�����ʽ
switch length(varargin)
    case 0
        matrixData = vectorData;
    case 1
        matrixData = reshape(vectorData, varargin{1}, length(vectorData)/varargin{1});
    case 2
        matrixData = reshape(vectorData, varargin{1}, varargin{2}, length(vectorData)/(varargin{1}*varargin{2}));
    case 3
        matrixData = reshape(vectorData, varargin{1}, varargin{2}, varargin{3}, length(vectorData)/(varargin{1}*varargin{2}*varargin{3}));
    otherwise
        disp('�����������');
end

%%
% ���� : ��32λα���㸴��ת��Ϊ���㸴��
% ���������
%   pfDataIn     - �����α���㸴��
% ���:
%   fDataOut     - ת����ĸ��㸴��
% �޸�ʱ�䣺2022.06.02
function [fDataOut] = pseudoFloatCplx2FloatCplx(pfDataIn)

idx = bitshift(pfDataIn,-28);

realPart = bitand(pfDataIn,hex2dec('3FFF'));
negIdx = find(realPart >= 2^13);
realPart(negIdx) = realPart(negIdx) - 2^14;

imagPart = bitand(bitshift(pfDataIn,-14),hex2dec('3FFF'));
negIdx = find(imagPart >= 2^13);
imagPart(negIdx) = imagPart(negIdx) - 2^14;

fDataOut = (realPart + sqrt(-1)*imagPart) .* 2.^(idx - 13);
