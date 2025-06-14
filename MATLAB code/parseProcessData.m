% 功能 : 解析由UART打印的HEX格式数据
% 输入参数：
%   filePath     - UART保存文件路径
%   variableType - 变量类型（int8_t; int16_t; uint32_t; float; pfloat）
%   dataType     - 数据类型（Real; ImRe; ReIm; ReImPf）
%   varargin     - 定义数据解析为矩阵的维度
% 输出 : vectorData - 最终解析的格式化数据——向量形式
%        matrixData - 最终解析的格式化数据——矩阵形式
% 修改时间：2023.03.07

function [vectorData, matrixData] = parseProcessData(filePath, variableType, dataType, varargin)

%% 读入原始数据
cellData = textread(filePath, '%s');
uint8Data = hex2dec(char(cellData));

%% 根据变量类型进行转换
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

%% 根据数据类型进行转换
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

%% 转换为矩阵格式
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
        disp('输入参数错误！');
end

%%
% 功能 : 将32位伪浮点复数转化为浮点复数
% 输入参数：
%   pfDataIn     - 输入的伪浮点复数
% 输出:
%   fDataOut     - 转化后的浮点复数
% 修改时间：2022.06.02
function [fDataOut] = pseudoFloatCplx2FloatCplx(pfDataIn)

idx = bitshift(pfDataIn,-28);

realPart = bitand(pfDataIn,hex2dec('3FFF'));
negIdx = find(realPart >= 2^13);
realPart(negIdx) = realPart(negIdx) - 2^14;

imagPart = bitand(bitshift(pfDataIn,-14),hex2dec('3FFF'));
negIdx = find(imagPart >= 2^13);
imagPart(negIdx) = imagPart(negIdx) - 2^14;

fDataOut = (realPart + sqrt(-1)*imagPart) .* 2.^(idx - 13);
