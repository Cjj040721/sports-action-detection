
function [dataOut] = parseSkyData(dataIn,sign,W,D)

dataInTmp = bitand(dataIn,(2^W - 1));

switch sign
    case 'S'
        negIdx = find(dataInTmp >= 2^(W - 1));
        dataInTmp(negIdx) = dataInTmp(negIdx) - 2^W;
end

dataOut = dataInTmp ./ 2^D;