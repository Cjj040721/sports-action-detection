function [bytesBuffer, bytesBufferLen, isBufferFull, bytesAvailableFlag] = readUARTtoBuffer(spDataHandle, bytesBuffer, bytesBufferLen)

BYTES_BUFFER_MAX_SIZE = 2^22;
isBufferFull = 0;
bytesAvailableFlag = 0;

%get num bytes at input buffer
numBytesToRead = get(spDataHandle,'BytesAvailable');

%append new bytes to bytes buffer
if numBytesToRead 
    indx_start_append = bytesBufferLen+1;
    if(numBytesToRead > BYTES_BUFFER_MAX_SIZE - bytesBufferLen)
        % more bytes to read than space in buffer
        numBytesToRead = BYTES_BUFFER_MAX_SIZE - bytesBufferLen;
    end
    %read in bytes at serial input buffer  
    [newBytes, byteCount] = fread(spDataHandle, numBytesToRead, 'uint8');
    
    %append to buffer
    bytesBuffer(indx_start_append:indx_start_append+byteCount-1) = newBytes;
    bytesBufferLen = bytesBufferLen + byteCount;
    bytesAvailableFlag = 1;
    
end

return