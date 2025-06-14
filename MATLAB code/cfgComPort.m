% 功能：打开或关闭串口
% 输入参数：
% comPort   - 串口参数，包含comNum（串口号）和baudRate（波特率）
% comState  - 配置状态（0：关闭；1：打开）
% 输出：无
% 修改时间：2022.07.04

function [hComPort] = cfgComPort(comPort, comState)
    % 创建串口号的字符串表示，例如 COM1, COM2
    comportnum_str = ['COM' num2str(comPort.comNum)];
    
    % 删除已经连接的串口对象（如果存在）
    connectedPorts = instrfind('Type','serial');  % 查找所有的串口连接
    findFlag = 0;  % 初始化标志变量，判断是否找到并删除已有的串口连接
    
    % 如果当前有串口连接
    if(~isempty(connectedPorts))
        numPorts = size(connectedPorts, 2);  % 获取当前串口连接的数量
        
        % 如果只有一个串口连接，检查是否是当前串口
        if(numPorts == 1)
            if(strcmp(connectedPorts.Name, ['Serial-' comportnum_str]))
                % 如果串口名称匹配，则删除连接并标记为已处理
                delete(connectedPorts)
                findFlag = 1;
            end
        else
            % 如果有多个串口连接，则遍历每个串口
            indx = 1;
            while(indx <= size(connectedPorts, 2))
                if(strcmp(connectedPorts.Name(indx), ['Serial-' comportnum_str]))
                    % 如果找到匹配的串口，删除该串口连接并标记为已处理
                    delete(connectedPorts(indx))
                    findFlag = 1;
                    connectedPorts = instrfind('Type','serial');  % 更新连接的串口列表
                else
                    indx = indx + 1;
                end
            end
        end
    end
    
    % 如果串口关闭（comState == 0）
    if comState == 0
        % 如果串口连接存在并已删除，输出串口关闭的信息
        if findFlag == 1
            fprintf([comportnum_str ' closed. \n']);
        else
            % 如果没有找到串口连接，则输出未连接的信息
            fprintf([comportnum_str ' not connected. \n']);
        end
        % 返回空，表示没有打开串口
        hComPort = [];
    else
        % 如果需要打开串口（comState == 1）
        % 创建串口对象并设置波特率
        hComPort = serial(comportnum_str, 'BaudRate', comPort.baudRate);
        
        try
            % 设置串口的输入缓冲区大小为 2^16 字节
            set(hComPort, 'InputBufferSize', 2^16);
            % 设置串口超时时间为 2 秒
            set(hComPort, 'Timeout', 2); 
            % 打开串口
            fopen(hComPort);
            % 输出串口打开成功的信息
            fprintf([comportnum_str ' opened. \n']);
        catch ME
            % 如果打开串口失败，输出错误信息
            fprintf(['Error: ' comportnum_str ' could not be opened! \n']);
            % 删除创建的串口对象
            delete(hComPort);
            % 将串口句柄设置为 -1，表示失败
            hComPort = -1;
        end
    end
    
    return
