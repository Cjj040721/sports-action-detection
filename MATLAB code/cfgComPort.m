% ���ܣ��򿪻�رմ���
% ���������
% comPort   - ���ڲ���������comNum�����ںţ���baudRate�������ʣ�
% comState  - ����״̬��0���رգ�1���򿪣�
% �������
% �޸�ʱ�䣺2022.07.04

function [hComPort] = cfgComPort(comPort, comState)
    % �������ںŵ��ַ�����ʾ������ COM1, COM2
    comportnum_str = ['COM' num2str(comPort.comNum)];
    
    % ɾ���Ѿ����ӵĴ��ڶ���������ڣ�
    connectedPorts = instrfind('Type','serial');  % �������еĴ�������
    findFlag = 0;  % ��ʼ����־�������ж��Ƿ��ҵ���ɾ�����еĴ�������
    
    % �����ǰ�д�������
    if(~isempty(connectedPorts))
        numPorts = size(connectedPorts, 2);  % ��ȡ��ǰ�������ӵ�����
        
        % ���ֻ��һ���������ӣ�����Ƿ��ǵ�ǰ����
        if(numPorts == 1)
            if(strcmp(connectedPorts.Name, ['Serial-' comportnum_str]))
                % �����������ƥ�䣬��ɾ�����Ӳ����Ϊ�Ѵ���
                delete(connectedPorts)
                findFlag = 1;
            end
        else
            % ����ж���������ӣ������ÿ������
            indx = 1;
            while(indx <= size(connectedPorts, 2))
                if(strcmp(connectedPorts.Name(indx), ['Serial-' comportnum_str]))
                    % ����ҵ�ƥ��Ĵ��ڣ�ɾ���ô������Ӳ����Ϊ�Ѵ���
                    delete(connectedPorts(indx))
                    findFlag = 1;
                    connectedPorts = instrfind('Type','serial');  % �������ӵĴ����б�
                else
                    indx = indx + 1;
                end
            end
        end
    end
    
    % ������ڹرգ�comState == 0��
    if comState == 0
        % ����������Ӵ��ڲ���ɾ����������ڹرյ���Ϣ
        if findFlag == 1
            fprintf([comportnum_str ' closed. \n']);
        else
            % ���û���ҵ��������ӣ������δ���ӵ���Ϣ
            fprintf([comportnum_str ' not connected. \n']);
        end
        % ���ؿգ���ʾû�д򿪴���
        hComPort = [];
    else
        % �����Ҫ�򿪴��ڣ�comState == 1��
        % �������ڶ������ò�����
        hComPort = serial(comportnum_str, 'BaudRate', comPort.baudRate);
        
        try
            % ���ô��ڵ����뻺������СΪ 2^16 �ֽ�
            set(hComPort, 'InputBufferSize', 2^16);
            % ���ô��ڳ�ʱʱ��Ϊ 2 ��
            set(hComPort, 'Timeout', 2); 
            % �򿪴���
            fopen(hComPort);
            % ������ڴ򿪳ɹ�����Ϣ
            fprintf([comportnum_str ' opened. \n']);
        catch ME
            % ����򿪴���ʧ�ܣ����������Ϣ
            fprintf(['Error: ' comportnum_str ' could not be opened! \n']);
            % ɾ�������Ĵ��ڶ���
            delete(hComPort);
            % �����ھ������Ϊ -1����ʾʧ��
            hComPort = -1;
        end
    end
    
    return
