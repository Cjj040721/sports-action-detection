clear ;
clc
close all

addpath('校准系数');

commonCfg.numSamp =256;
commonCfg.numHead =0;
commonCfg.numChirp=16;
commonCfg.numAnt  =16;
% commonCfg.numAnt  =8;
dlen = (commonCfg.numSamp+commonCfg.numHead) * commonCfg.numChirp * commonCfg.numAnt;


%Path='F:\北交大工作\北交大2022\北航雷达\4T4R使用说明-20210820\4T4R使用说明-20210820\UartTool_V1p000\Matlab\Data\';
Path='D:\paper\FMCW\data\orignal_test\Out_walking';
File=dir(fullfile(Path,'*.bin'));
FileNames ={File.name}';
NumberFile=length(File);


for Num=1:NumberFile
    %for Num=1:1
    %for Num=2:2
    K_trace= strcat(Path,FileNames(Num,1));
    K_trace_new=string(K_trace);
    fid = fopen(K_trace_new, 'r'); % sampling data
    [FrameDataBuff] = fread(fid);
    fclose(fid);
    
    for i=1:length(FrameDataBuff)
        if(FrameDataBuff(i)>128)
            FrameDataBuff(i) = FrameDataBuff(i)-256;
        end
    end
    frameTotal = floor(length(FrameDataBuff)/(dlen+8))-1; %截位取整
    % raw data
    rawData = zeros(commonCfg.numSamp, commonCfg.numChirp, commonCfg.numAnt, frameTotal);
    for frameNow=1:frameTotal
        RecDataBuff=FrameDataBuff((dlen+8)*(frameNow-1)+1+8:(dlen+8)*frameNow);
        for antNow=1:commonCfg.numAnt
            for chirpNow=1:commonCfg.numChirp
                rawData(:,chirpNow,antNow,frameNow) = RecDataBuff((commonCfg.numSamp+commonCfg.numHead)*(commonCfg.numChirp*(antNow-1)+(chirpNow-1))+1+commonCfg.numHead:...
                    (commonCfg.numSamp+commonCfg.numHead)*(commonCfg.numChirp*(antNow-1)+    chirpNow));
            end
        end
    end
    
    [nSample, nChirp, nAnt, nFrame] = size(rawData);
    %{
%     figure;
%     for iFrame = 1 : frameTotal
%         hold off;
%         for iAnt = 1 : commonCfg.numAnt
%             subplot(commonCfg.numAnt,1,iAnt);
%             plot(rawData(:,:,iAnt,iFrame)); hold on;
%         end
%         pause(0.01);
%     end
    %}
    %% data calibration
    k_calib = 2.585e9/85e-6;% FM slope, calibration parameter, can not change
    fs_calib = 3.6363e6;% sampling rate, calibration parameter, no need to change
    
    k_use = 2.585e9/85e-6;% FM slope, sample parameter, change according to actual setting
    fs_use = 3.6363e6;%sampling rate, sample parameter, change according to actual setting
    
    Bandwidth = 2.585e9;% FM slope, calibration parameter, can not change
    fs_calib = 3.6363e6;% sampling rate, calibration parameter, no need to change
    tc=85e-6;
    C=299792458;
    %distance=C*tc/(2*Bandwidth)*FF;
    
    Range_resolution=0.070060; %m
    velocity_resolution=0.312500; %m/s
    range_thresh=12.8;
    Frame_time=40e-3; %秒
    
    Total_time_x_axis=Frame_time/ nChirp: Frame_time/ nChirp :Frame_time* nFrame;
    Total_range_y_axis= Range_resolution: Range_resolution: Range_resolution*127;
    
    load amp_phase_calib.mat;
    % load phase_calib.mat;
    load freq_calib.mat;
    
    for i = 1:nFrame
        data = rawData(:,:,:,i);
        %     data_pre = adcSample(:,:,:,i-1);
        
        for ant_i = 1 : nAnt
            for chirp_i = 1: nChirp
                data_calib(:,chirp_i,ant_i,i) = amp_phase_calib(ant_i,1)*data(:,chirp_i,ant_i)...
                    .*exp(1i*2*pi*(freq_calib(ant_i,1)*(k_use*fs_calib/(fs_use*k_calib)))*[0:1:nSample-1]'/nSample);
            end
        end
    end
    
    %%
    [nSample, nChirp, nAnt, nFrame] = size(data_calib);
    
    %做个cube
    Range_Time_Space_cube_origin=zeros(nSample/2+1,nChirp*nFrame,nAnt);
    Range_Time_Space_cube=zeros(nSample/2+1,nChirp*nFrame,nAnt);
    for antenna_number=1:nAnt
        
        for i=1:nFrame
            Data_Antenna_1(:,:,i)=data_calib(:,:,antenna_number,i);
        end
        
        AA_range_map_reshape_2=reshape(Data_Antenna_1,[] ,1);
        
        Fs=3.6363e6;
        NFFT=256;
        Window_size=NFFT;
        overlap=0;
        [S,FF,T,P]=   spectrogram(AA_range_map_reshape_2, Window_size, overlap,NFFT,Fs,'MinThreshold',-70,'yaxis' ); %将低于-20的归零
       % figure,imagesc(Total_time_x_axis,Total_range_y_axis,(10*log10(abs(P)))  );
        %%%一列一列地减，这个方法也可以把固定的物品的回波消掉
        %     for i=1:nFrame*nChirp-1
        %         AA_dopp_time_reshape_no_fix_object(:,i)=P(:,i+1)-P(:,i);
        %     end
        %     range_time_power=10*log10(abs(AA_dopp_time_reshape_no_fix_object));
        %{
    %%%%Range-time-map
    for i=1:nFrame
        Frame_data=Data_Antenna_1(:,:,i);
        sig_fft = fft(Frame_data, nSample);
        sig_fft_RangeMap = abs(sig_fft ./ nSample);
        sig_fft_RangeMap_final(:,:,i)= sig_fft_RangeMap(1:nSample/2,:);
    end
    %figure,imagesc( 10*log10( sig_fft_RangeMap_final) );
  
    %plot(sig_fft_RangeMap_final(:,:,3));

    AA_range_map_reshape=reshape(sig_fft_RangeMap_final,nSample/2 ,nFrame*nChirp);
figure,imagesc(Total_time_x_axis,Total_range_y_axis,(10*log10(abs(AA_range_map_reshape)))  );
        
    %每一列做normalization
    for i=1:nFrame*nChirp
        AA_range_map_reshape_normalized(:,i)=AA_range_map_reshape(:,i)./max(AA_range_map_reshape(:,i));
    end
    figure,imagesc(Total_time_x_axis,Total_range_y_axis,(10*log10(abs(AA_range_map_reshape_normalized)))  );

    
      %%%一列一列地减，这个方法也可以把固定的物品的回波消掉
    for i=1:nFrame*nChirp-1
        AA_dopp_time_reshape_no_fix_object(:,i)=AA_range_map_reshape_normalized(:,i+1)-AA_range_map_reshape_normalized(:,i);
    end

    range_time_power=10*log10(abs(AA_dopp_time_reshape_no_fix_object));
    %figure,imagesc(Total_time_x_axis,Total_range_y_axis, range_time_power );

    for i=1:nSample/2
        for j=1:nFrame*nChirp-1
            if range_time_power(i,j)<-17
                range_time_power(i,j)=-40;
            end
        end
    end
        %}
        %figure,imagesc(Total_time_x_axis,Total_range_y_axis, range_time_power );
        
        
        
        %有的时候最简单的加窗都要更好用。
        
        %%butterworth filter
        fc_low=1e6;
        fc=1.6e6;
        [Buter_low,Buter_up]=butter(10,[fc_low fc]/(fs_use/2));
        %freqz(Buter_low,Buter_up)
        dataOut = filter(Buter_low,Buter_up,AA_range_map_reshape_2);
        [S,FF,T,P_filter]=   spectrogram(AA_range_map_reshape_2, Window_size, overlap,NFFT,Fs,'MinThreshold',-70,'yaxis' ); %将低于-20的归零
        %figure,imagesc(Total_time_x_axis,Total_range_y_axis,(10*log10(abs(P_filter)))  );
        %      title('Range-Time Map')
        %      xlabel('Time (s)')
        %      ylabel('Range (m)')
        [sizea sizeb]=size(P);
        [sizec sized]=size(P_filter);
        
        if sizea==Window_size
            range_time_cube_data_ori= P(1:nSample/2+1,:);
        else
            range_time_cube_data_ori=P;
        end
        if sizec==Window_size
            Range_Time_Space_cube_processed= P_filter(1:nSample/2+1,:) ;
        else
            Range_Time_Space_cube_processed=P_filter;
        end
        
        Range_Time_Space_cube_origin(:,:,antenna_number)=(10*log10(abs(range_time_cube_data_ori)));
        Range_Time_Space_cube(:,:,antenna_number)=(10*log10(abs(Range_Time_Space_cube_processed)));
        
        FileNames_save_1=strrep(FileNames,'.bin','_time_range_space_origi.xlsx');
        FileNames_save_2=strrep(FileNames,'.bin','_time_range_space_proce.xlsx');
        
        K_trace_save= strcat(Path,FileNames_save_1(Num,1));
        K_trace_save_str=string(K_trace_save);
        sheet_num= string(strcat('Sheet',string(antenna_number)));
        xlswrite(K_trace_save_str, Range_Time_Space_cube_origin(:,:,antenna_number), sheet_num ) ;
        
        K_trace_save_2= strcat(Path,FileNames_save_2(Num,1));
        K_trace_save_str_2=string(K_trace_save_2);
        sheet_num_2= string(strcat('Sheet',string(antenna_number)));
        xlswrite(K_trace_save_str_2, Range_Time_Space_cube(:,:,antenna_number), sheet_num_2 ) ;
        
        
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%range-doppler-map-correct
    %fft( fft(Frame_data, nSample).', nChirp);
    %     for i=1:nFrame
    %         Frame_data=Data_Antenna_1(:,:,i);
    %         sig_fft = fft(Frame_data, nSample);
    %         sig_fft_dopp= fft(sig_fft.', nChirp);
    %         sig_fft_Range_dopp_Map = fftshift( sig_fft_dopp(2:nChirp/2,:) );
    %         sig_fft_Range_dopp_Map_final(:,:,i)= (abs(sig_fft_Range_dopp_Map));
    %     end
    %
    
    
    
    data_range_doppler_map_reshape=reshape(Range_Time_Space_cube_processed,129 ,16,[]);
    
    for i=1:nFrame
        Frame_data=data_range_doppler_map_reshape(:,:,i);
        sig_fft_dopp= fft(Frame_data.', nChirp);
        sig_fft_Range_dopp_Map = fftshift( sig_fft_dopp(1:nChirp/2,1:64) );
        sig_fft_Range_dopp_Map_final(:,:,i)= (abs(sig_fft_Range_dopp_Map));
        
        
        FileNames_save_3=strrep(FileNames,'.bin','_range_speed_time_proce.xlsx');
        K_trace_save= strcat(Path,FileNames_save_3(Num,1));
        K_trace_save_str=string(K_trace_save);
        sheet_num= string(strcat('Sheet',string(i)));
        xlswrite(K_trace_save_str, 10*log10(sig_fft_Range_dopp_Map_final(:,:,i)) , sheet_num ) ;
   
        
    end
    
    doppler_axis = linspace(-2.5,2.5,nChirp);
    range_axis = Range_resolution*nSample/2: -Range_resolution: 0;
    %figure,imagesc(range_axis, doppler_axis,(10*log10(abs(sig_fft_Range_dopp_Map_final(:,:,15)))) );
    
    sig_fft_Range_dopp_Map_final_save=10*log10(sig_fft_Range_dopp_Map_final) ;
    
        
    

          
    
    
    
    
    
    
    %{
     for i=1:nFrame
        Frame_data=Data_Antenna_1(:,:,i);
        sig_fft_dopp = fft2(Frame_data, nSample, nChirp);
        sig_fft_Range_dopp_Map = fftshift( sig_fft_dopp(1:nSample/2,1:nChirp) );
        %sig_fft_Range_dopp_Map_final(:,:,i)= 10*log10(abs(sig_fft_Range_dopp_Map));
        sig_fft_Range_dopp_Map_final(:,:,i)= sig_fft_Range_dopp_Map;
    end
    %}
    
    
    
    %AA_Range_dopp_map_reshape=zeros(nSample/2,nChirp);
    %for i=1:nFrame
    %    AA_Range_dopp_map_reshape=AA_Range_dopp_map_reshape+(sig_fft_Range_dopp_Map_final(:,:,i));
    %end
    %figure,imagesc(range_axis,doppler_axis, (10*log10(abs(AA_Range_dopp_map_reshape)))'  );
    %     title('Range-Doppler Map')
    %     ylabel('Speed (m/s)')
    %     xlabel('Range (m)')
    %
    
    
    
    %%%%%%range-doppler-map
    
    %{
    %%This is not correct. Range-doppler is not reshape, it should be sum.
    for i=1:nFrame
        Frame_data=Data_Antenna_1(:,:,i);
        sig_fft_dopp = fft2(Frame_data, nSample, nChirp);
        sig_fft_Range_dopp_Map = fftshift( sig_fft_dopp(1:nSample/2,1:nChirp) );
        sig_fft_Range_dopp_Map_final(:,:,i)= 10*log10(abs(sig_fft_Range_dopp_Map));
    end
    
  
doppler_axis = linspace(-5,5,nChirp);
range_axis = 0: Range_resolution: Range_resolution*127;
%figure,surf(doppler_axis,range_axis,sig_fft_Range_dopp_Map_final(:,:,1));
imagesc(doppler_axis,range_axis,sig_fft_Range_dopp_Map_final(:,:,1) );

    %AA_range_dopp_map_reshape=reshape(sig_fft_Range_dopp_Map_final,128,nFrame*16);
    %AA_range_dopp_map_reshape=reshape(sig_fft_Range_dopp_Map_final,nFrame*128,16);%shape一定不对，是对应点相加
    sig_fft_Range_dopp_Map_final_use=zeros(nSample/2,nChirp);
    for i=1:nFrame
        sig_fft_Range_dopp_Map_final_use=sig_fft_Range_dopp_Map_final_use+abs(sig_fft_Range_dopp_Map_final(:,:,i));
    end
    
    figure,imagesc(doppler_axis,range_axis, (10*log10(abs(sig_fft_Range_dopp_Map_final_use)))  );
    title('Range-Doppler Map')
    xlabel('Speed (m/s)')
    ylabel('Range (m)')
    %}
    
    
    
    
    
    
    
    
    
end






