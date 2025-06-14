function varargout = mainSetup(varargin)
% MAINSETUP MATLAB code for mainSetup.fig
%      MAINSETUP, by itself, creates a new MAINSETUP or raises the existing
%      singleton*.
%
%      H = MAINSETUP returns the handle to a new MAINSETUP or the handle to
%      the existing singleton*.
%
%      MAINSETUP('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MAINSETUP.M with the given input arguments.
%
%      MAINSETUP('Property','Value',...) creates a new MAINSETUP or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before mainSetup_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to mainSetup_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help mainSetup

% Last Modified by GUIDE v2.5 04-Jul-2022 19:31:40

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @mainSetup_OpeningFcn, ...
                   'gui_OutputFcn',  @mainSetup_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before mainSetup is made visible.
function mainSetup_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to mainSetup (see VARARGIN)

% Choose default command line output for mainSetup
handles.output = hObject;

if exist('state.mat')
    load state.mat;
    handles.comPort.comNum = comPort.comNum;
    handles.comPort.baudRate = comPort.baudRate;
else
    handles.comPort.comNum = [];
    handles.comPort.baudRate = [];
end

%% 设置端口号
% 搜索存在的端口
command = 'wmic path win32_pnpentity get caption /format:list | find "COM"';
[~, cmdout] = system (command);
startIdx = strfind(cmdout, 'Caption');                % 定位搜索结果
startIdx(1,size(startIdx,2)+1) = size(cmdout,2) + 1;  % 附上末尾位置
for i = 1 : size(startIdx,2) - 1
    temp = cmdout(startIdx(i)+8 : startIdx(i+1)-3);
    tempIdx = strfind(temp, '(COM');
    cellPort(i+1,1) = cellstr([temp(tempIdx+1:end-1) ' ' temp(1:tempIdx-2)]);
    if handles.comPort.comNum == str2num(temp(tempIdx+4:end-1))
        cellPort(1,1) = cellPort(i+1,1);
    end
end

selectComNum = findobj('Tag','selectComNum');
set(selectComNum,'String',cellPort);

%% 设置波特率
baudrate = [9600 115200 460800 750000 921600 2000000 3000000];
for i = 1 : size(baudrate,2)
    cellBaudrate(i+1,1) = cellstr(num2str(baudrate(i)));
end
cellBaudrate(1,1) = cellstr(num2str(handles.comPort.baudRate));

selectComBaudrate = findobj('Tag','selectComBaudrate');
set(selectComBaudrate,'String',cellBaudrate);

%%
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes mainSetup wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = mainSetup_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in start.
function start_Callback(hObject, eventdata, handles)
% hObject    handle to start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

fprintf(handles.hSerialPort,'AT+START');
main(handles);

fprintf('stop\r\n');

% --- Executes on button press in stop.
function stop_Callback(hObject, eventdata, handles)
% hObject    handle to stop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% --- Executes on button press in openComPort.
global RUN_STATE;
RUN_STATE = 0;


function openComPort_Callback(hObject, eventdata, handles)
% hObject    handle to openComPort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.comPort = readComPort;
handles.hSerialPort = cfgComPort(handles.comPort,1);
%Update Handles
guidata(hObject, handles);


% --- Executes on button press in closeComPort.
function closeComPort_Callback(hObject, eventdata, handles)
% hObject    handle to closeComPort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.comPort = readComPort;
handles.hSerialPort = cfgComPort(handles.comPort,0);
%Update Handles
guidata(hObject, handles);


function [comPort] = readComPort

selectComNum = findobj('Tag','selectComNum');
contents = get(selectComNum,'String');
strComNum = contents{get(selectComNum,'Value')};
spaceIdx = strfind(strComNum, ' ');
comPort.comNum = str2num(strComNum(4:spaceIdx-1));

selectComBaudrate = findobj('Tag','selectComBaudrate');
contents = get(selectComBaudrate,'String');
strBaudrate = contents{get(selectComBaudrate,'Value')};
comPort.baudRate = str2num(strBaudrate);


% --- Executes on selection change in selectComNum.
function selectComNum_Callback(hObject, eventdata, handles)
% hObject    handle to selectComNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectComNum contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectComNum

contents = get(hObject,'String');
strComNum = contents{get(hObject,'Value')};
spaceIdx = strfind(strComNum, ' ');
handles.comPort.comNum = str2num(strComNum(4:spaceIdx-1));
% Update handles structure
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function selectComNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectComNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in selectComBaudrate.
function selectComBaudrate_Callback(hObject, eventdata, handles)
% hObject    handle to selectComBaudrate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectComBaudrate contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectComBaudrate

contents = get(hObject,'String');
strBaudrate = contents{get(hObject,'Value')};
handles.comPort.baudRate = str2num(strBaudrate);
% Update handles structure
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function selectComBaudrate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectComBaudrate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

comPort = handles.comPort;
save('state.mat','comPort');

cfgComPort(comPort,0);

% Hint: delete(hObject) closes the figure
delete(hObject);
