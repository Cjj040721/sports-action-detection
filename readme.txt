1. Run the system backend
    (1) Open the system backend folder with pycharm
    (2) conda create -n envName python=3.11
    (3) conda activate envName
    (4) pip install numpy scipy matplotlib seaborn opencv-python torch torchvision flask flask-cors scikit-learn

2. Run the matlab code
    (1) Connect the radar to the computer.
    (2) Open the matlab code folder and run mainSetup.m file -- open serial port → start.

3. Run the system front end
    (1) Install node.js 2022
    (2) cd to the system front end folder in the command line (use administrator)
    (3) Input "npm run dev" in the command line
    (4) Enter http://localhost:8000/ in the browser


【Note】: To avoid path error issues --
1. Please ensure that the /data folder is in the system_backend and the matlab code, system backend and system frontend are in the same folder.
2. Please ensure that you do not open the "Code -- Intelligent Sports Action Detection System" with pycharm or matlab. Instead, open system backend folder with pycharm and matlab code folder with matlab.
3. The collected dataset is in the system backend folder.
