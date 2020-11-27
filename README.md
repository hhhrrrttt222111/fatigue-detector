# FATIGUE-DETECTOR


## Clone the repo
```
git clone https://github.com/hhhrrrttt222111/fatigue-detector.git
cd fatigue-detector
```

## Installation
#### For windows
```
py -m pip install --upgrade pip
py -m pip install --user virtualenv
py -m venv env
.\env\Scripts\activate
py -m pip install -r requirements.txt
```
*OR*
```
py -m pip install --upgrade pip
py -m pip install --user virtualenv
py -m venv env
source env/Scripts/activate
py -m pip install -r requirements.txt
```
#### For MacOS & Linux
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Trouble installing dependencies ?

This project files requires **Python 3** and the following Python libraries installed:

- [OpenCV](https://opencv.org/)
- [dlib](https://github.com/davisking/dlib)
- [imutils](https://github.com/jrosebr1/imutils)
- [flask](https://flask.palletsprojects.com/en/1.1.x/)
- [scipy](https://www.scipy.org/)


[OpenCV](https://github.com/opencv/opencv) - [Mac](https://www.learnopencv.com/install-opencv3-on-macos/) | [Windows](https://www.learnopencv.com/install-opencv3-on-windows/) | [Ubuntu](https://www.learnopencv.com/install-opencv3-on-ubuntu/)


[Dlib](https://github.com/davisking/dlib) -   [Mac](https://www.learnopencv.com/install-dlib-on-macos/) | [Windows](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f) | [Ubuntu](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)


### Run

```
python app.py
```  