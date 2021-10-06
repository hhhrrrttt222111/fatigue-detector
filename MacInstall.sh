cd ~/Desktop
git clone https://github.com/hhhrrrttt222111/fatigue-detector.git
cd fatigue-detector
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
