# textmining

REST Api Get for text sentiment analyzer.

Python 3
Flask 
Sklearn
zappa

To start:

eb init

pip install virtualenv

source your_virtual_environment_name/bin/activate

pip install -r requirements.txt

pip install zappa

zappa init

zappa deploy your-environment-name

Project Structure:
- .ebtextensions : configs for run project
- api
    - application.py : main api class
- templates : pages templates for front api
