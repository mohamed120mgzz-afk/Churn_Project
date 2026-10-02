from dotenv import load_dotenv
import os
import joblib

load_dotenv(override=True)


#.env Variables
APP_NAME= os.getenv('APP_NAME')
VERSION=os.getenv('VERSION')
SECRET_KEY_TOKEN=os.getenv('SECRET_KEY_TOKEN')




BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER_PATH=os.path.join(BASE_DIR,'models')


#models
preprocessor=joblib.load(os.path.join(MODELS_FOLDER_PATH,'preprocessor.joblib'))
forest_model=joblib.load(os.path.join(MODELS_FOLDER_PATH,'forest_tuned.pkl'))
xg_boost_model=joblib.load(os.path.join(MODELS_FOLDER_PATH,'xgb_tuned.pkl'))