from fastapi import FastAPI,HTTPException,Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from utils.inferance import predict_new
from utils.CustomerData import CustomerData

from utils.config import APP_NAME,VERSION,SECRET_KEY_TOKEN,preprocessor,forest_model,xg_boost_model
app=FastAPI(title=APP_NAME,version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/',tags=["Genral"])
async def home():
    return {
        'message':f'welcome to my {APP_NAME} Api vs {VERSION}'
    }

api_key_header=APIKeyHeader(name='X-Api_Key')
async def verify_api_key(api_key:str=Depends(api_key_header)):
    if api_key!=SECRET_KEY_TOKEN:
        raise HTTPException(status_code=403,detail="You are not authorized to use this api")

    return api_key



@app.post('/predict/forest',tags=["Models"])
async def predict_forest(data: CustomerData,api_key:str=Depends(verify_api_key)) -> dict:
    try:
        result=predict_new(data=data,preproceesor=preprocessor,model=forest_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
    

@app.post('/predict/XGBoost',tags=["Models"])
async def predict_XGBoost(data: CustomerData,api_key:str=Depends(verify_api_key)) -> dict:
    try:
        result=predict_new(data=data,preproceesor=preprocessor,model=xg_boost_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    