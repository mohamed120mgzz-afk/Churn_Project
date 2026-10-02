from .CustomerData import CustomerData

import pandas as pd
def predict_new(data: CustomerData,preproceesor,model):

    #to data frame
    df=pd.DataFrame([data.model_dump()])
    #transform
    X_processed=preproceesor.transform(df)

    #predict
    y_pred=model.predict(X_processed)
    y_prop=model.predict_proba(X_processed)

    return {
        "churn_prediction":bool(y_pred[0]),
        "churn_Probability":float(y_prop[0][1])
    }