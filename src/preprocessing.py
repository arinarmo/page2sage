import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

TARGET_COL = "LogSalePrice"
if __name__ == "__main__":
    house_data = pd.read_csv("/opt/ml/processing/input/house_prices.csv")
    logging.info(f"Initial shape: {house_data.shape}")
    
    columns = [
        "SalePrice",
        "MSZoning",
        "LotFrontage",
        "LotArea",
        "GrLivArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YearRemodAdd",
        "RoofStyle",
        "FullBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageType",
        "HouseStyle"
    ]
    cat_cols = ["MSZoning", "RoofStyle", "GarageType", "HouseStyle"]
    num_cols = [col for col in columns if not col in cat_cols]
    
    house_data = house_data[columns]
    logging.info(f"Shape after select: {house_data.shape}")
    
    logging.info("Transforming positive columns to log")
    positive_cols = ["SalePrice", "LotFrontage", "LotArea", "GrLivArea"]

    for col in positive_cols:
        house_data["Log"+col] = np.log10(house_data[col])

    logging.info("Imputing missing data")
    house_data.loc[pd.isna(house_data.LotFrontage), "LotFrontage"] = 0
    house_data.loc[pd.isna(house_data.LogLotFrontage), "LogLotFrontage"]  = 0

    house_data.loc[pd.isna(house_data.GarageType), "GarageType"] = "None"
    
    logging.info("Encoding categorical variables")
    enc = OneHotEncoder()
    house_data_cat = pd.DataFrame(
        enc.fit_transform(house_data[cat_cols]).todense(), 
        columns=enc.get_feature_names(house_data[cat_cols].columns)
    )

    logging.info(f"Categorical data shape: {house_data_cat.shape}")
    
    house_data_num = house_data[num_cols].drop(positive_cols, axis=1)
    target = house_data[TARGET_COL]
    
    house_data_enc = pd.concat([house_data_num, house_data_cat], axis=1)
    logging.info(f"Final data shape: {house_data_enc.shape}") 
    
    train_X, val_X, train_y, val_y = train_test_split(house_data_enc, target, random_state=42)
    
    pd.concat([train_y, train_X], axis=1).to_csv("/opt/ml/processing/train/train.csv", index=False, header=False)
    pd.concat([val_y, val_X], axis=1).to_csv("/opt/ml/processing/validation/validation.csv", index=False, header=False)
    logging.info(f"Wrote {train_X.shape[0]} training rows and {val_X.shape[0]} validation rows")
    