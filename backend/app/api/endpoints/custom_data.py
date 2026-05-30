import io
import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data"))
CUSTOM_DATA_PATH = os.path.join(DATA_DIR, "user_contributed_prices.csv")
EXPECTED_COLUMNS = ["area_type", "availability", "location", "size", "society", "total_sqft", "bath", "balcony", "price"]


class CustomPropertyInput(BaseModel):
    area_type: str = Field(default="Super built-up  Area")
    availability: str = Field(default="Ready To Move")
    location: str = Field(...)
    size: str = Field(..., description="e.g. 2 BHK, 3 Bedroom")
    society: str | None = Field(default="")
    total_sqft: float = Field(..., gt=0)
    bath: float = Field(..., ge=1)
    balcony: float = Field(default=1.0, ge=0)
    price: float = Field(..., gt=0, description="Price in Lakhs")


@router.get("/custom-data")
def get_custom_data():
    if not os.path.exists(CUSTOM_DATA_PATH):
        return []
    try:
        df = pd.read_csv(CUSTOM_DATA_PATH)
        df = df.fillna("")
        records = df.to_dict(orient="records")
        # Assign a stable ID based on index
        for i, rec in enumerate(records):
            rec["id"] = i
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom-data/add")
def add_custom_property(item: CustomPropertyInput):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        new_row = pd.DataFrame([{
            "area_type": item.area_type,
            "availability": item.availability,
            "location": item.location,
            "size": item.size,
            "society": item.society or "",
            "total_sqft": item.total_sqft,
            "bath": item.bath,
            "balcony": item.balcony,
            "price": item.price
        }])
        
        if os.path.exists(CUSTOM_DATA_PATH):
            df = pd.read_csv(CUSTOM_DATA_PATH)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
            
        df.to_csv(CUSTOM_DATA_PATH, index=False)
        return {"status": "success", "message": "Property added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom-data/upload")
async def upload_custom_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        contents = await file.read()
        df_uploaded = pd.read_csv(io.BytesIO(contents))
        
        # Verify core required columns for training
        required = ["location", "total_sqft", "bath", "price"]
        missing = [col for col in required if col not in df_uploaded.columns]
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Uploaded CSV is missing required columns: {', '.join(missing)}"
            )
            
        # Add missing optional columns with defaults
        if "area_type" not in df_uploaded.columns:
            df_uploaded["area_type"] = "Super built-up  Area"
        if "availability" not in df_uploaded.columns:
            df_uploaded["availability"] = "Ready To Move"
        if "size" not in df_uploaded.columns:
            if "bhk" in df_uploaded.columns:
                df_uploaded["size"] = df_uploaded["bhk"].apply(lambda x: f"{int(x)} BHK" if pd.notnull(x) else "2 BHK")
            else:
                df_uploaded["size"] = "2 BHK"
        if "society" not in df_uploaded.columns:
            df_uploaded["society"] = ""
        if "balcony" not in df_uploaded.columns:
            df_uploaded["balcony"] = 1.0
            
        # Subset and align with EXPECTED_COLUMNS
        df_standard = df_uploaded[EXPECTED_COLUMNS].copy()
        
        # Clean null values in critical fields
        df_standard = df_standard.dropna(subset=["location", "total_sqft", "bath", "price"])
        
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(CUSTOM_DATA_PATH):
            df_existing = pd.read_csv(CUSTOM_DATA_PATH)
            df_combined = pd.concat([df_existing, df_standard], ignore_index=True)
        else:
            df_combined = df_standard
            
        df_combined.to_csv(CUSTOM_DATA_PATH, index=False)
        return {"status": "success", "message": f"Successfully imported {len(df_standard)} properties"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/custom-data/{index}")
def delete_custom_property(index: int):
    if not os.path.exists(CUSTOM_DATA_PATH):
        raise HTTPException(status_code=404, detail="No custom data file found")
    try:
        df = pd.read_csv(CUSTOM_DATA_PATH)
        if index < 0 or index >= len(df):
            raise HTTPException(status_code=400, detail="Invalid index")
        df = df.drop(index).reset_index(drop=True)
        if len(df) == 0:
            if os.path.exists(CUSTOM_DATA_PATH):
                os.remove(CUSTOM_DATA_PATH)
        else:
            df.to_csv(CUSTOM_DATA_PATH, index=False)
        return {"status": "success", "message": f"Deleted listing {index}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom-data/clear")
def clear_custom_data():
    if os.path.exists(CUSTOM_DATA_PATH):
        try:
            os.remove(CUSTOM_DATA_PATH)
            return {"status": "success", "message": "Custom data cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "message": "No custom data to clear"}


@router.get("/custom-data/stats")
def get_custom_data_stats():
    original_path = os.path.join(DATA_DIR, "bengaluru_house_prices.csv")
    original_count = 0
    if os.path.exists(original_path):
        try:
            df = pd.read_csv(original_path)
            original_count = len(df)
        except:
            pass
            
    custom_count = 0
    if os.path.exists(CUSTOM_DATA_PATH):
        try:
            df_custom = pd.read_csv(CUSTOM_DATA_PATH)
            custom_count = len(df_custom)
        except:
            pass
            
    return {
        "original_dataset_size": original_count,
        "custom_dataset_size": custom_count,
        "total_dataset_size": original_count + custom_count
    }
