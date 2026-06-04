import os
import json
import logging
import httpx
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

# Local Project Imports
from ml_project.preprocessing import load_and_prepare_training_frame
from app.config import get_settings
from app.schemas.house import HouseInput
from app.api.endpoints.predict import _predict_sync, _explain_sync

# Setup Logger
logger = logging.getLogger(__name__)
router = APIRouter()


# =====================================================================
# 1. DATASET IN-MEMORY LOAD
# =====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "bengaluru_house_prices.csv")
CUSTOM_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "user_contributed_prices.csv")

df_data = None
try:
    if os.path.exists(CSV_PATH):
        df_data = load_and_prepare_training_frame(
            CSV_PATH, 
            custom_csv_path=CUSTOM_CSV_PATH if os.path.exists(CUSTOM_CSV_PATH) else None,
            apply_training_filters=False
        )
        logger.info("Successfully loaded %d records for AI agent queries", len(df_data))
    else:
        logger.warning("Dataset not found at %s. AI Agent CSV search will be unavailable.", CSV_PATH)
except Exception as e:
    logger.error("Failed to load in-memory dataset for agent: %s", e)


# =====================================================================
# 2. CORE AGENT TOOLS (PYTHON IMPLEMENTATION)
# =====================================================================
def search_listings(
    location: Optional[str] = None, 
    bhk: Optional[int] = None, 
    budget_min_lakhs: Optional[float] = None, 
    budget_max_lakhs: Optional[float] = None, 
    limit: int = 5
) -> dict:
    """
    Queries the in-memory Pandas dataframe for historical listings matching filters.
    Computes overall matching statistics and extracts a limited set of samples.
    """
    if df_data is None:
        return {"error": "Historical dataset is currently unavailable."}
        
    filtered = df_data.copy()
    
    if location:
        loc_clean = location.strip().lower()
        filtered = filtered[filtered["location"].str.lower().str.contains(loc_clean, na=False)]
        
    if bhk:
        filtered = filtered[filtered["bhk"] == bhk]
        
    if budget_min_lakhs:
        filtered = filtered[filtered["price"] >= budget_min_lakhs]
        
    if budget_max_lakhs:
        filtered = filtered[filtered["price"] <= budget_max_lakhs]
        
    count = len(filtered)
    if count == 0:
        return {
            "message": "No matching listings found in the dataset for these criteria.",
            "total_matches": 0,
            "sample_listings": []
        }
        
    avg_price = filtered["price"].mean()
    median_price = filtered["price"].median()
    avg_sqft = filtered["total_sqft"].mean()
    
    if "price_per_sqft" in filtered.columns:
        avg_pps = filtered["price_per_sqft"].mean()
    else:
        avg_pps = (filtered["price"] * 100000 / filtered["total_sqft"]).mean()
        
    sample = filtered.head(limit).to_dict(orient="records")
    simplified_listings = []
    for s in sample:
        simplified_listings.append({
            "location": s.get("location"),
            "total_sqft": int(s.get("total_sqft", 0)),
            "bhk": int(s.get("bhk", 0)),
            "bath": int(s.get("bath", 0)),
            "balcony": int(s.get("balcony", 0)),
            "price_lakhs": round(s.get("price", 0), 2),
            "ready_to_move": bool(s.get("is_ready_to_move", True))
        })
        
    return {
        "total_matches_in_dataset": count,
        "average_price_lakhs": round(avg_price, 2),
        "median_price_lakhs": round(median_price, 2),
        "average_sqft": round(avg_sqft, 1),
        "average_price_per_sqft": round(avg_pps, 1),
        "sample_listings": simplified_listings
    }


def compare_locations(location_a: str, location_b: str) -> dict:
    """
    Compares statistical metrics of two locations side-by-side.
    Returns matched names, listing counts, average price, average sqft, and price/sqft.
    """
    if df_data is None:
        return {"error": "Historical dataset is currently unavailable."}
        
    loc_a_clean = location_a.strip().lower()
    loc_b_clean = location_b.strip().lower()
    
    df_a = df_data[df_data["location"].str.lower() == loc_a_clean]
    df_b = df_data[df_data["location"].str.lower() == loc_b_clean]
    
    # Fallback to substring matching
    if len(df_a) == 0:
        df_a = df_data[df_data["location"].str.lower().str.contains(loc_a_clean, na=False)]
    if len(df_b) == 0:
        df_b = df_data[df_data["location"].str.lower().str.contains(loc_b_clean, na=False)]
        
    res = {}
    for key, subdf, original_query in [("location_a", df_a, location_a), ("location_b", df_b, location_b)]:
        if len(subdf) == 0:
            res[key] = {"query": original_query, "error": f"No listings found in the dataset for '{original_query}'."}
            continue
            
        avg_pps = subdf["price_per_sqft"].mean() if "price_per_sqft" in subdf.columns else (subdf["price"] * 100000 / subdf["total_sqft"]).mean()
        res[key] = {
            "query": original_query,
            "matched_name": subdf["location"].iloc[0],
            "total_listings": len(subdf),
            "average_price_lakhs": round(subdf["price"].mean(), 2),
            "median_price_lakhs": round(subdf["price"].median(), 2),
            "average_sqft": round(subdf["total_sqft"].mean(), 1),
            "average_price_per_sqft": round(avg_pps, 1)
        }
        
    return res


def get_ml_prediction(
    location: str,
    total_sqft: float,
    bhk: int,
    bath: int,
    balcony: int,
    area_type: Optional[str] = None,
    is_ready_to_move: int = 1
) -> dict:
    """
    Executes the trained machine learning model ensemble to get a house price estimate.
    Standardizes input area types and handles HTTPException validations gracefully.
    """
    try:
        area_mapping = {
            "super built-up": "Super built-up  Area",
            "super built-up area": "Super built-up  Area",
            "built-up": "Built-up  Area",
            "built-up area": "Built-up  Area",
            "plot": "Plot  Area",
            "plot area": "Plot  Area",
            "carpet": "Carpet  Area",
            "carpet area": "Carpet  Area"
        }
        mapped_area = area_type
        if area_type and area_type.lower().strip() in area_mapping:
            mapped_area = area_mapping[area_type.lower().strip()]

        input_data = HouseInput(
            location=location.strip(),
            total_sqft=float(total_sqft),
            bhk=int(bhk),
            bath=int(bath),
            balcony=int(balcony),
            area_type=mapped_area,
            is_ready_to_move=int(is_ready_to_move)
        )
        
        pred_res = _predict_sync(input_data)
        return {
            "predicted_price_lakhs": round(pred_res.predicted_price_lakhs, 2),
            "predicted_price_crores": round(pred_res.predicted_price_crores, 4),
            "model_consensus": {k: round(v, 2) for k, v in pred_res.model_consensus.items()},
            "consensus_method": pred_res.consensus_method,
            "spread_pct": pred_res.spread_pct
        }
    except HTTPException as he:
        return {"error": he.detail}
    except Exception as e:
        return {"error": str(e)}


def explain_ml_prediction(
    location: str,
    total_sqft: float,
    bhk: int,
    bath: int,
    balcony: int,
    area_type: Optional[str] = None,
    is_ready_to_move: int = 1
) -> dict:
    """
    Generates a SHAP explanation to show the pricing impact of features in Lakhs.
    Standardizes input area types and handles HTTPException validations gracefully.
    """
    try:
        area_mapping = {
            "super built-up": "Super built-up  Area",
            "super built-up area": "Super built-up  Area",
            "built-up": "Built-up  Area",
            "built-up area": "Built-up  Area",
            "plot": "Plot  Area",
            "plot area": "Plot  Area",
            "carpet": "Carpet  Area",
            "carpet area": "Carpet  Area"
        }
        mapped_area = area_type
        if area_type and area_type.lower().strip() in area_mapping:
            mapped_area = area_mapping[area_type.lower().strip()]

        input_data = HouseInput(
            location=location.strip(),
            total_sqft=float(total_sqft),
            bhk=int(bhk),
            bath=int(bath),
            balcony=int(balcony),
            area_type=mapped_area,
            is_ready_to_move=int(is_ready_to_move)
        )
        
        explain_res = _explain_sync(input_data)
        
        contribs = []
        for c in explain_res.contributions:
            contribs.append({
                "feature": c.display_name,
                "value": c.raw_value,
                "price_impact_lakhs": round(c.contribution_lakhs, 2)
            })
            
        return {
            "base_value_lakhs": round(explain_res.base_value_lakhs, 2),
            "predicted_price_lakhs": round(explain_res.predicted_price_lakhs, 2),
            "contributions": contribs
        }
    except HTTPException as he:
        return {"error": he.detail}
    except Exception as e:
        return {"error": str(e)}


# =====================================================================
# 3. OLLAMA TOOL SCHEMAS
# =====================================================================
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": "Search the historical house dataset for listings in Bangalore matching certain criteria, returning statistics and samples. Use this for queries about average house sizes, average pricing, or past sale details in specific neighborhoods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Name of the location or neighborhood in Bangalore (e.g. Indira Nagar, Whitefield)."
                    },
                    "bhk": {
                        "type": "integer",
                        "description": "Number of bedrooms (BHK)."
                    },
                    "budget_min_lakhs": {
                        "type": "number",
                        "description": "Minimum budget boundary in Lakhs (e.g. 50)."
                    },
                    "budget_max_lakhs": {
                        "type": "number",
                        "description": "Maximum budget boundary in Lakhs (e.g. 150)."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of listings to return (default 5)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_locations",
            "description": "Compares historical listings, pricing averages, and sizes of two neighborhoods side-by-side (e.g. comparing Indira Nagar vs Electronic City).",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_a": {
                        "type": "string",
                        "description": "First location name."
                    },
                    "location_b": {
                        "type": "string",
                        "description": "Second location name."
                    }
                },
                "required": ["location_a", "location_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ml_prediction",
            "description": "Call the machine learning models to calculate a price estimate for a house with exact specifications. Use this whenever the user wants to estimate, evaluate, or predict the price of a specific house.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name in Bangalore (e.g. Indira Nagar)."
                    },
                    "total_sqft": {
                        "type": "number",
                        "description": "Total area in square feet."
                    },
                    "bhk": {
                        "type": "integer",
                        "description": "Number of bedrooms."
                    },
                    "bath": {
                        "type": "integer",
                        "description": "Number of bathrooms."
                    },
                    "balcony": {
                        "type": "integer",
                        "description": "Number of balconies."
                    },
                    "area_type": {
                        "type": "string",
                        "enum": ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"],
                        "description": "Type of area layout."
                    },
                    "is_ready_to_move": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "1 if ready to move, 0 if under construction."
                    }
                },
                "required": ["location", "total_sqft", "bhk", "bath", "balcony"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_ml_prediction",
            "description": "Runs SHAP values to explain the specific pricing factors for a house. Shows how much features like size, bedrooms, or location increased or decreased the predicted value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name in Bangalore."
                    },
                    "total_sqft": {
                        "type": "number",
                        "description": "Total area in square feet."
                    },
                    "bhk": {
                        "type": "integer",
                        "description": "Number of bedrooms."
                    },
                    "bath": {
                        "type": "integer",
                        "description": "Number of bathrooms."
                    },
                    "balcony": {
                        "type": "integer",
                        "description": "Number of balconies."
                    },
                    "area_type": {
                        "type": "string",
                        "enum": ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"],
                        "description": "Type of layout area."
                    },
                    "is_ready_to_move": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "1 if ready to move, 0 if under construction."
                    }
                },
                "required": ["location", "total_sqft", "bhk", "bath", "balcony"]
            }
        }
    }
]


# =====================================================================
# 4. TOOL REGISTRY & DISPATCHER
# =====================================================================
def execute_tool(name: str, args: dict) -> dict:
    """Routes tool execution requests to the correct python functions."""
    if name == "search_listings":
        return search_listings(
            location=args.get("location"),
            bhk=args.get("bhk"),
            budget_min_lakhs=args.get("budget_min_lakhs"),
            budget_max_lakhs=args.get("budget_max_lakhs"),
            limit=args.get("limit", 5)
        )
    elif name == "compare_locations":
        return compare_locations(
            location_a=args.get("location_a"),
            location_b=args.get("location_b")
        )
    elif name == "get_ml_prediction":
        return get_ml_prediction(
            location=args.get("location"),
            total_sqft=args.get("total_sqft"),
            bhk=args.get("bhk"),
            bath=args.get("bath"),
            balcony=args.get("balcony"),
            area_type=args.get("area_type"),
            is_ready_to_move=args.get("is_ready_to_move", 1)
        )
    elif name == "explain_ml_prediction":
        return explain_ml_prediction(
            location=args.get("location"),
            total_sqft=args.get("total_sqft"),
            bhk=args.get("bhk"),
            bath=args.get("bath"),
            balcony=args.get("balcony"),
            area_type=args.get("area_type"),
            is_ready_to_move=args.get("is_ready_to_move", 1)
        )
    else:
        return {"error": f"Unknown tool name: {name}"}


# =====================================================================
# 5. FASTAPI REQUEST SCHEMAS & CONTROLLER
# =====================================================================
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[dict]] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


async def resolve_ollama_model(settings: dict) -> str:
    """Helper to query available local Ollama models and find qwen or fallback."""
    url = f"{settings['ollama_url']}/api/tags"
    default_model = settings["ollama_model"]
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                models = [m["name"] for m in data.get("models", [])]
                if default_model in models:
                    return default_model
                
                # Check for variations of qwen
                for m in models:
                    if "qwen" in m.lower():
                        return m
                # Fallback to first available if none is qwen
                if models:
                    return models[0]
    except Exception:
        pass
    return default_model


@router.post("/chat")
async def chat_with_agent(req: ChatRequest):
    """
    Main agent endpoint. Handles conversational history, routes system prompts, 
    and resolves sequential tool-calling loops with local Ollama.
    """
    settings = get_settings()
    model = await resolve_ollama_model(settings)
    ollama_endpoint = f"{settings['ollama_url']}/api/chat"
    
    system_prompt = (
        "You are the Bangalore Property AI Advisor, an expert local real estate advisor. "
        "You are integrated into a machine learning dashboard. "
        "CRITICAL: You MUST call one of the provided tools to query real data before stating any prices, "
        "average prices, listing counts, or area comparisons. Do not rely on your own memory for pricing data. "
        "If you do not call the correct tool, your response will be completely wrong. "
        "For comparisons, always call 'compare_locations'.\n\n"
        "SECURITY POLICY:\n"
        "If the user asks to delete, drop, remove, clear, or modify any listings or data in the dataset, "
        "you MUST politely refuse. State that you do not have write or deletion permissions on the dataset "
        "for safety and data integrity reasons, and that dataset modifications must be done manually or via the Data Portal.\n\n"
        "Instructions:\n"
        "1. If the user asks for house recommendations or averages, call 'search_listings'.\n"
        "2. If the user asks to compare areas, call 'compare_locations'.\n"
        "3. If the user asks to predict, estimate, or evaluate a specific property, call 'get_ml_prediction'. You can also call 'explain_ml_prediction' if they want details on why it is priced that way.\n"
        "4. Format your final responses in clean, structured Markdown. Use tables for listings, comparisons, or predictions.\n"
        "5. Keep responses concise and practical. Highlight prices in Bold."
    )
    
    # Reconstruct messages list for Ollama
    chat_messages = []
    has_system = False
    
    for msg in req.messages:
        if msg.role == "system":
            has_system = True
            chat_messages.append({"role": msg.role, "content": msg.content})
        else:
            item = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                item["tool_calls"] = msg.tool_calls
            chat_messages.append(item)
            
    if not has_system:
        chat_messages.insert(0, {"role": "system", "content": system_prompt})
        
    async with httpx.AsyncClient() as client:
        # Loop for handling sequential tool calling (up to 3 levels of depth)
        for _ in range(3):
            payload = {
                "model": model,
                "messages": chat_messages,
                "tools": AGENT_TOOLS,
                "stream": False
            }
            
            try:
                logger.info("Sending chat request to Ollama (%s)", model)
                response = await client.post(ollama_endpoint, json=payload, timeout=30.0)
                
                if response.status_code != 200:
                    logger.error("Ollama HTTP Error: %s", response.text)
                    raise HTTPException(
                        status_code=502, 
                        detail=f"Ollama server returned an error: {response.text}"
                    )
                    
                res_data = response.json()
                assistant_message = res_data.get("message", {})
                tool_calls = assistant_message.get("tool_calls", [])
                
                if not tool_calls:
                    # Return final text message to frontend
                    return assistant_message
                    
                # Append assistant tool-calling message to history
                chat_messages.append(assistant_message)
                
                # Execute each requested tool call
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name")
                    args = func.get("arguments", {})
                    
                    logger.info("Executing local tool '%s' with args %s", name, args)
                    tool_result = execute_tool(name, args)
                    
                    # Append tool result to history
                    chat_messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result)
                    })
                    
            except httpx.RequestError as e:
                logger.error("Failed to connect to local Ollama: %s", e)
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not connect to Ollama at {settings['ollama_url']}. Please ensure Ollama is running (`ollama serve`)."
                )
                
        raise HTTPException(status_code=500, detail="Failed to resolve agent response (too many tool calls)")
