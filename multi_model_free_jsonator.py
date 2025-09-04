from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from groq import Groq
import os
from dotenv import load_dotenv
import json
import pandas as pd
#batch processing
import pandas as pd
from tqdm import tqdm

#from langchain_core.runnables import Runnable
import time

load_dotenv(override=True)
print("API KEY FOUND?", os.getenv("GROQ_API_KEY_2") is not None)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#running api_2 scout model on 30 addr/call

# Suggested Updates to Your Code to Add Model Switching Scheduler

from groq import Groq

# === Define your available models ===
LLM_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "limit_type": "calls",
        "max_calls": 1000,
        "used_calls": 0,
        "tokens_per_minute": 30000,
        "tokens_this_minute": 0,
        "minute_start_time": time.time(),
        "batched_input_capacity":30,
        "active": True
    },
    "llama-3.1-8b-instant": {
        "limit_type": "tokens",
        "max_tokens": 500000,
        "used_tokens": 0,
        "tokens_per_minute": 6000,
        "tokens_this_minute": 0,
        "minute_start_time": time.time(),
        "batched_input_capacity":10,
        "active":True
    },
    "llama-3.3-70b-versatile": {
        "limit_type": "tokens",
        "max_tokens": 100000,
        "used_tokens": 0,
        "tokens_per_minute": 12000,
        "tokens_this_minute": 0,
        "minute_start_time": time.time(),
        "batched_input_capacity":10,
        "active":True
    },
    "llama3-70b-8192": {
        "limit_type": "tokens",
        "max_tokens": 500000,
        "used_tokens": 0,
        "tokens_per_minute": 6000,
        "tokens_this_minute": 0,
        "minute_start_time": time.time(),
        "batched_input_capacity":10,
        "active":True
    },
    
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "limit_type": "calls",
        "max_calls": 1000,
        "used_calls": 0,
        "tokens_per_minute": 6000,
        "tokens_this_minute": 0,
        "minute_start_time": time.time(),
        "batched_input_capacity":10,
        "active":True
    },
    "qwen/qwen3-32b": {#does not follow strict json format all the time with the used prompt 
        "limit_type": "calls",
        "max_calls": 1000,
        "used_calls": 0,
        "tokens_per_minute": 6000,
        "tokens_this_minute": 0,
        "minute_start_time": time.time(),
        "batched_input_capacity":10,
        "active":False
    },
}
#current_model="meta-llama/llama-4-scout-17b-16e-instruct"
# === Request with auto-failover ===
def request_with_failover(messages):
    for model_name, info in LLM_MODELS.items():
        if not info["active"]:
            continue

        # 🕒 Per-minute throttle check
        now = time.time()
        elapsed = now - info["minute_start_time"]
        if elapsed >= 60:
            info["tokens_this_minute"] = 0
            info["minute_start_time"] = now
        # Estimate how many tokens this request will need
        est_input = estimate_tokens(messages)
        est_output = 500 # you can tweak this per model if needed
        total_est = est_input + est_output

        if info["tokens_this_minute"] + total_est > info["tokens_per_minute"]:
            wait_time = 60 - elapsed
            print(f"⏳ Throttle: sleeping {round(wait_time, 2)}s for {model_name} to reset token quota...")
            time.sleep(wait_time)
            info["tokens_this_minute"] = 0
            info["minute_start_time"] = time.time()
        
        try:
            current_model=model_name
            print(f"🧠 Using model: {model_name}")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0
            )
            input_tokens = est_input
            output_tokens = estimate_tokens(response.choices[0].message.content)

            # Update token usage
            info["tokens_this_minute"] += input_tokens + output_tokens

            if info["limit_type"] == "calls":
                info["used_calls"] += 1
                if info["used_calls"] >= info["max_calls"]:
                    print(f"🚫 Max calls reached for {model_name}")
                    info["active"] = False

            elif info["limit_type"] == "tokens":
                info["used_tokens"] += input_tokens + output_tokens
                if info["used_tokens"] >= info["max_tokens"]:
                    print(f"🚫 Max tokens reached for {model_name}")
                    info["active"] = False

            return response

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error from model {model_name}: {error_msg}")
            if "Rate limit" in error_msg or "429" in error_msg:
                print(f"🔕 Rate limit hit for {model_name}. Marking as inactive.")
            info["active"] = False

    raise RuntimeError("🛑 All models have been exhausted or failed.")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)
expected_schema={
     "house_number":"string","plot_number":"string","floor":"string", "road_details":"string",
     "khasra_number":"string", 
     "block":"string", "apartment":"string", "landmark":"string", 
     "locality":"string", "area":"string","locality2":"string","village":"string", "pincode":"string", "complete_address":"string"
}
expected_schemas=json.dumps(expected_schema).replace("{","{{").replace("}","}}")

system_prompt = {
   "role": "system",
    "content": ("RULES:"
    "-you are an api that outputs json in this strict schema"+ expected_schemas + "\n\n"
    "-qwen stricty output only json no verbose output tokens of thinking processes"
    "-strictly follow and use example as refrence."
    "- If a key is missing → fill as ''."
    "- Landmark → only if it comes after 'near'."
    "- Khasra → any number starting with 'KH', 'Khasra', etc → put in 'khasra_number'."
    "- Words 'Gali', 'Street', 'Lane', 'Mohalla', 'Colony', 'Road',all represent the same information→ so, merge them (comma-separated) into 'road_details'."
    "- Blocks → only if appearing after or before word 'Block'.Also treat Pusta numbers as blocks."
    "- Area → comes before suffixes like 'Nagar', 'Pur', etc."
    "- 'G No-18' → means 'Gali No. 18' → goes in 'road_details'."
    "- No duplicate/repetation of content across fields."
    "- Dont guess or hallucinate fields."
    "- In 'house_number' → keep full phrases (e.g., 'House No. 5', not just '5'), and also keep the old house no. details too."
    "- In 'complete_address' → expand abbreviations → e.g., 'EXTN'→'Extension', 'CMPLX'→'Complex'."
    "Now, examples:"'['
     '\{'
    '"house_number": "","plot_number": "Plot No. 33, Old Plot No. 15","floor": "","road_details": "Service Road, Lane No. 8","khasra_number": "64/17/2","block": "",'
    '"apartment": "","landmark": "Metro Pillar 123","locality": "Rajendra Park","area": "Najafgarh","locality2": "Nangli Sakrawati","village": "","pincode": "110043",'
    '"complete_address": "Plot No. 33, Old Plot No. 15, Service Road, Lane No. 8, Khasra No-64/17/2, Near Metro Pillar 123, Rajendra Park, Najafgarh, Nangli Sakrawati, Delhi - 110043"'
    '\},'
    '\{'
    '"house_number": "D-3/45","plot_number": "","floor": "Second Floor","road_details": "Gali No. 9, CRPF Camp Road","khasra_number": "","block": "School Block","apartment": "","landmark": "CRPF Camp",'
    '"locality": "Sonia Vihar","area": "North East Delhi","locality2": "","village": "","pincode": "110094",'
    '"complete_address": "D-3/45, Second Floor, Gali No. 9, Near CRPF Camp Road, School Block, Near CRPF Camp, Sonia Vihar, North East Delhi, Delhi - 110094"'
    '\}'
    '\{'
    '"house_number": "House No. 17-B",''"plot_number": "Plot No. 45, Old Plot No. 22",''"floor": "Third Floor",''"road_details": "Gali No. 4, Street No. 2, Mohalla Shyam Nagar, Main Road",'
    '"khasra_number": "KH No. 78/3/2",''"block": "B Block",''"apartment": "Shyam Residency Apartment",''"landmark": "Shiv Mandir",''"locality": "Shyam Nagar",''"area": "Laxmi Nagar",'
    '"locality2": "Patparganj Village",''"village": "Patparganj",''"pincode": "110092",'
    '"complete_address": "House No. 17-B, Plot No. 45, Old Plot No. 22, Third Floor, Gali No. 4, Street No. 2, Mohalla Shyam Nagar, Main Road, KH No. 78/3/2, B Block, Shyam Residency Apartment, Near Shiv Mandir, Shyam Nagar, Laxmi Nagar, Patparganj Village, Delhi - 110092"'
    '\}'']'
)
}

def make_messages(address_block):
    addresses = address_block.strip().split("\n")
    numbered_input = "\n".join([f"{i+1}. {a}" for i, a in enumerate(addresses)])
    return [
        system_prompt,
        {"role": "user", "content": f"Segment the following addresses:\n{numbered_input}\n\nReturn each as a separate JSON object with only one set of curly braces per address."}
    ]


df = pd.read_csv("exp1.csv")
start_add=183070
addresses=df["CUSTADDR"].iloc[start_add:].sample(n=30000,random_state=42).tolist()
# Break into small batches (e.g., 5 per call)
#print("Capacity according to ",current_model,"\n")
#capacity = LLM_MODELS[current_model]["batched_input_capacity"]
batched_input = ["\n".join(addresses[i:i+15]) for i in range(0, len(addresses),15)]#alternativve of batching 


import re
def extract_json(text):
    try:
        # First, try parsing the entire response as a list
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        return parsed
    except Exception as e:
        print("⚠️ Full JSON parse failed, falling back to block extraction...")

    # Fallback: extract individual JSON objects
    matches = re.findall(r"\{[\s\S]*?\}", text)
    print(f"🧪 Fallback: Found {len(matches)} JSON-like blocks")
    parsed = []
    for match in matches:
        try:
            parsed.append(json.loads(match))
        except Exception as e:
            print(f"❌ Failed to parse:\n{match}\nError: {e}")
    return parsed
def merge_fields(df):
    # Fields to merge
    gali_related = ["gali", "street", "mohalla", "colony", "road", "lane_number", "street_number"]
    property_related = ["property_number", "prop_number","old_property_number"]

    # Safely filter only existing columns
    existing_gali_cols = [col for col in gali_related if col in df.columns]
    existing_prop_cols = [col for col in property_related if col in df.columns]

    # Merge only if there are columns to merge
    if existing_gali_cols:
        df["road_details"] = df[existing_gali_cols].astype(str).apply(
            lambda row: ", ".join([v.strip() for v in row if v.strip() and v.lower() != "nan"]),
            axis=1
        )
        df.drop(columns=existing_gali_cols, inplace=True)

    if existing_prop_cols:
        df["property_details"] = df[existing_prop_cols].astype(str).apply(
            lambda row: ", ".join([v.strip() for v in row if v.strip() and v.lower() != "nan"]),
            axis=1
        )
        df.drop(columns=existing_prop_cols, inplace=True)

    return df


def align_columns(df, column_order):
    # Add missing columns as empty strings
    for col in column_order:
        if col not in df.columns:
            df[col] = ""
    # Sort columns: known → unknown (for future flexibility)
    known = [c for c in column_order if c in df.columns]
    unknown = [c for c in df.columns if c not in column_order]
    return df[known + unknown]
def save_to_csv(parsed_data, column_order):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df = pd.json_normalize(parsed_data)
    
    #df = merge_fields(df)
    df = align_columns(df, column_order)  # Put gali_details in desired order
    output_path = "multi_model_run_on_(" + str(start_add) + "," + str(start_add+len(all_parsed_jsons)) + ").csv"
    df.to_csv(output_path, index=False)
    print(f"📁 Saved CSV to → {output_path}")
    print("🧾 CSV Headers:", df.columns.tolist())
# Desired order (including keys from your schema + extras if expected)
DESIRED_COLUMNS = [
     "house_number","plot_number","floor", "road_details",
     "khasra_number", 
     "block", "apartment", "landmark", 
     "locality", "area","locality2","village", "pincode", "complete_address"]



Calls_min = 15# Safe for 30 calls/min limit
MAX_CALLS_PER_MIN = 30
MAX_TOKENS_PER_MIN = 30000
AVG_INPUT_OUTPUT_TOKENS = 300  # rough estimate per address (input+output)
total_daily_tokens_used = 0



def estimate_tokens(text):
    if isinstance(text, list):
        # Handle list of dicts (e.g., chat messages)
        text = " ".join([m.get("content", "") for m in text if isinstance(m, dict)])
    elif not isinstance(text, str):
        text = str(text)
    return int(len(text.split()) * 1.3)


batch_start = 0
timestamp_minute = time.time()
token_counter = 0
call_counter = 0
all_parsed_jsons = []
class TokenLimitReached(Exception):
    pass

#parsing loop
try:
    while batch_start < len(batched_input):
        batch =batched_input[batch_start: batch_start + Calls_min]
        for addr in batch:
            print(f"\n📬 Processing batch:\n{addr}\n")
            messages = make_messages(addr)
            response = request_with_failover(messages)
            
            message_str = "\n".join([m["content"] for m in messages if "content" in m])
            text_output = response.choices[0].message.content
            print("🧠 LLM Output:\n", text_output, "\n")

            batch_jsons = extract_json(text_output)
            if isinstance(addr, str):
                expected = len(addr.strip().split("\n"))
            elif isinstance(addr, list):
                expected = len(addr)
            else:
                expected = 0  # or raise Exception("Unexpected input type")

            actual = len(batch_jsons)
            if expected != actual:
                print(f"⚠️ Output mismatch → Expected: {expected}, Got: {actual}")
                print("🔍 Input:\n", addr)
                print("🔎 Output:\n", text_output)

            all_parsed_jsons.extend(batch_jsons)
            print(f"✅ Total parsed so far: {len(all_parsed_jsons)}")

            call_counter += 1
            
            tokens_this_call = estimate_tokens(message_str) + estimate_tokens(text_output)

            token_counter += tokens_this_call
            total_daily_tokens_used += tokens_this_call

            
            
        batch_start += Calls_min
    # ✅ All done, save results
    print("✅ All batches processed. Saving final CSV...")
    save_to_csv(all_parsed_jsons, DESIRED_COLUMNS)

except RuntimeError as all_models_dead:
    print(str(all_models_dead))
    save_to_csv(all_parsed_jsons, DESIRED_COLUMNS)
    print("💾 All models failed. Exiting gracefully.")
except Exception as e:
    print(f"💥 Unhandled error: {e}")
    print("📦 Saving partial data just in case...")
    save_to_csv(all_parsed_jsons, DESIRED_COLUMNS)
