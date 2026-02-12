import os
import re
import pandas as pd
from fastapi import FastAPI, Form, File, UploadFile, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pdfplumber
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
import uvicorn

# Download NLTK data if needed (for sentence tokenization)
nltk.download('punkt', quiet=True)

app = FastAPI(title="AI Research Portal")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------- Advanced Open-Source Extraction Functions ----------

def extract_income_statement(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    metadata = {
        "currency": "USD",
        "units": "Actual",
        "periods": []
    }
    
    # Advanced currency detection (more currencies)
    currency_patterns = {
        "USD": r'\$|USD|US Dollar',
        "EUR": r'€|EUR|Euro',
        "INR": r'₹|INR|Indian Rupee',
        "GBP": r'£|GBP|Pound Sterling',
        "JPY": r'¥|JPY|Japanese Yen',
        "CNY": r'¥|CNY|Chinese Yuan'
    }
    for curr, pattern in currency_patterns.items():
        if re.search(pattern, text, re.I):
            metadata["currency"] = curr
            break
    
    # Advanced units detection (handle billions, lakhs, etc.)
    units_patterns = [
        (r'billion', 'billions'),
        (r'million', 'millions'),
        (r'thousand', 'thousands'),
        (r'lakh', 'lakhs'),
        (r'crore', 'crores'),
        (r'\(in\s+([\w\s]+?)(?:s)?\s*(?:of|dollar)', lambda m: m.group(1).strip() + "s")
    ]
    for pattern, unit in units_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            metadata["units"] = unit(match) if callable(unit) else unit
            break
    
    # Advanced periods detection (handle fiscal years, quarters, more robust)
    period_candidates = []
    for line in lines:
        # Find lines with multiple years like "2023 2022 2021" or "FY23 FY22"
        years = re.findall(r'\b(?:FY|Q)?(\d{2,4})(?:/\d{2})?\b', line)
        unique_years = sorted(set(years), reverse=True)
        if 2 <= len(unique_years) <= 5:  # Up to 5 years
            period_candidates.append(unique_years)
    if period_candidates:
        metadata["periods"] = max(period_candidates, key=len)  # Take longest match
    if not metadata["periods"]:
        metadata["periods"] = ["Current Year", "Previous Year"]
    
    num_periods = len(metadata["periods"])
    
    # Extended standard line items and synonyms (more comprehensive)
    standard_order = [
        "Total Revenue",
        "Cost of Revenue",
        "Gross Profit",
        "Research and Development",
        "Selling, General and Administrative Expenses",
        "Operating Income (Loss)",
        "Interest Expense, net",
        "Other Income (Expense), net",
        "Income Before Tax",
        "Income Tax Expense",
        "Net Income (Loss)",
        "Basic Earnings Per Share",
        "Diluted Earnings Per Share"
    ]
    
    synonym_map = {
        "Total Revenue": ["total revenue", "net revenue", "net sales", "revenue", "sales", "operating revenue", "turnover"],
        "Cost of Revenue": ["cost of revenue", "cost of goods sold", "cogs", "cost of sales", "direct costs"],
        "Gross Profit": ["gross profit", "gross margin", "gross income"],
        "Research and Development": ["research and development", "r&d", "r and d", "research development"],
        "Selling, General and Administrative Expenses": ["selling, general and administrative", "sg&a", "operating expenses", "s g & a", "general expenses", "admin expenses"],
        "Operating Income (Loss)": ["operating income", "operating profit", "ebit", "income from operations", "operating loss", "operating result"],
        "Interest Expense, net": ["interest expense", "net interest", "finance costs", "interest income net"],
        "Other Income (Expense), net": ["other income", "other expense", "non-operating", "sundry income", "miscellaneous expense"],
        "Income Before Tax": ["income before tax", "pre-tax income", "pretax income", "earnings before tax", "profit before tax", "pbt"],
        "Income Tax Expense": ["income tax", "tax expense", "provision for income taxes", "taxes", "deferred tax"],
        "Net Income (Loss)": ["net income", "net profit", "net loss", "profit after tax", "bottom line", "net earnings"],
        "Basic Earnings Per Share": ["basic eps", "basic earnings per share", "eps basic"],
        "Diluted Earnings Per Share": ["diluted eps", "diluted earnings per share", "eps diluted"]
    }
    
    extracted_values = {item: ["N/A"] * num_periods for item in standard_order}
    
    def get_numbers_from_line(line: str):
        # Advanced number parsing: handles (negatives), commas, decimals, scientific notation
        pattern = r'\(?\s*-?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:E[+-]?\d+)?\s*\)?'
        matches = re.findall(pattern, line)
        values = []
        for m in matches:
            num_str = m.strip("() \t").replace(",", "").replace(" ", "")
            try:
                num = float(num_str)
                if "(" in m or "-" in m:
                    num = abs(num) if "(" in m else num  # Assume ( ) means negative
                    num = -num
                values.append(num)
            except ValueError:
                pass
        return values
    
    # Scan lines with window for better matching
    for idx in range(len(lines)):
        line = lines[idx]
        lower = line.lower()
        numbers = get_numbers_from_line(line)
        
        # Look ahead up to 3 lines for numbers (handles wrapped tables)
        lookahead = 3
        for offset in range(1, lookahead + 1):
            if idx + offset < len(lines) and len(numbers) < num_periods:
                numbers += get_numbers_from_line(lines[idx + offset])
        
        for standard, synonyms in synonym_map.items():
            if extracted_values[standard][0] != "N/A":
                continue  # Skip if already filled
            if any(syn in lower for syn in synonyms):
                if numbers:
                    vals = numbers[:num_periods]
                    pad = ["N/A"] * (num_periods - len(vals))
                    final_vals = [str(v) if v == int(v) else str(v) for v in vals] + pad
                    extracted_values[standard] = final_vals[:num_periods]
                    break  # Assume one match per line
    
    line_items = [{"item": item, "values": extracted_values[item]} for item in standard_order]
    
    return {"metadata": metadata, "line_items": line_items}

def summarize_earnings(text: str):
    sentences = sent_tokenize(text)  # Using NLTK for better sentence splitting
    
    # Advanced sentiment with polarity and subjectivity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    tone = "optimistic" if polarity > 0.1 else "pessimistic" if polarity < -0.1 else "cautious" if subjectivity > 0.5 else "neutral"
    confidence = "high" if abs(polarity) > 0.3 or subjectivity > 0.7 else "medium" if abs(polarity) > 0.1 else "low"
    
    # Extract positives/concerns with threshold and length limit
    key_positives = [s.strip()[:200] for s in sentences if TextBlob(s).sentiment.polarity > 0.15][:5]
    key_concerns = [s.strip()[:200] for s in sentences if TextBlob(s).sentiment.polarity < -0.15][:5]
    
    # Extended keywords for guidance
    guidance_kw = ["guidance", "expect", "outlook", "forecast", "project", "anticipate", "looking ahead", "plan to", "target", "estimate", "predict"]
    guidance_sents = [s for s in sentences if any(k in s.lower() for k in guidance_kw)]
    forward_guidance = ". ".join(guidance_sents[:4]) or "No specific guidance provided"
    if len(forward_guidance) > 500:
        forward_guidance = forward_guidance[:500] + "..."
    
    # Capacity with more keywords
    capacity_kw = ["capacity", "utilization", "data center", "production capacity", "facility", "utilisation", "occupancy", "load factor"]
    capacity_sents = [s for s in sentences if any(k in s.lower() for k in capacity_kw)]
    capacity_utilization = ". ".join(capacity_sents[:3]) or "Not mentioned"
    
    # Growth initiatives with positive sentiment filter
    growth_kw = ["new", "launch", "acquisition", "initiative", "expansion", "investment", "platform", "product", "partnership", "strategy", "growth plan"]
    growth_sents = [s.strip() for s in sentences if any(k in s.lower() for k in growth_kw) and TextBlob(s).sentiment.polarity >= 0][:3]
    growth_initiatives = growth_sents or ["Not mentioned"]
    
    return {
        "tone": tone,
        "confidence": confidence,
        "key_positives": key_positives or ["None mentioned"],
        "key_concerns": key_concerns or ["None mentioned"],
        "forward_guidance": forward_guidance,
        "capacity_utilization": capacity_utilization,
        "growth_initiatives": growth_initiatives
    }

# ---------- Routes ----------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.post("/process")
async def process_document(request: Request, tool: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return templates.TemplateResponse("result.html", {"request": request, "error": "Only PDF files are allowed."})
    
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    if not text.strip():
        return templates.TemplateResponse("result.html", {"request": request, "error": "Could not extract text from PDF."})
    
    try:
        if tool == "financial_extraction":
            data = extract_income_statement(text)
            
            periods = data["metadata"]["periods"]
            rows = []
            for item in data["line_items"]:
                row = {"Line Item": item["item"]}
                for i, period in enumerate(periods):
                    val = item["values"][i] if i < len(item["values"]) else "N/A"
                    row[period] = val
                rows.append(row)
            
            df = pd.DataFrame(rows).set_index("Line Item")
            output_file = f"outputs/{os.path.splitext(file.filename)[0]}_income_statement.xlsx"
            df.to_excel(output_file)
            
            return templates.TemplateResponse("result.html", {
                "request": request,
                "tool": tool,
                "metadata": data["metadata"],
                "table_html": df.to_html(classes="table table-striped table-hover table-bordered", border=0),
                "download_file": os.path.basename(output_file)
            })
        
        else:  # earnings_summary
            summary = summarize_earnings(text)
            return templates.TemplateResponse("result.html", {
                "request": request,
                "tool": tool,
                "summary": summary
            })
            
    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Processing failed: {str(e)}"
        })

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"outputs/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)