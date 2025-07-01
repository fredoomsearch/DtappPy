import hashlib
from fastapi import FastAPI, HTTPException, Request, logger
from fastapi.responses import HTMLResponse
from api import crypto, events, data_processing, ml_models
from utils import database
from models import crypto_model, event_model, scraped_data_model
from fastapi.middleware.cors import CORSMiddleware
from api.data_processing import router as news_router


app.include_router(crypto.router)
app.include_router(events.router)
app.include_router(data_processing.router)
app.include_router(ml_models.router)
app.include_router(csv_download.router, prefix="/api")  # <--- FIXED
app.include_router(news_router, prefix="/api/news")

# ✅ CORS for frontend (localhost & Render)
origins = [
    "http://localhost:4200",
    "https://dtapppyfront.onrender.com"
]

app.add_middleware(
    CORSMiddleware,  # <--- This is required!
    allow_origins=[
        "https://dtapppyfront.onrender.com",
        "http://localhost:4200"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)







app = FastAPI()

# CORS configuration
origins = [
    "https://dtapppyfront.onrender.com",
    "http://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(crypto.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(data_processing.router, prefix="/api")
app.include_router(ml_models.router, prefix="/api")

# ...rest of your code (DB setup, PayU, etc.)...










from sqlalchemy.orm import Session
from utils.database import engine, SessionLocal
from models.crypto_model import Base
from models.event_model import Base as eventBase
from models.scraped_data_model import Base as scrapedBase

@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    eventBase.metadata.create_all(bind=engine)
    scrapedBase.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# ... (rest of your code)

@app.post("/api/payu/notification")
async def payu_notification(request: Request):
    try:
        # PayU sends application/x-www-form-urlencoded data
        form_data = await request.form()
        logger.info(f"Received PayU IPN: {form_data}")

        # Extract relevant fields for signature validation
        # Adjust these field names based on PayU's actual IPN structure
        # (Refer to PayU IPN documentation for exact field names)
        merchant_id = form_data.get("merchant_id")
        reference_code = form_data.get("reference_code")
        transaction_value = form_data.get("value") # PayU might send 'value' or 'TX_VALUE'
        currency = form_data.get("currency")
        transaction_state = form_data.get("state_pol") # '4' for approved, '6' for declined, etc.
        signature_payu = form_data.get("sign") # The signature sent by PayU

        # --- IPN Signature Validation (Crucial Security Step) ---
        # PayU IPN signature format: MD5(ApiKey~merchantId~referenceCode~value~currency~statePol)
        # Ensure 'value' (amount) is correctly formatted (e.g., "100.00")
        
        # Example formatting for value (might need adjustment based on PayU's exact format in IPN)
        # PayU often sends value with decimals. Ensure consistent formatting.
        if transaction_value:
            # Try to convert to float and format to 2 decimal places
            try:
                formatted_value = "{:.2f}".format(float(transaction_value))
            except ValueError:
                formatted_value = transaction_value # Fallback if not a valid number
        else:
            formatted_value = ""

        import os
        PAYU_API_KEY = os.getenv("PAYU_API_KEY", "ehCsEmI12qCYcU6h3RJc9q6tBR")  # Replace with your actual key or use environment variable

        string_to_hash = f"{PAYU_API_KEY}~{merchant_id}~{reference_code}~{formatted_value}~{currency}~{transaction_state}"
        
        calculated_signature = hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()

        logger.info(f"String to hash for IPN: {string_to_hash}")
        logger.info(f"Calculated IPN Signature: {calculated_signature}")
        logger.info(f"PayU Sent IPN Signature: {signature_payu}")

        if calculated_signature == signature_payu:
            logger.info("IPN Signature validated successfully!")
            # Process the payment status here
            # E.g., Update database, send confirmation email, trigger prediction
            if transaction_state == "4": # Example: Approved transaction
                logger.info(f"Payment for reference {reference_code} APPROVED.")
                # You would perform your business logic here (e.g., mark order as paid, send prediction)
                return HTMLResponse(content="OK", status_code=200) # PayU expects 'OK' for successful processing
            else:
                logger.warning(f"Payment for reference {reference_code} FAILED/PENDING. State: {transaction_state}")
                return HTMLResponse(content="OK", status_code=200) # Always return 200 OK to acknowledge receipt
        else:
            logger.error("IPN Signature validation FAILED!")
            return HTMLResponse(content="Signature Mismatch", status_code=400) # Bad Request if signature doesn't match

    except Exception as e:
        logger.exception(f"Error processing PayU IPN: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    
