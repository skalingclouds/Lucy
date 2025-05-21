#!/usr/bin/env python
"""
Test script to verify PDF extraction functionality.
This script will:
1. Generate a SAS URL for a test PDF
2. Extract text from that PDF
3. Print the result
"""

import asyncio
from dotenv import load_dotenv
import logging
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure environment is loaded
load_dotenv()

# Import functions from apex.py
from apex import generate_sas_url, extract_text_from_pdf

async def main():
    # Test storage account name
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    if not storage_account:
        logger.error("AZURE_STORAGE_ACCOUNT_NAME not set")
        return
    
    # Test with a known PDF path from our earlier successful SAS test
    container = "lucyrag"
    pdf_path = "New Hampshire Ball Bearings, Inc/NHBB - Pedro Perez.pdf"
    
    # Full blob URL
    blob_url = f"https://{storage_account}.blob.core.windows.net/{container}/{pdf_path}"
    logger.info(f"Generating SAS URL for: {blob_url}")
    
    # Generate SAS URL
    sas_url = generate_sas_url(blob_url)
    if not sas_url or sas_url.startswith("ERROR"):
        logger.error(f"Failed to generate SAS URL: {sas_url}")
        return
    
    logger.info(f"Successfully generated SAS URL: {sas_url[:50]}...")
    
    # Extract text from PDF
    text = await extract_text_from_pdf(sas_url)
    
    if text.startswith("ERROR"):
        logger.error(f"Failed to extract text: {text}")
    else:
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        logger.info("First 500 characters:")
        logger.info(text[:500])

if __name__ == "__main__":
    asyncio.run(main()) 