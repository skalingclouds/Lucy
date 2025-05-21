import asyncio
import logging
import argparse
import json
from typing import Optional

import httpx

# Re-use helpers from the main application
from apex import (
    execute_search,  # async
    extract_pdf_url,  # sync
    generate_sas_url,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("ApexLinkProbe")


async def _head_request(url: str) -> tuple[int, str]:
    """Perform a HEAD request and return (status_code, reason)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.head(url)
        return r.status_code, r.reason_phrase


async def probe(apex_id: str, address: str = "", *, verbose: bool = False) -> dict:
    """End-to-end diagnostic.

    Returns a dict with keys: search, sas, http.
    """
    result: dict[str, Optional[str | int]] = {
        "search": None,
        "sas": None,
        "http_status": None,
    }

    # ------------------------------------------------------------------
    # A) Azure AI Search – find the notice
    # ------------------------------------------------------------------
    user_data = {"apex_id": apex_id, "address": address}
    docs = await execute_search(user_data)
    if not docs:
        logger.error("Azure Search returned 0 hits")
        result["search"] = "0 results"
        return result

    # Find the first document that resolves to a valid PDF SAS URL
    sas_url: Optional[str] = None
    for idx, doc in enumerate(docs):
        if verbose:
            logger.info("Examining search doc #%d", idx + 1)
            logger.debug(json.dumps(dict(doc), indent=2)[:400])

        extracted = extract_pdf_url(doc)
        if extracted and not extracted.startswith("ERROR"):
            sas_url = extracted  # already a SAS URL (extract_pdf_url performs generation)
            logger.info("Found PDF SAS for doc #%d", idx + 1)
            break

    if not sas_url:
        logger.error("No valid PDF SAS URL found in %d search results", len(docs))
        result["search"] = "no pdf"
        return result

    result["search"] = "ok"
    result["sas"] = "ok"
    logger.info("SAS URL  : %s", sas_url)

    # ------------------------------------------------------------------
    # B) HTTP reachability check
    # ------------------------------------------------------------------
    status_code, reason = await _head_request(sas_url)
    result["http_status"] = status_code
    logger.info("HTTP HEAD: %s (%s)", status_code, reason)

    return result


async def _run_cli():
    ap = argparse.ArgumentParser(description="End-to-end PDF / SAS probe")
    ap.add_argument("--apex-id", required=True, help="APEX ID (file name stem)")
    ap.add_argument("--address", default="", help="Address string (optional)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    summary = await probe(args.apex_id, args.address, verbose=args.verbose)

    print("\n==== PROBE SUMMARY ====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(_run_cli()) 