# Azure AI Projects client and tools
import atexit
from datetime import datetime, timedelta, timezone
import os
import sys
import base64  # For HTTP requests to agent portal
import json  # For JSON handling
import logging
import asyncio
import smtplib
import aiohttp
from unittest.mock import MagicMock
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from urllib.parse import quote, unquote, urlparse
import chainlit as cl
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas, BlobClient
from azure.ai.projects import AIProjectClient
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ServiceResponseError, ClientAuthenticationError, HttpResponseError
from azure.ai.projects.models._patch import (
    AzureAISearchTool,
    FileSearchTool,
    FunctionTool,
    ToolSet,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv
from user_functions import setup_dynamics_functions
import uuid
from backoff import expo, on_exception

# for simple user_input parsing

# For generating unique IDs
trace = MagicMock()
trace.get_tracer = lambda name: MagicMock()
trace.get_current_span = lambda: MagicMock()
trace.set_tracer_provider = lambda provider: None


def trace_function(*args, **kwargs):
    def decorator(func):
        return func

    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator


# Load environment variables
load_dotenv()


# Force light theme
os.environ["CHAINLIT_CONFIG"] = ".chainlit/config.toml"
os.environ["CHAINLIT_LIGHT_THEME"] = "true"
os.environ["CHAINLIT_THEME_LIGHT"] = "true"
os.environ["CHAINLIT_THEME_DARK"] = "false"
os.environ["CHAINLIT_HIDE_BRANDING"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CHAINLIT_TELEMETRY_ENABLED"] = "true"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ApexAI")
logger.info("✅ Apex AI Assistant starting up...")

# ---------------- Startup diagnostics (defined later) -----------------
# We need generate_sas_url first; diagnostics will be executed further down.

# Check critical environment variables
critical_env_vars = [
    "AZURE_PROJECT_CONNSTRING",
    "AZURE_SEARCH_INDEX_NAME",
    "AZURE_GPT_MODEL",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_ACCOUNT_NAME",
    "AI_SEARCH_CONNECTION_NAME",
]
for var in critical_env_vars:
    if os.getenv(var):
        logger.info(f"✅ Environment variable {var} is set")
    else:
        logger.error(f"❌ Critical environment variable {var} is MISSING")

# Environment configurations
DYNAMICS_CONFIG = {
    "tenant_id": os.getenv("D365_TENANT_ID"),
    "client_id": os.getenv("D365_CLIENT_ID"),
    "client_secret": os.getenv("D365_CLIENT_SECRET"),
    "resource_url": os.getenv("D365_RESOURCE_URL"),
}
EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER"),
    "smtp_port": int(os.getenv("SMTP_PORT", 587)),
    "sender_email": os.getenv("SENDER_EMAIL"),
    "sender_password": os.getenv("SENDER_PASSWORD"),
    "receiver_email": os.getenv("RECEIVER_EMAIL"),
}
DYNAMICS_ENABLED = all(DYNAMICS_CONFIG.values())
if not DYNAMICS_ENABLED:
    logger.warning("⚠️ Dynamics 365 credentials missing. Dynamics features disabled.")

# Add near the environment configurations section
AGENT_PORTAL_CONFIG = {
    "url": os.getenv("AGENT_PORTAL_URL", "http://localhost:8000"),
    "enabled": os.getenv("AGENT_PORTAL_ENABLED", "false").lower() == "true",
}
if AGENT_PORTAL_CONFIG["enabled"]:
    logger.info(f"✅ Agent Portal integration enabled at {AGENT_PORTAL_CONFIG['url']}")
else:
    logger.warning(
        "⚠️ Agent Portal integration disabled. Set AGENT_PORTAL_ENABLED=true to enable."
    )

# Global variables
persistent_agent = None
persistent_client = None
vector_store = None
function_tool = None
# Hold the Azure AI Search Tool globally for direct querying
ai_search_tool: Optional[AzureAISearchTool] = None


# Retry logic for robustness
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
async def send_handoff_email(user_input: str, apex_id: str = "Unknown"):
    if not all(
        [
            EMAIL_CONFIG["smtp_server"],
            EMAIL_CONFIG["sender_email"],
            EMAIL_CONFIG["sender_password"],
            EMAIL_CONFIG["receiver_email"],
        ]
    ):
        logger.warning("⚠️ Email configuration incomplete. Skipping email handoff.")
        return False
    try:
        msg = MIMEText(
            f"A user requested human assistance.\nApex ID: {apex_id}\nMessage: {user_input}"
        )
        msg["Subject"] = "Human Handoff Request from APEX AI Assistant"
        msg["From"] = EMAIL_CONFIG["sender_email"]
        msg["To"] = EMAIL_CONFIG["receiver_email"]
        with smtplib.SMTP(
            EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]
        ) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.sendmail(
                EMAIL_CONFIG["sender_email"],
                EMAIL_CONFIG["receiver_email"],
                msg.as_string(),
            )
        logger.info("✅ Handoff email sent successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error sending handoff email: {e}")
        return False


def normalize_apex_id(apex_id: str) -> str:
    return apex_id.upper()


def generate_sas_url(blob_url: str) -> Optional[str]:
    """Generate a read‑only SAS URL for the given blob using BlobClient.
    
    This approach follows the official Azure SDK example pattern, which correctly
    handles URL encoding and SAS token generation together.

    Supports two authentication modes:
    1. Connection‑string with account key (local development / legacy).
    2. Managed Identity or any Azure AD identity (container deployment).
    """
    logger.info(f"Generating SAS URL for: {blob_url}")
    
    try:
        # Get connection string and check auth mode
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        if connection_string:
            # Connection string approach - simpler and more reliable
            # Create a blob service client from the connection string
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Extract account key from connection string or env var
            account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
            if not account_key:
                try:
                    import re as _re
                    match = _re.search(r"AccountKey=([^;]+)", connection_string, _re.I)
                    if match:
                        account_key = match.group(1)
                except Exception as parse_err:
                    logger.debug(f"Could not parse AccountKey from connection string: {parse_err}")
            
            if not account_key and hasattr(blob_service_client, "credential"):
                account_key = getattr(blob_service_client.credential, "key", None)
                
            if not account_key:
                msg = "ERROR: storage account key missing - provide via AZURE_STORAGE_ACCOUNT_KEY or in connection string"
                logger.error(msg)
                return msg
                
            # Create a blob client from the URL
            blob_client = BlobClient.from_blob_url(blob_url, credential=None)
            logger.info(f"Created blob client for: account={blob_client.account_name}, container={blob_client.container_name}, blob={blob_client.blob_name}")
            
            # Generate SAS token using the blob client properties
            start_time = datetime.now(timezone.utc)
            expiry_time = start_time + timedelta(hours=2)
            
            # Create permission with read access
            permission = BlobSasPermissions(read=True)
            
            # Generate SAS token with content disposition set to inline
            # This ensures the PDF displays in browser rather than downloading
            content_disposition = "inline"
            
            sas_token = generate_blob_sas(
                account_name=blob_client.account_name,
                container_name=blob_client.container_name,
                blob_name=blob_client.blob_name,
                account_key=account_key,
                permission=permission,
                expiry=expiry_time,
                start=start_time,
                content_disposition=f"inline; filename=\"{os.path.basename(blob_client.blob_name)}\"",
                content_type="application/pdf"
            )
            
            # Construct the SAS URL correctly
            sas_url = f"{blob_client.url}?{sas_token}"
            logger.info(f"Generated SAS URL (first 100 chars): {sas_url[:100]}...")
            
            # Per SDK update, ensure we return a string
            if not isinstance(sas_url, str):
                sas_url = str(sas_url)
                
            return sas_url
            
        else:
            # Managed Identity approach
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as ie:
                msg = f"ERROR: azure-identity is required for managed identity SAS generation ({ie})"
                logger.error(msg)
                return msg
                
            # Create a blob client from the URL
            blob_client = BlobClient.from_blob_url(blob_url, credential=None)
            logger.info(f"Created blob client for: account={blob_client.account_name}, container={blob_client.container_name}, blob={blob_client.blob_name}")
            
            # Create BlobServiceClient with DefaultAzureCredential
            credential = DefaultAzureCredential(exclude_shared_token_cache_credential=False)
            blob_service_client = BlobServiceClient(
                f"https://{blob_client.account_name}.blob.core.windows.net",
                credential=credential
            )
            
            # Get user delegation key
            user_delegation_key = blob_service_client.get_user_delegation_key(
                key_start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
                key_expiry_time=datetime.now(timezone.utc) + timedelta(hours=2)
            )
            
            # Generate SAS token with content disposition set to inline
            # This ensures the PDF displays in browser rather than downloading
            sas_token = generate_blob_sas(
                account_name=blob_client.account_name,
                container_name=blob_client.container_name,
                blob_name=blob_client.blob_name,
                user_delegation_key=user_delegation_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=2),
                content_disposition=f"inline; filename=\"{os.path.basename(blob_client.blob_name)}\"",
                content_type="application/pdf"
            )
            
            # Construct the SAS URL correctly
            sas_url = f"{blob_client.url}?{sas_token}"
            logger.info(f"Generated SAS URL with managed identity (first 100 chars): {sas_url[:100]}...")
            return sas_url
            
    except Exception as e:
        error_msg = f"SAS generation failed: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return f"ERROR: {error_msg}"


# ---- Execute startup diagnostics now that generate_sas_url exists ----

def _run_startup_diagnostics():
    storage_cs = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    logger.info(f"Storage connection string detected: length={len(storage_cs)}")

    test_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    if not test_account:
        logger.warning("AZURE_STORAGE_ACCOUNT_NAME not set – skipping SAS self-test")
        return

    sample_blob = (
        f"https://{test_account}.blob.core.windows.net/lucyrag/DIAG_PING.pdf"
    )
    try:
        _sas_test = generate_sas_url(sample_blob)
    except Exception as diag_err:
        logger.error(f"SAS self-test raised exception: {diag_err}")
        _sas_test = None

    if _sas_test and not str(_sas_test).startswith("ERROR"):
        logger.info("✅ SAS self-test succeeded – credentials look good")
    else:
        logger.error(f"❌ SAS self-test FAILED: {_sas_test}")


# Run immediately at import time
_run_startup_diagnostics()


@retry(
    retry=retry_if_exception_type(ServiceResponseError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def extract_pdf_url(search_result: Dict) -> Optional[str]:
    logger.info("🔍 Extracting PDF URL from search result")
    blob_url_candidate: Optional[str] = None

    # New logic: try metadata_storage_path first
    if (
        "metadata_storage_path" in search_result
        and search_result["metadata_storage_path"]
    ):
        storage_path = search_result["metadata_storage_path"]
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        
        # Preserve the full path to the blob (including folder structure)
        # First make sure path is properly formatted
        storage_path = storage_path.lstrip("/")
        
        # Check if storage path already contains the container name
        if storage_path.lower().startswith("lucyrag/"):
            # Container name is already in the path
            blob_url_candidate = f"https://{storage_account}.blob.core.windows.net/{storage_path}"
        else:
            # Add container name as prefix
            container_name = "lucyrag"
            blob_url_candidate = (
                f"https://{storage_account}.blob.core.windows.net/"
                f"{container_name}/{storage_path}"
            )
            
        # Log the constructed URL for debugging
        logger.info(f"Constructed blob URL: {blob_url_candidate}")

    # Fallback: decode parent_id which is base64‑encoded full blob URL
    elif "parent_id" in search_result and search_result["parent_id"]:
        try:
            padded = search_result["parent_id"] + "=="  # ensure correct padding
            decoded = base64.urlsafe_b64decode(padded).decode("utf-8", "ignore").strip()
            if decoded.lower().startswith("http") and decoded.lower().endswith(".pdf"):
                blob_url_candidate = decoded
        except Exception as decode_err:
            logger.debug(f"parent_id base64 decode failed: {decode_err}")

    if blob_url_candidate:
        sas_url = generate_sas_url(blob_url_candidate)
        if sas_url and not sas_url.startswith("ERROR"):
            logger.info(f"✅ Generated SAS URL: {sas_url[:50]}...")
            return sas_url
        else:
            logger.error(f"❌ Failed to generate SAS token: {sas_url}")
            return sas_url  # still return string so FunctionTool is valid
    # blob_url_candidate was None.
    logger.warning("❌ No valid PDF URL found in search result")
    return None


@retry(
    retry=retry_if_exception_type(ServiceResponseError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def initialize_persistent_agent():
    global persistent_agent, persistent_client, vector_store
    # Always reset persistent_agent to reload tool definitions and ensure updated functions
    persistent_agent = None
    logger.debug("[DEBUG] Starting initialize_persistent_agent()...")
    # Initialize agent if not already present (or after reset)
    if persistent_agent is None:
        try:
            logger.info("Initializing AI Project client and agent...")
            conn_str = os.getenv("AZURE_PROJECT_CONNSTRING")
            if not conn_str:
                logger.error("❌ AZURE_PROJECT_CONNSTRING not set")
                raise ValueError("AZURE_PROJECT_CONNSTRING not set")
            
            logger.debug("[DEBUG] Checking if running in container...")
            is_container = os.getenv("WEBSITE_SITE_NAME") or os.getenv("WEBSITES_PORT")
            if is_container:
                logger.debug("[DEBUG] Setting up managed identity credential...")
                from azure.identity import (
                    ManagedIdentityCredential,
                    ChainedTokenCredential,
                    EnvironmentCredential,
                )

                logger.info("Container environment detected. Using managed identity.")
                credential = ChainedTokenCredential(
                    ManagedIdentityCredential(
                        client_id=os.getenv("MANAGED_IDENTITY_CLIENT_ID")
                    ),
                    EnvironmentCredential(),
                )
            else:
                logger.info("Local environment detected. Using Azure CLI credentials.")
                logger.debug("[DEBUG] Clearing any conflicting environment variables...")
                for var in [
                    "AZURE_CLIENT_ID",
                    "AZURE_CLIENT_SECRET",
                    "AZURE_TENANT_ID",
                ]:
                    if var in os.environ:
                        logger.debug(f"Clearing {var} to avoid conflicts")
                        del os.environ[var]
                try:
                    import subprocess
                    logger.debug("[DEBUG] Checking Azure CLI login status...")
                    cli_user = (
                        subprocess.check_output(
                            [
                                "az",
                                "ad",
                                "signed-in-user",
                                "show",
                                "--query",
                                "objectId",
                                "-o",
                                "tsv",
                            ]
                        )
                        .decode()
                        .strip()
                    )
                    logger.info(f"Azure CLI user object ID: {cli_user}")
                except Exception as e:
                    logger.warning(f"Could not retrieve CLI user ID: {str(e)}")
                
                logger.debug("[DEBUG] Creating DefaultAzureCredential...")
                credential = DefaultAzureCredential(
                    exclude_environment_credential=False,
                    exclude_managed_identity_credential=False,
                    exclude_interactive_browser_credential=False,
                    exclude_visual_studio_code_credential=False,
                    exclude_shared_token_cache_credential=False,
                    exclude_cli_credential=False,
                )
            try:
                logger.debug("[DEBUG] Creating AIProjectClient...")
                persistent_client = AIProjectClient.from_connection_string(
                    credential=credential, conn_str=conn_str
                )
                logger.info("✅ AI Project client initialized")

                # Cleanup lingering agents to avoid stale FunctionTool registrations
                try:
                    logger.debug("[DEBUG] Cleaning up lingering agents...")
                    listing = persistent_client.agents.list_agents()
                    old_agents = getattr(listing, "data", listing)
                    for old in old_agents:
                        agent_id = (
                            old if isinstance(old, str) else getattr(old, "id", None)
                        )
                        if not agent_id:
                            continue
                        try:
                            persistent_client.agents.delete_agent(agent_id)
                            logger.info(f"🧹 Deleted old agent: {agent_id}")
                        except Exception as ie:
                            logger.warning(f"Could not delete agent {agent_id}: {ie}")
                except Exception as ce:
                    logger.error(f"Failed to cleanup old agents: {ce}")
            except Exception as e:
                logger.error(
                    f"❌ Failed to initialize AI Project client: {str(e)}",
                    exc_info=True,
                )
                if "does not have permissions" in str(e).lower():
                    logger.error(
                        "Permission error: Ensure your Azure CLI user has 'Contributor' role"
                    )
                if "interactionrequired" in str(e).lower() or "token" in str(e).lower():
                    logger.error(
                        "Authentication error: Run 'az login' to refresh your CLI session."
                    )
                raise
                
            logger.debug("[DEBUG] Creating vector store...")
            vector_store = persistent_client.agents.create_vector_store_and_poll(
                file_ids=[], name="class_action_notices"
            )
            logger.info(f"✅ Created vector store, ID: {vector_store.id}")

            # --- Azure AI Search Tool ---
            logger.debug("[DEBUG] Setting up Azure AI Search Tool...")
            search_conn_name = os.getenv("AI_SEARCH_CONNECTION_NAME")
            if not search_conn_name:
                logger.error("❌ AI_SEARCH_CONNECTION_NAME not set")
                raise ValueError("AI_SEARCH_CONNECTION_NAME not set")
                
            logger.debug(f"[DEBUG] Getting connection: {search_conn_name}")
            connection = persistent_client.connections.get(
                connection_name=search_conn_name
            )
            if not connection:
                logger.error(
                    f"❌ Could not find Azure AI Search connection with name: {search_conn_name}"
                )
                raise ValueError(
                    f"Could not find Azure AI Search connection with name: {search_conn_name}"
                )
            search_conn_id = connection.id
            logger.info(
                f"Using Azure AI Search connection: {search_conn_name} (ID: {search_conn_id})"
            )
            search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
            if not search_index_name:
                logger.error("❌ AZURE_SEARCH_INDEX_NAME not set")
                raise ValueError("AZURE_SEARCH_INDEX_NAME not set")
            
            logger.debug("[DEBUG] Creating ai_search_tool...")
            global ai_search_tool
            ai_search_tool = AzureAISearchTool(
                index_connection_id=search_conn_id,
                index_name=search_index_name,
                query_type="vector_semantic_hybrid",
                top_k=3,
                filter="",
            )

            # --- File Search Tool ---
            logger.debug("[DEBUG] Creating file_search_tool...")
            file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])

            # --- Function Tool ---
            logger.debug("[DEBUG] Setting up Function Tool...")
            # Always add Dynamics functions, so tool names are
            # registered even if credentials are missing.
            all_functions = setup_dynamics_functions() + [
                generate_sas_url,
                render_pdf,
                get_current_datetime,
                execute_search_tool,  # sync wrapper so FunctionTool gets a JSON-serialisable value
                extract_text_from_pdf_tool,  # Add PDF text extraction function (sync wrapper)
            ]
            function_tool = FunctionTool(functions=all_functions)

            # --- ToolSet (April 2025 best practice) ---
            logger.debug("[DEBUG] Creating toolset...")
            toolset = ToolSet()
            toolset.add(ai_search_tool)
            toolset.add(file_search_tool)
            toolset.add(function_tool)

            # Enable automatic execution of FunctionTools so that calls like
            # query_entity_sync are actually run during create_and_process_run.
            try:
                logger.debug("[DEBUG] Enabling auto function calls...")
                # The max_retry parameter isn't supported in this version
                persistent_client.agents.enable_auto_function_calls(
                    toolset=toolset
                )
                logger.info("✅ Auto function call handling enabled for agent runs")
            except Exception as auto_exc:
                logger.error(f"Could not enable auto function calls: {auto_exc}")

            # --- System prompt: always load and log ---
            logger.debug("[DEBUG] Loading system prompt...")
            instructions = load_system_prompt()
            logger.info(
                f"Loaded system prompt (first 200 chars): {instructions[:200]}..."
            )

            # --- Create Agent (latest pattern, toolset only) ---
            logger.debug("[DEBUG] Creating persistent agent...")
            persistent_agent = persistent_client.agents.create_agent(
                model=os.getenv("AZURE_GPT_MODEL", "gpt-4o"),
                name="apex-ai-assistant",
                instructions=instructions,
                toolset=toolset,
            )
            if persistent_agent is None:
                logger.error("❌ Failed to create persistent agent")
                raise Exception("Failed to create persistent agent")
            logger.info(f"✅ Persistent agent created with ID: {persistent_agent.id}")
            return True
        except Exception as e:
            logger.error(
                f"❌ Error initializing persistent agent: {str(e)}", exc_info=True
            )
            raise


def construct_search_query(user_data: Dict) -> str:
    """Construct a search query string based on user data.

    Args:
        user_data: Dictionary containing user information like address and apex_id

    Returns:
        A formatted search query string for Azure AI Search
    """
    # ⓵ If the caller already supplied an explicit *search_query* string, use it as-is.
    #    This path is used by higher-level helpers (e.g. *build_member_queries*) that
    #    craft a complete query expression.  Avoid rebuilding it here – just return.
    override_query = user_data.get("search_query")
    if isinstance(override_query, str) and override_query.strip():
        logger.info(f"🔍 Using caller-supplied search_query override: '{override_query.strip()}'")
        return override_query.strip()

    # ⓶ Otherwise, fall back to constructing a query from individual profile fields.

    address = user_data.get("address", "").strip()
    apex_id = user_data.get("apex_id", "").strip().upper()
    query_parts = []

    try:
        # Handle APEX ID with proper formatting
        if apex_id:
            # Remove any special characters that might cause search issues
            apex_id = "".join(c for c in apex_id if c.isalnum() or c.isspace())
            # Use the exact blob filename pattern (APEXID.pdf) for higher precision
            query_parts.append(f'"{apex_id}.pdf"')  # e.g. "FGUY001.pdf"

        # Handle address with proper formatting
        if address:
            # Clean address format to improve search matching
            clean_address = address.replace(",", " ").strip()
            clean_address = " ".join(clean_address.split())
            # Limit address length for search query
            if len(clean_address) > 100:
                clean_address = clean_address[:100]
            query_parts.append(f'"{clean_address}"')

        # If we have no specific search terms, return an empty string so the caller can
        # decide what to do (it will typically short-circuit and skip the search).
        if not query_parts:
            logger.info("🔍 No search criteria – skipping search query construction")
            return ""

        # Combine search terms with AND for more precise results
        query = " AND ".join(query_parts)
        logger.info(f"🔍 Constructed search query: '{query}'")
        return query
    except Exception as e:
        logger.error(f"Error constructing search query: {str(e)}", exc_info=True)
        # On any unexpected error, skip the search instead of issuing a broad query
        return ""


def construct_search_filter(_user_data: Dict) -> str:
    """Return an empty string so we skip `$filter` by default.

    A number of stock *metadata_* fields created by the Azure Blob indexer (for
    example ``metadata_storage_content_type``) are **not** marked *filterable* in
    many index templates.  When a field is missing that flag the service throws
    "is not a filterable field" errors and the query fails.

    If you decide to mark the field as filterable later, simply change this
    implementation to ``return "metadata_storage_content_type eq 'application/pdf'"``.
    """

    return ""


@retry(
    retry=retry_if_exception_type(ServiceResponseError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def execute_search(user_data: Dict, func_tool=None) -> List[Dict]:
    """Directly query Azure AI Search instead of relying on the Agent to call the tool."""
    global ai_search_tool
    logger.debug(f"[DEBUG] execute_search called with user_data: {user_data}")

    # --- Robust input handling ---------------------------------------------------
    # The FunctionTool may sometimes pass a *string* payload (e.g. when invoked
    # directly via the LLM) instead of the expected ``dict``.  Accept such input
    # gracefully by converting it into the canonical dictionary structure so the
    # rest of the function can operate without branching logic.

    if isinstance(user_data, str):
        try:
            # First try to parse JSON – the tool wrapper usually returns JSON text
            parsed = json.loads(user_data)
            if isinstance(parsed, dict):
                user_data = parsed  # type: ignore[assignment]
            else:
                # If the parsed JSON is a list or other type, wrap it as a query
                user_data = {"search_query": user_data}
        except json.JSONDecodeError:
            # Not JSON → treat the raw string as *search_query* override
            user_data = {"search_query": user_data}

    try:
        # Direct approach using SearchClient regardless of ai_search_tool initialization
        query = construct_search_query(user_data)
        # If we ended up with no query terms, skip the expensive call instead of
        # issuing a broad "empty" search which can return the entire corpus.
        if not query.strip():
            logger.info("🔍 No query terms available – skipping Azure Search call")
            return []

        filter_expr = construct_search_filter(user_data)
        logger.info(f"[AzureSearch] Direct query: {query} | filter: {filter_expr}")

        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

        if not endpoint or not api_key or not index_name:
            logger.error(
                f"Missing required search config: endpoint={bool(endpoint)}, api_key={bool(api_key)}, index_name={bool(index_name)}"
            )
            return []

        try:
            # Check if the endpoint is reachable before attempting the search
            try:
                # Simple test to see if hostname resolves
                import socket

                hostname = (
                    endpoint.replace("https://", "").replace("http://", "").rstrip("/")
                )
                if "/" in hostname:
                    hostname = hostname.split("/")[0]
                socket.gethostbyname(hostname)
                logger.info(f"Successfully resolved Azure Search hostname: {hostname}")
            except Exception as dns_error:
                logger.error(f"Cannot resolve Azure Search hostname: {str(dns_error)}")
                # Return empty results - no mock fallback
                return []

            # Log the connection details (without sensitive info)
            logger.info(
                f"Connecting to Azure Search at: {endpoint}, index: {index_name}"
            )

            sc = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )

            # Use try/except specifically for the search operation
            try:
                top_k = int(os.getenv("SEARCH_TOP_K", "5"))

                # Determine query type – allow disabling semantic if index lacks config
                query_type_env = os.getenv("SEARCH_QUERY_TYPE", "semantic").lower()

                # Include the full-text *chunk* field so Lucy can summarise notices.
                select_fields = [
                    "metadata_storage_path",
                    "metadata_storage_name",
                    "chunk",  # contains extracted notice text
                ]

                search_kwargs = dict(
                    search_mode="any",
                    search_text=query,
                    top=top_k,
                    select=select_fields,
                )

                if filter_expr:
                    search_kwargs["filter"] = filter_expr

                logger.info(
                    "[AzureSearch] Calling SearchClient.search | "
                    f"search_text='{query}', top={top_k}, query_type='semantic'"
                )

                try:
                    response = sc.search(**search_kwargs)
                except HttpResponseError as prop_err:
                    err_msg = str(prop_err)
                    logger.error(f"SearchClient.search failed: {err_msg}")

                    retry = False
                    # ① Missing semantic configuration → retry with simple query
                    if "semanticConfiguration" in err_msg:
                        logger.warning("Retrying search without semantic query_type")
                        search_kwargs.pop("query_type", None)
                        retry = True

                    # ② Filter field not marked filterable → drop filter and retry
                    if "is not a filterable field" in err_msg:
                        bad_filter = search_kwargs.pop("filter", None)
                        logger.warning(
                            f"Retrying search without unsupported filter: {bad_filter}"
                        )
                        retry = True

                    if retry:
                        response = sc.search(**search_kwargs)
                    else:
                        raise

                search_results = [doc for doc in response]

                if search_results:
                    logger.info(f"✅ Search returned {len(search_results)} results")
                    # Log some field names from the first result for debugging
                    if search_results and len(search_results) > 0:
                        logger.debug(
                            f"First result fields: {list(search_results[0].keys())}"
                        )
                else:
                    logger.warning("⚠️ No search results found via SearchClient")

                return search_results
            except Exception as search_error:
                logger.error(
                    f"❌ Search operation failed: {search_error}", exc_info=True
                )
                # We no longer attempt a broad fallback search. Instead, log and return
                # an empty result set so we don't accidentally retrieve a massive corpus
                # that can blow out token limits.
                return []

        except Exception as e:
            logger.error(f"❌ Azure SearchClient error: {e}", exc_info=True)
            logger.error(
                f"Search client parameters: endpoint={endpoint}, index_name={index_name}"
            )
        return []
    except Exception as e:
        logger.error(f"❌ Error executing search: {str(e)}", exc_info=True)
        return []


@cl.on_chat_start
async def on_chat_start():
    logger.info("[DEBUG] on_chat_start called")
    try:
        logger.info("Starting new chat session")
        logger.debug("[DEBUG] Checking persistent_agent status...")
        if persistent_agent is None:
            logger.warning("Persistent agent not initialized. Initializing now...")
            logger.debug("[DEBUG] Calling initialize_persistent_agent()...")
        await initialize_persistent_agent()
        logger.debug("[DEBUG] Initialize_persistent_agent completed")
        if persistent_agent is None:
            raise Exception("Failed to initialize persistent agent")
        logger.debug("[DEBUG] Creating thread...")
        thread = await _safe_create_thread()
        if thread is None:
            raise Exception("Failed to create thread")
        logger.debug("[DEBUG] Setting thread_id in user_session...")
        cl.user_session.set("thread_id", thread.id)
        logger.info(f"✅ User session thread created with ID: {thread.id}")
        logger.debug("[DEBUG] on_chat_start completing...")

    except Exception as e:
        logger.error(f"❌ Error in on_chat_start: {e}", exc_info=True)
        logger.info("Retrying agent initialization...")
        try:
            await initialize_persistent_agent()
            thread = await _safe_create_thread()
            cl.user_session.set("thread_id", thread.id)
            logger.info(f"✅ Retry successful, thread created with ID: {thread.id}")
        except Exception as retry_e:
            logger.error(f"❌ Retry failed: {retry_e}", exc_info=True)
        raise


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Explain My Notice",
            message="Could you explain my class action notice to me?",
            icon="/public/icons/notice.svg",
            css_class="notice-starter",
        ),
        cl.Starter(
            label="Case Status",
            message="What is the current status of my case?",
            icon="/public/icons/update.svg",
            css_class="status-starter",
        ),
        cl.Starter(
            label="Update Address",
            message="I need to update my address on file.",
            icon="/public/icons/payment.svg",
            css_class="address-starter",
        ),
        cl.Starter(
            label="Request Check Reissue",
            message="I need to request a reissue of my check.",
            icon="/public/icons/callback.svg",
            css_class="reissue-starter",
        ),
        cl.Starter(
            label="Am I Included In A Case",
            message="I want to know if I'm included in a class action case.",
            icon="/public/icons/callback.svg",
            css_class="included-starter",
        ),
    ]


async def extract_text_from_pdf(sas_url: str) -> str:
    """Extract text from a PDF using its SAS URL.
    
    Args:
        sas_url: The SAS URL to the PDF blob.
        
    Returns:
        The extracted text from the PDF, or an error message if extraction fails.
    """
    try:
        import PyPDF2
        import io
        import requests
        
        # Download PDF from SAS URL
        try:
            logger.info(f"Downloading PDF from SAS URL: {sas_url[:80]}...")
            response = requests.get(sas_url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # If we get a 403 error, the SAS token might have expired
            if e.response.status_code == 403 and 'blob_url' in sas_url:
                # Extract the blob URL and regenerate the SAS token
                blob_url = sas_url.split('?')[0]
                new_sas_url = generate_sas_url(blob_url)
                if new_sas_url and not new_sas_url.startswith("ERROR"):
                    logger.info("Successfully regenerated SAS URL, retrying download")
                    response = requests.get(new_sas_url, timeout=15)
                    response.raise_for_status()
                else:
                    raise Exception(f"Failed to regenerate SAS URL: {new_sas_url}")
        
        # Use PyPDF2 to extract text
        pdf_content = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_content)
        
        # Extract text from all pages
        full_text = ""
        total_pages = len(pdf_reader.pages)
        logger.info(f"Extracting text from PDF with {total_pages} pages")
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""
            full_text += page_text + "\n\n"
            
        # No character limit - return the entire content
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF ({total_pages} pages)")
        return full_text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}", exc_info=True)
        return f"ERROR: Could not extract text from PDF: {str(e)}"


async def find_notice_for_user(user_profile: dict, max_attempts: int = 5) -> dict:
    """Find a notice for the user based on their profile.
    
    Args:
        user_profile: A dictionary containing user profile information.
        max_attempts: Maximum number of attempts to search for a notice.
        
    Returns:
        A dictionary containing the found notice information.
    """
    # Log the user profile for debugging
    logger.info(f"Looking for notice for user profile: {json.dumps(user_profile, indent=2)}")
    
    def _notice_not_found(reason: str = ""):
        """Helper function to return a formatted 'not found' response"""
        message = "I couldn't find information about any class action settlements you may be eligible for."
        if reason:
            message += f" {reason}"
        return {
            "found": False,
            "content": message
        }
    
    # Build queries based on the profile
    queries = []
    
    # Add location-specific queries
    if "state" in user_profile:
        if "city" in user_profile:
            queries.append(f"class action settlement {user_profile['city']} {user_profile['state']}")
        queries.append(f"class action settlement {user_profile['state']}")
    
    # Add generic queries
    queries.append("class action settlement notice")
    
    # Try each query until we find a PDF
    for query in queries[:max_attempts]:
        logger.info(f"Searching for: {query}")
        
        # Execute search
        search_results = await search_notices(query)
        
        if not search_results:
            logger.warning(f"No search results found for query: {query}")
            continue
            
        results = search_results.get("results", [])
        logger.info(f"Found {len(results)} results for query: {query}")
        
        # Filter results to find PDFs and extract PDF URL
        pdf_docs = [r for r in results if r.get("metadata", {}).get("content_type", "").startswith("application/pdf")]
        
        if pdf_docs:
            # We found a PDF - use the first one
            pdf_doc = pdf_docs[0]
            blob_url = pdf_doc.get("metadata", {}).get("blob_url")
            
            if not blob_url:
                # This shouldn't typically happen, but as a fallback
                logger.warning("Found PDF but no blob URL")
                continue
                
            logger.info(f"Found PDF at {blob_url}")
            
            # Generate a SAS URL for the blob
            sas_url = generate_sas_url(blob_url)
            
            if sas_url.startswith("ERROR"):
                logger.error(f"Failed to generate SAS URL: {sas_url}")
                continue
                
            # Extract text from the PDF directly as the primary method
            pdf_text = await extract_text_from_pdf(sas_url)
            
            if pdf_text.startswith("ERROR"):
                logger.error(f"Failed to extract text from PDF: {pdf_text}")
                
                # Fallback to using the chunks as a backup method
                logger.info("Falling back to using search chunks")
                
                related_chunks = []
                for result in results:
                    if result.get("metadata", {}).get("blob_url") == blob_url:
                        content = result.get("content", "")
                        if content:
                            related_chunks.append(content)
                
                if related_chunks:
                    logger.info(f"Found {len(related_chunks)} related chunks")
                    return {
                        "found": True,
                        "content": "\n\n".join(related_chunks),
                        "blob_url": blob_url,
                        "sas_url": sas_url,
                        "source": "chunks"
                    }
                else:
                    continue  # Try next query if chunks also failed
            
            # Return the full PDF text that was extracted successfully
            return {
                "found": True,
                "content": pdf_text,
                "blob_url": blob_url,
                "sas_url": sas_url,
                "source": "pdf_direct"
            }
    
    # If we've exhausted our queries and found nothing
    return _notice_not_found("Please try again later or provide more information about your location or interests.")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Main handler for incoming user messages.
    """
    global function_tool, persistent_client, persistent_agent
    user_input = message.content

    # Test command for PDF extraction
    if user_input.lower().startswith("/find_notice"):
        try:
            thinking_msg = cl.Message(content="Searching for notices...", author="APEX AI Assistant")
            await thinking_msg.send()
            
            # Parse location from command if provided
            args = user_input.split(' ', 1)
            user_profile = {}
            
            if len(args) > 1:
                location_text = args[1].strip()
                # Simple parsing for state and city
                parts = location_text.split(',')
                if len(parts) >= 2:
                    user_profile["city"] = parts[0].strip()
                    user_profile["state"] = parts[1].strip()
                else:
                    user_profile["state"] = location_text
            
            logger.info(f"Searching for notice with profile: {user_profile}")
            
            # Find notice
            notice_result = await find_notice_for_user(user_profile)
            
            if notice_result["found"]:
                # Display PDF if available
                if "sas_url" in notice_result:
                    await send_pdf_notice(notice_result["sas_url"])
                
                # Update thinking message with summary
                source_info = f"(Source: {notice_result.get('source', 'unknown')})"
                if notice_result.get("source") == "pdf_direct":
                    thinking_msg.content = f"Found a notice that matches your criteria. I've displayed the PDF for you. {source_info}"
                else:
                    content_snippet = notice_result.get("content", "")[:500] + "..."
                    thinking_msg.content = f"Found notice information: {content_snippet} {source_info}"
            else:
                thinking_msg.content = notice_result.get("content", "No notices found.")
                
            await thinking_msg.update()
            return
        except Exception as e:
            logger.error(f"Error processing find_notice command: {e}")
            await cl.Message(content=f"Error processing your request: {str(e)}", author="System").send()
            return
    
    # Safety-net: ensure persistent_client/agent are initialised (on_chat_start may have failed)
    if persistent_client is None or persistent_agent is None:
        logger.warning("persistent_client not ready inside on_message – initialising lazily")
        try:
            await initialize_persistent_agent()
        except Exception as init_err:
            logger.error(f"Lazy init failed: {init_err}")
            await cl.Message(
                content="System error initialising backend services. Please try again later.",
                author="System",
            ).send()
            return

    # Get thread ID
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        logger.warning("No thread ID found. Creating new thread.")
        thread = await _safe_create_thread()
        cl.user_session.set("thread_id", thread.id)
        thread_id = thread.id
    
    # Check if we need to refresh time awareness for long sessions
    await refresh_time_awareness(thread_id)
    
    # Check if this thread is awaiting agent handoff
    awaiting_agent = cl.user_session.get("awaiting_agent", False)
    if awaiting_agent:
        # Check if an agent has joined
        agent_joined = await check_for_agent_presence(thread_id)

        if agent_joined:
            # Agent has joined, forward this message to the agent portal
            try:
                # Forward message to agent portal
                async with aiohttp.ClientSession() as session:
                    portal_url = f"{AGENT_PORTAL_CONFIG['url']}/api/conversations/{thread_id}/messages"
                    await session.post(
                        portal_url,
                        json={
                            "role": "user",
                            "content": user_input,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

                # Let the user know their message was sent to the human agent
                await cl.Message(
                    content="Your message has been sent to the human agent.",
                    author="System",
                ).send()

                # Don't process this message with the AI
                return
            except Exception as e:
                logger.error(
                    f"❌ Error forwarding message to agent: {str(e)}", exc_info=True
                )
                # Fall through to AI processing if forwarding fails
        else:
            # Agent hasn't joined yet, show a waiting message
            await cl.Message(
                content="A human agent has been notified and will join shortly. In the meantime, I'll continue to assist you.",
                author="System",
            ).send()
            # Continue processing with AI while waiting

    # Check for handoff keywords
    direct_handoff_request = any(
        keyword in user_input.lower()
        for keyword in [
            "speak to human",
            "talk to human",
            "agent",
            "representative",
            "person",
        ]
    )

    # Regular processing continues as normal
    thinking_msg = cl.Message(content="Thinking...", author="APEX AI Assistant")
    await thinking_msg.send()

    try:
        logger.info(f"Processing user input: {user_input[:50]}...")
        
        # Create user message
        persistent_client.agents.create_message(
            thread_id=thread_id, role="user", content=user_input
        )

        # Test search functionality before proceeding
        search_test_result = await test_rag_connectivity()
        if not search_test_result:
            logger.warning(
                "Search connectivity test failed. Service may be unavailable."
            )

        # Let the agent reason about the message and use tools as needed
        try:
            run = persistent_client.agents.create_and_process_run(
                thread_id=thread_id, agent_id=persistent_agent.id
            )
        except Exception as run_error:
            logger.error(f"Error creating agent run: {str(run_error)}", exc_info=True)
            thinking_msg.content = "I'm having trouble processing your request due to a system error. Please try again in a moment."
            await thinking_msg.update()
            return

        run_attempts = 0
        max_attempts = 3

        while (
            run.status in ["queued", "in_progress", "requires_action"]
            and run_attempts < max_attempts
        ):
            await asyncio.sleep(1)
            run_attempts += 1
            try:
                run = persistent_client.agents.retrieve_run(
                    thread_id=thread_id, run_id=run.id
                )
            except Exception as retrieve_error:
                logger.error(f"Error retrieving run: {str(retrieve_error)}")
                if run_attempts >= max_attempts:
                    break

        if run.status == "failed":
            error_details = getattr(run, "last_error", "No error info")
            logger.error(f"Agent run failed: {error_details}")

            # Handle search tool errors specifically
            if "search_tool_server_error" in str(error_details):
                logger.error(
                    "Search tool error detected. Responding with error message."
                )
                thinking_msg.content = (
                    "I'm having trouble accessing our document database right now. "
                    "Please try again later or contact support for assistance with your request."
                )
            else:
                thinking_msg.content = "Sorry, I couldn't generate a response due to a system error. Please try again."

            await thinking_msg.update()
            return

        # Get the latest agent message
        messages = persistent_client.agents.list_messages(thread_id=thread_id)
        logger.info(
            f"All agent messages: {[{'role': m.role, 'content': m.content[:100] + '...' if isinstance(m.content, str) and len(m.content) > 100 else m.content} for m in messages.data]}"
        )
        response = None
        
        # Get the tool outputs from the run
        tool_outputs = []
        if hasattr(run, "tool_calls"):
            for call in run.tool_calls:
                if hasattr(call, "output") and call.output:
                    # Ensure all tool outputs are strings for SDK compatibility
                    output = call.output
                    if not isinstance(output, str):
                        try:
                            # Convert non-string outputs to JSON strings
                            output = json.dumps(output)
                            logger.info(f"Converted non-string tool output to JSON string")
                        except Exception as e:
                            # Fallback to string representation if JSON fails
                            output = str(output)
                            logger.warning(f"Failed to JSON-serialize tool output: {e}")
                    
                    tool_outputs.append(output)
                    logger.info(f"Found tool output: {output[:100]}...")
        
        # First look for PDF markers in tool outputs
        pdf_markers = []
        for output in tool_outputs:
            # If the tool already returned markers, capture them
            if isinstance(output, str) and "PDF_RENDER_MARKER_BEGIN" in output:
                import re
                pdf_pattern = r"<<PDF_RENDER_MARKER_BEGIN:(.+):(side|inline|download):PDF_RENDER_MARKER_END>>"
                matches = re.findall(pdf_pattern, output)
                for match in matches:
                    url, display = match
                    url = url.strip()
                    pdf_markers.append((url, display))
                    logger.info(f"Found PDF marker in tool output: {url[:50]}...")
            # NEW: Detect raw search JSON and build markers
            if isinstance(output, str):
                auto_urls = _extract_pdf_sas_from_search_results(output)
                for sas_url in auto_urls:
                    url = sas_url.strip()
                    pdf_markers.append((sas_url, "side"))
                    # Append a marker to the assistant response so downstream logic renders it
                    if response is None:
                        response = ""
                    response += "\n\n" + render_pdf(sas_url)

        # Now get the response from agent messages
        for msg in messages.data:
            if msg.role in ["agent", "assistant", "system"]:
                # Try to extract text in a robust way
                if isinstance(msg.content, str):
                    response = msg.content
                    break
                elif isinstance(msg.content, list) and msg.content:
                    part = msg.content[0]
                    if hasattr(part, "text") and hasattr(part.text, "value"):
                        response = part.text.value
                        break
                    elif hasattr(part, "text"):
                        response = part.text
                        break
                    elif isinstance(part, str):
                        response = part
                        break
        if not response:
            response = "Sorry, I couldn't generate a response. Please try again."

        # --- Handle {{#tool}} blocks left un-executed (e.g., render_pdf calls) ---
        if response and "{{#tool" in response:
            tool_pattern = r"\{\{#tool[\s\S]*?\}\}([^\{]*\{[^}]*\})[^\{]*\{\{\/tool\}\}"  # capture inner JSON
            matches = re.findall(tool_pattern, response)
            for raw_json in matches:
                try:
                    data = json.loads(raw_json)
                    sas = data.get("sas_url") or data.get("url")
                    disp = data.get("display", "side")
                    if sas:
                        url = sas.strip()
                        pdf_markers.append((url, disp))
                except Exception:
                    pass
            # strip tool blocks from visible text
            response = re.sub(r"\{\{#tool[\s\S]*?\{\{\/tool\}\}", "", response).strip()

        # Check if the (cleaned) response also contains PDF markers
        if response and "PDF_RENDER_MARKER_BEGIN" in response:
            import re
            pdf_pattern = r"<<PDF_RENDER_MARKER_BEGIN:(.+):(side|inline|download):PDF_RENDER_MARKER_END>>"
            matches = re.findall(pdf_pattern, response)
            for match in matches:
                url, display = match
                url = url.strip()
                pdf_markers.append((url, display))
                logger.info(f"Found PDF marker in response: {url[:50]}...")
            
            # Clean the markers from the displayed response
            clean_response = re.sub(pdf_pattern, "", response)
            response = clean_response.strip()

        # --- Auto-detect direct PDF links in the assistant response ------------------
        if response:
            auto_urls_in_text = _extract_pdf_sas_from_search_results(response)
            for sas_url in auto_urls_in_text:
                if all(sas_url != u for u, _ in pdf_markers):  # avoid duplicates
                    url = sas_url.strip()
                    pdf_markers.append((url, "side"))
        # -----------------------------------------------------------------------------

        logger.info("✅ Response processed and sent to user")

        # Update the message with the cleaned response
        thinking_msg.content = response
        await thinking_msg.update()

        # Process each PDF marker (render PDFs)
        logger.info(f"Found {len(pdf_markers)} PDF markers to process")
        for url, display in pdf_markers:
            logger.info(f"Processing PDF marker: URL={url[:50]}..., display={display}")
            try:
                pdf_element = cl.Pdf(
                    name="Class Action Notice",
                    display=display,
                    url=url,
                    page=1
                )

                # Attach to current assistant message
                thinking_msg.elements.append(pdf_element)
                await thinking_msg.update()
                logger.info("✅ PDF element added to thinking_msg")

                # Also send in separate message for side preview
                # Make sure to include the exact PDF name in the message content to ensure proper linking
                pdf_msg = cl.Message(
                    content=f"Here is **Class Action Notice** for your convenience.",
                    author="Lucy",
                )
                await pdf_msg.send()
                await pdf_element.send(for_id=pdf_msg.id)
                logger.info("✅ PDF element sent in separate message")
            except Exception as e:
                logger.error(f"❌ Failed to render PDF element: {e}", exc_info=True)
                fallback_msg = (
                    "\n\nI tried to display the PDF, but there was an issue. "
                    f"You can still download it: [Download Notice]({url})"
                )
                thinking_msg.content += fallback_msg
                await thinking_msg.update()

        # All done processing response
    except Exception as e:
        logger.error(f"❌ Error in on_message: {str(e)}", exc_info=True)
        thinking_msg.content = (
            f"Sorry, an error occurred: {str(e)}. Please try again or contact support."
        )
        await thinking_msg.update()


def monitor_agent_run(client, thread_id: str, run_id: str) -> Dict[str, int]:
    """Monitor an agent run and track tool usage.
    
    Args:
        client: The AI Project client
        thread_id: ID of the thread
        run_id: ID of the run to monitor
        
    Returns:
        Dict tracking counts of different tool usages
    """
    tool_usage = {
        "azure_ai_search": 0,
        "file_search": 0,
        "code_interpreter": 0,
        "functions": 0,
    }
    try:
        run = client.agents.retrieve_run(thread_id=thread_id, run_id=run_id)
        if hasattr(run, "tool_calls"):
            for call in run.tool_calls:
                tool_type = call.get("type", "").lower()
                if "search" in tool_type:
                    tool_usage["azure_ai_search"] += 1
                elif "file" in tool_type:
                    tool_usage["file_search"] += 1
                elif "code" in tool_type:
                    tool_usage["code_interpreter"] += 1
                elif "function" in tool_type:
                    tool_usage["functions"] += 1
        logger.debug(f"Tool usage: {tool_usage}")
    except Exception as e:
        logger.error(f"❌ Error monitoring agent run: {str(e)}")
    return tool_usage


# --- PDF SAS URL and Chainlit Inline PDF Preview ---
async def send_pdf_notice(sas_url: str):
    """Send a PDF notice to the user with inline preview.

    The new Chainlit API (v0.8+) no longer supports the **file=** kwarg on
    ``cl.Message``.  Instead we:

    1. Send a normal text message.
    2. Attach a ``cl.Pdf`` element to that message via **for_id** so the PDF
       shows up in the side panel.
    """

    # 1. Send the container message - make sure to include the PDF name in the content
    msg = cl.Message(
        content="Here is your **Class Action Notice**. You can view it below or download it.",
        author="Lucy",
    )
    await msg.send()

    # 2. Attach the PDF element
    pdf_el = cl.Pdf(
        name="Class Action Notice",
        url=sas_url,
        display="side",
        page=1
    )
    await pdf_el.send(for_id=msg.id)
    logger.info("✅ PDF element sent in separate message")


# Register shutdown cleanup
def _cleanup():
    logger.info("Shutting down Azure SDK clients...")
    try:
        if persistent_client:
            persistent_client.close()
            logger.info("AIProjectClient closed successfully")
    except Exception as e:
        logger.error(f"Error closing AIProjectClient: {e}")


atexit.register(_cleanup)


# --- PDF Rendering Tool ---
def render_pdf(sas_url: str, *, display: str = "side") -> str:
    """Return a placeholder string that *marks* the PDF for in‑app preview.

    Chainlit Elements (``cl.Pdf``) must be sent **separately** from the text
    message.  We therefore return a lightweight marker that the
    ``on_message`` handler can detect *after* it receives the assistant
    response.  The handler will then:

    1. Strip the marker from the text so the user sees a clean sentence with a
       download link.
    2. Send the actual ``cl.Pdf`` element associated with the final message so
       the PDF appears in the requested *side* panel.

    The marker format is ``<<PDF_RENDER_MARKER_BEGIN:{url}:{display}:PDF_RENDER_MARKER_END>>``.
    """
    # Check input parameters and handle errors properly
    try:
        if not sas_url or not isinstance(sas_url, str):
            return json.dumps({"error": "Invalid SAS URL provided"})
            
        if display not in ["side", "inline", "page"]:
            display = "side"  # Default to side if invalid display mode
            
        logger.info(f"[Lucy] Embedding PDF marker for SAS URL: {sas_url[:80]}...")
        # Human‑readable sentence + marker the UI logic will replace with an element
        # Include the exact name "Class Action Notice" in the text to ensure proper linking
        marker = f"<<PDF_RENDER_MARKER_BEGIN:{sas_url}:{display}:PDF_RENDER_MARKER_END>>"
        result = (
            f"Here is your **Class Action Notice** – you can also download it directly: "
            f"[Download Notice]({sas_url})\n\n{marker}"
        )
        return result
    except Exception as e:
        logger.error(f"Error in render_pdf: {e}")
        return json.dumps({"error": str(e)})


async def send_pdf_directly(thinking_msg: cl.Message, url: str, display: str = "side"):
    """
    Directly append a PDF element to an existing message.

    Args:
        thinking_msg: The message to append to
        url: The PDF URL
        display: Display mode ('side', 'inline', or 'download')
    """
    try:
        pdf_element = cl.Pdf(
            name="Class Action Notice",
            display=display,
            url=url,
            page=1
        )
        # Append to message elements
        thinking_msg.elements.append(pdf_element)
        await thinking_msg.update()
        logger.info(f"✅ PDF element directly added to message")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to directly add PDF: {str(e)}", exc_info=True)
        return False


def get_current_datetime() -> str:
    """Return current date/time information as a JSON string.

    According to the Azure AI Projects SDK update, all tool outputs must be
    strings. This function returns a JSON-serialized string with date/time info.
    """
    now = datetime.now(timezone.utc)
    eastern = now.astimezone(timezone(timedelta(hours=-4)))  # Simplified EST/EDT
    data = {
        "iso": now.isoformat(),
        "utc": now.strftime("%A, %B %d, %Y at %I:%M %p UTC"),
        "eastern": eastern.strftime("%A, %B %d, %Y at %I:%M %p EST"),
    }
    # Convert the dictionary to a JSON string for tool output
    try:
        return json.dumps(data)
    except Exception as e:
        logger.error(f"Error serializing datetime data: {e}")
        return json.dumps({"error": str(e)})

# Load and enhance system prompt with current time
def load_system_prompt():
    """Load system prompt and enhance it with current datetime information."""
    with open("agent_instructions.txt", "r") as f:
        instructions = f.read()
    
    # Add current time context
    current_time_json = get_current_datetime()
    
    # Parse the JSON string back to a dictionary
    try:
        current_time = json.loads(current_time_json)
    except Exception as e:
        logger.error(f"Error parsing datetime JSON: {e}")
        # Fallback if parsing fails
        now = datetime.now(timezone.utc)
        current_time = {
            "utc": now.strftime("%A, %B %d, %Y at %I:%M %p UTC"),
            "eastern": now.astimezone(timezone(timedelta(hours=-4))).strftime("%A, %B %d, %Y at %I:%M %p EST")
        }
    
    time_context = f"\nCURRENT TIME: Today is {current_time['utc']} / {current_time['eastern']}.\n"
    time_context += f"Always consider this current date in your responses when discussing deadlines, timelines, or events.\n"
    
    # Add time context near the beginning of the prompt (after the first blank line)
    if "\n\n" in instructions:
        parts = instructions.split("\n\n", 1)
        enhanced_instructions = parts[0] + "\n" + time_context + "\n" + parts[1]
    else:
        enhanced_instructions = instructions + "\n" + time_context
        
    logger.info(f"Added current time to system prompt: {current_time['utc']}")
    return enhanced_instructions

async def refresh_time_awareness(thread_id: str):
    """Refresh Lucy's awareness of the current time during long sessions.
    
    Sends a system message with current date/time information if the session
    has been active for more than 30 minutes.
    """
    try:
        messages = persistent_client.agents.list_messages(thread_id=thread_id)
        if not messages or not hasattr(messages, "data") or len(messages.data) < 2:
            return False
            
        # Find the most recent system message with time information
        latest_time_msg = None
        for msg in messages.data:
            if msg.role == "assistant" and "CURRENT TIME:" in str(msg.content):
                latest_time_msg = msg
                break
                
        # If no time message found or message is old (30+ minutes), refresh
        current_time = datetime.now(timezone.utc)
        should_refresh = False
        
        if not latest_time_msg:
            should_refresh = True
        elif hasattr(latest_time_msg, "created_at"):
            # Calculate time difference
            msg_time = latest_time_msg.created_at
            if isinstance(msg_time, str):
                msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00'))
            time_diff = current_time - msg_time
            if time_diff.total_seconds() > 1800:  # 30 minutes
                should_refresh = True
                
        if should_refresh:
            time_info = get_current_datetime()
            time_message = f"SYSTEM UPDATE: The current time is now {time_info['utc']} / {time_info['eastern']}."
            
            # Send a new system message with updated time
            persistent_client.agents.create_message(
                thread_id=thread_id,
                role="assistant",
                content=time_message
            )
            logger.info(f"Refreshed time awareness: {time_message}")
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error refreshing time awareness: {str(e)}")
        return False

def _extract_pdf_sas_from_search_results(raw_output):
    """Parse tool output (JSON *or* plain text) and extract valid PDF SAS URLs.

    Supports three formats:
    1. List[dict] with *metadata_storage_name* + *metadata_storage_path* (classic).
    2. List[dict] with only *parent_id* holding base64-encoded blob URL.
    3. Raw string that already contains one or more ``https://...pdf?...`` links.
    """

    import re as _re
    import json as _json

    pdf_sas_urls: list[str] = []

    # --- Helper: normalise blob path → SAS url ----------------------------------
    def _build_sas_from_path(raw_path: str) -> str | None:
        raw_path = raw_path.lstrip("/")
        if not raw_path.lower().endswith(".pdf"):
            return None
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        if not storage_account:
            return None
        # Keep the full path structure including folders
        blob_url = f"https://{storage_account}.blob.core.windows.net/{raw_path}"
        logger.debug(f"Building SAS URL from path: {blob_url}")
        
        # Generate SAS URL with enhanced error handling
        sas_url = generate_sas_url(blob_url)
        if sas_url and not sas_url.startswith("ERROR"):
            logger.info(f"Successfully generated SAS URL for path with length: {len(sas_url)}")
            return sas_url
        else:
            logger.error(f"Failed to generate SAS URL for {blob_url}: {sas_url}")
            return None

    # ❶ Attempt JSON parse --------------------------------------------------------
    try:
        data = _json.loads(raw_output)
        docs: list = []
        if isinstance(data, list):
            docs = data
        elif isinstance(data, dict) and "data" in data:
            docs = data["data"]  # type: ignore[index]

        for doc in docs:
            if not isinstance(doc, dict):
                continue

            name = (doc.get("metadata_storage_name") or "").lower()
            path = (doc.get("metadata_storage_path") or "")

            # Case A: explicit name + path
            if name.endswith(".pdf") and path.lower().endswith(".pdf"):
                sas = _build_sas_from_path(path)
                if sas and sas not in pdf_sas_urls:
                    pdf_sas_urls.append(sas)
                continue

            # Case B: parent_id base64 containing blob URL
            parent_id = doc.get("parent_id")
            if parent_id and isinstance(parent_id, str):
                try:
                    import base64 as _b64
                    padded = parent_id + "=="  # ensure padding
                    decoded = _b64.urlsafe_b64decode(padded).decode("utf-8", "ignore").strip()
                    if decoded.lower().startswith("http") and decoded.lower().endswith(".pdf"):
                        # If decoded URL already has SAS token, keep as-is; else build one
                        sas_url = decoded if "?" in decoded else generate_sas_url(decoded)
                        if sas_url and sas_url not in pdf_sas_urls:
                            pdf_sas_urls.append(sas_url)
                except Exception:
                    pass
    except Exception:
        # Not JSON or malformed – fall through to regex extraction
        pass

    # ❷ Regex scan for direct SAS URLs in plain-text output ----------------------
    if not pdf_sas_urls:
        # Stop the match at common markdown/formatting delimiters (')', ']', '*', etc.) so we do
        # not accidentally include trailing punctuation which would corrupt the SAS token.
        url_pattern = r"https?://[\w\-.]+/[^\s\)\]\"'>]+\.pdf[^\s\)\]\"'>]*"
        for m in _re.findall(url_pattern, raw_output):
            # Strip trailing markdown characters that sometimes sneak in (e.g. ")**")
            cleaned = m.rstrip("*)]>")  # remove right-hand delimiters
            if cleaned not in pdf_sas_urls:
                pdf_sas_urls.append(cleaned)

    return pdf_sas_urls

# Prevent blank queries ------------------------------------------------------------
def _is_blank_search_input(data: Any) -> bool:
    if not data:
        return True
    if isinstance(data, dict):
        return not any(data.values())
    return False

# --- Safe thread creation helper ---
async def _safe_create_thread(max_retries: int = 2):
    """Create a new agent thread, re-initialising the persistent client if the
    connection was dropped due to idle timeout.

    Args:
        max_retries: how many times to attempt client re-init + thread creation.

    Returns:
        The created thread object.
    """

    global persistent_client

    for attempt in range(max_retries + 1):
        try:
            return persistent_client.agents.create_thread()
        except (ServiceResponseError, ClientAuthenticationError) as err:
            logging.warning(
                "Thread creation failed (attempt %s/%s): %s – refreshing client",
                attempt + 1,
                max_retries,
                err,
            )
            try:
                await initialize_persistent_agent()
            except Exception as init_err:  # pragma: no cover – best-effort
                logging.error("Re-initialisation failed: %s", init_err)
        except Exception:  # other unexpected errors propagate
            raise
    raise RuntimeError("Unable to create thread after retries")

# ---------------- Connectivity quick test (used during on_message) -----------------

async def test_rag_connectivity():
    """Ping Azure AI Search once to verify connectivity (lightweight)."""
    try:
        sample_query = {"search_query": "class action notice"}
        results = await execute_search(sample_query)
        return bool(results)
    except Exception:
        return False

# ---------------- Name & Query Helpers -----------------


def _pick_best_name(rec: dict) -> str:
    """Return the authoritative name string for a member record.

    Priority:
    1. If *new_fullname* exists **and** differs from first+last → use it.
    2. Else concatenate new_firstname + new_lastname.
    3. Else return whichever non-empty field is available.
    """

    full = (rec.get("new_fullname") or "").strip()
    first = (rec.get("new_firstname") or "").strip()
    last = (rec.get("new_lastname") or "").strip()

    simple = f"{first} {last}".strip()

    if full and simple and full.lower() != simple.lower():
        return full
    return simple or full


def build_member_queries(rec: dict) -> list[str]:
    """Generate ordered RAG search queries as requested.

    Desired order:
    1. FirstName AND LastName AND Address
    2. FullName  AND Address
    3. FirstName AND LastName AND ApexID
    4. FullName  AND ApexID

    Rules:
    • Only include FullName queries *if* FullName differs from First+Last (case-insensitive).
    • Skip any component that is blank.
    • Deduplicate identical queries.
    """

    first = (rec.get("new_firstname") or "").strip()
    last = (rec.get("new_lastname") or "").strip()
    full = (rec.get("new_fullname") or "").strip()
    address = (rec.get("address") or "").strip()
    apex_id = (rec.get("apex_id") or "").strip()

    # Full name considered *distinct* only if not equal to "first last" (case-insensitive)
    simple_full = f"{first} {last}".strip()
    has_distinct_full = bool(full) and (full.lower() != simple_full.lower())

    def _quoted(s: str) -> str:
        return f'"{s}"'

    queries: list[str] = []

    # 0️⃣ ApexID only – highest precision, try this first if available
    if apex_id:
        queries.append(_quoted(apex_id))

    # 1️⃣ First + Last + Address
    if first and last and address:
        queries.append(f"{_quoted(first)} AND {_quoted(last)} AND {_quoted(address)}")

    # 2️⃣ Full + Address (only if distinct full exists)
    if has_distinct_full and address:
        queries.append(f"{_quoted(full)} AND {_quoted(address)}")

    # 3️⃣ First + Last + ApexID
    if first and last and apex_id:
        queries.append(f"{_quoted(first)} AND {_quoted(last)} AND {_quoted(apex_id)}")

    # 4️⃣ Full + ApexID
    if has_distinct_full and apex_id:
        queries.append(f"{_quoted(full)} AND {_quoted(apex_id)}")

    # Ensure uniqueness while preserving order
    seen = set()
    ordered_unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            ordered_unique.append(q)

    return ordered_unique

# ----------------------- FunctionTool wrapper -----------------------
# The Azure AI Projects FunctionTool expects *synchronous* callables that return
# JSON-serialisable results.  ``execute_search`` is *asynchronous* (because it
# may call async helpers such as ``initialize_persistent_agent``).  Wrapping it
# here ensures the coroutine is awaited and a plain Python list is returned so
# the SDK can serialise the value.


def execute_search_tool(user_data: Any = None, *, func_tool=None) -> str:
    """Synchronous wrapper around :pyfunc:`execute_search` for FunctionTool.

    Azure AI Projects will call this from a worker thread with no active event
    loop.  We therefore create (or reuse) a loop, run the coroutine to
    completion, and return the result.

    Args:
        user_data: The same dictionary you would pass to ``execute_search``.
        func_tool: Unused placeholder – kept for signature compatibility when
            invoked via FunctionTool.

    Returns:
        String representation of search results (JSON string).
    """

    if _is_blank_search_input(user_data):
        # Do not trigger a broad "" search – return empty result set immediately.
        return "[]"

    if user_data is None:
        user_data = {}

    try:
        # If a loop is already running (e.g. inside Chainlit callbacks) we cannot
        # invoke ``asyncio.run``.  In that case create a *dedicated* loop and
        # execute the coroutine there.
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    try:
        if running_loop and running_loop.is_running():
            temp_loop = asyncio.new_event_loop()
            try:
                result = temp_loop.run_until_complete(
                    execute_search(user_data, func_tool=func_tool)
                )
            finally:
                temp_loop.close()
        else:
            result = asyncio.run(execute_search(user_data, func_tool=func_tool))

        # Reduce payload size: retain only minimal fields needed downstream
        minimal: List[Dict] = []
        for doc in result:
            if not isinstance(doc, dict):
                continue
            minimal.append(
                {
                    "metadata_storage_name": doc.get("metadata_storage_name"),
                    "metadata_storage_path": doc.get("metadata_storage_path"),
                    "parent_id": doc.get("parent_id"),
                }
            )

        # FunctionTool expects a string payload → return JSON text (trimmed)
        try:
            return json.dumps(minimal, default=str)
        except Exception as json_err:
            logger.error(f"Failed to serialize search results: {json_err}")
            # Fallback to simple string representation if JSON fails
            return str(minimal)
    except Exception as e:
        logger.error(f"Error in execute_search_tool: {e}")
        return json.dumps({"error": str(e)})

async def search_notices(query: str) -> dict:
    """
    Search for notices using Azure AI Search.
    
    Args:
        query: The search query string.
        
    Returns:
        A dictionary containing search results and metadata.
    """
    logger.info(f"Searching for notices with query: {query}")
    
    try:
        # Use the existing execute_search function
        search_results = await execute_search({"search_query": query})
        
        if not search_results:
            logger.warning(f"No search results found for query: {query}")
            return {"results": []}
        
        # Transform the results into a more usable format
        formatted_results = []
        for result in search_results:
            # Extract key information from search results
            metadata = {
                "blob_url": result.get("metadata_storage_path", ""),
                "file_name": result.get("metadata_storage_name", ""),
                "content_type": "application/pdf" if result.get("metadata_storage_name", "").lower().endswith(".pdf") else "text/plain"
            }
            
            formatted_result = {
                "metadata": metadata,
                "content": result.get("chunk", ""),
                "id": str(uuid.uuid4())  # Generate a unique ID for each result
            }
            
            formatted_results.append(formatted_result)
        
        logger.info(f"Transformed {len(formatted_results)} search results")
        return {"results": formatted_results}
    
    except Exception as e:
        logger.error(f"Error in search_notices: {str(e)}", exc_info=True)
        return {"results": [], "error": str(e)}

def extract_text_from_pdf_tool(sas_url: str, *, func_tool=None) -> str:
    """Synchronous wrapper around extract_text_from_pdf for FunctionTool.
    
    Azure AI Projects will call this from a worker thread with no active event
    loop. We therefore create (or reuse) a loop, run the coroutine to
    completion, and return the result.
    
    Args:
        sas_url: The SAS URL to the PDF blob.
        func_tool: Unused placeholder – kept for signature compatibility when
            invoked via FunctionTool.
    
    Returns:
        The extracted text from the PDF, or an error message if extraction fails.
    """
    if not sas_url or not isinstance(sas_url, str):
        return "ERROR: Invalid SAS URL provided"
        
    try:
        # If a loop is already running (e.g. inside Chainlit callbacks) we cannot
        # invoke ``asyncio.run``.  In that case create a *dedicated* loop and
        # execute the coroutine there.
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    try:
        if running_loop and running_loop.is_running():
            temp_loop = asyncio.new_event_loop()
            try:
                result = temp_loop.run_until_complete(
                    extract_text_from_pdf(sas_url)
                )
            finally:
                temp_loop.close()
        else:
            result = asyncio.run(extract_text_from_pdf(sas_url))
            
        # Ensure we're returning a string, not an object
        if result is None:
            return "No text extracted from PDF."
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf_tool: {e}")
        return f"ERROR: Failed to extract text from PDF: {str(e)}"

async def check_for_agent_presence(thread_id: str) -> bool:
    """
    Check if a human agent has joined this conversation thread.
    
    Args:
        thread_id: The ID of the conversation thread.
        
    Returns:
        True if a human agent has joined, False otherwise.
    """
    if not AGENT_PORTAL_CONFIG["enabled"]:
        return False  # If agent portal integration is disabled
    
    try:
        async with aiohttp.ClientSession() as session:
            portal_url = f"{AGENT_PORTAL_CONFIG['url']}/api/conversations/{thread_id}/status"
            async with session.get(portal_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("agent_joined", False)
                else:
                    logger.warning(f"Failed to check agent status: HTTP {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error checking for agent presence: {str(e)}", exc_info=True)
        return False
