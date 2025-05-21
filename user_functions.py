import os
import logging
import requests
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import asyncio
import functools

# Load environment variables
load_dotenv()

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith("app.log") for h in logger.handlers):
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

# Dynamics configuration
DYNAMICS_CONFIG = {
    "tenant_id": os.getenv("D365_TENANT_ID"),
    "client_id": os.getenv("D365_CLIENT_ID"),
    "client_secret": os.getenv("D365_CLIENT_SECRET"),
    "resource_url": os.getenv("D365_RESOURCE_URL")
}
DYNAMICS_ENABLED = all(DYNAMICS_CONFIG.values())
if not DYNAMICS_ENABLED:
    logger.warning("⚠️ Dynamics 365 credentials incomplete or missing")

# Entity name and field mapping to handle specific naming conventions
ENTITY_NAME_MAP = {
    "new_classactioncases": "incidents",  # Map to actual entity name
    "new_classactioncase": "incidents",   # Map to actual entity name
    # Add other mapped entities as needed
}

# Field name mapping based on entity
FIELD_NAME_MAP = {
    "incidents": {
        "new_classactioncaseid": "incidentid"  # Map field names for incidents entity
    }
    # Add mappings for other entities as needed
}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
async def get_access_token() -> str:
    if not DYNAMICS_ENABLED:
        raise Exception("Dynamics 365 credentials not configured.")
    try:
        token_endpoint = f"https://login.microsoftonline.com/{DYNAMICS_CONFIG['tenant_id']}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": DYNAMICS_CONFIG["client_id"],
            "client_secret": DYNAMICS_CONFIG["client_secret"],
            "resource": DYNAMICS_CONFIG["resource_url"]
        }
        logger.debug(f"Getting access token from: {token_endpoint}")
        response = requests.post(token_endpoint, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        response.raise_for_status()
        token = response.json()["access_token"]
        logger.debug("✅ Access token fetched successfully")
        return token
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to get Dynamics 365 access token: {str(e)}", exc_info=True)
        raise Exception(f"Unable to authenticate with Dynamics 365: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error getting access token: {str(e)}", exc_info=True)
        raise Exception(f"Unexpected error during authentication: {str(e)}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
async def query_entity(entity: str, filter_str: Optional[str] = None, select: Optional[str] = None) -> List[Dict[str, Any]]:
    # Check if we have a mapped name for this entity
    original_entity = entity
    mapped_entity = ENTITY_NAME_MAP.get(entity, entity)
    if mapped_entity != entity:
        logger.info(f"Mapped entity name from {entity} to {mapped_entity}")
        entity = mapped_entity
        
    # Map field names in filter string if necessary
    if filter_str and entity in FIELD_NAME_MAP:
        field_mappings = FIELD_NAME_MAP[entity]
        # For each field mapping, replace the field name in the filter string
        for original_field, mapped_field in field_mappings.items():
            if original_field in filter_str:
                new_filter = filter_str.replace(original_field, mapped_field)
                logger.info(f"Mapped field in filter: {filter_str} -> {new_filter}")
                filter_str = new_filter
        
    access_token = await get_access_token()
    try:
        url = f"{DYNAMICS_CONFIG['resource_url']}/api/data/v9.2/{entity}"
        params = []
        if filter_str:
            logger.debug(f"Using filter: {filter_str}")
            params.append(f"$filter={filter_str}")
        if select:
            logger.debug(f"Selecting fields: {select}")
            params.append(f"$select={select}")
        if params:
            url += "?" + "&".join(params)
        logger.info(f"Executing Dynamics 365 query: {url}")
        
        # Log full URL and API version for debugging
        logger.debug(f"Using API version: v9.2")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        result = response.json().get("value", [])
        logger.info(f"✅ Query for {entity} returned {len(result)} results")
        return result
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # Enhanced retry logic handling both plural and singular forms
            if not entity.endswith("s"):
                # Try pluralized version
                plural_entity = f"{entity}s"
                logger.info(f"Entity not found. Retrying with plural: {plural_entity}")
                return await query_entity(plural_entity, filter_str, select)
            elif entity.endswith("s"):
                # Try singular version (remove trailing 's')
                singular_entity = entity[:-1]
                logger.info(f"Entity not found. Retrying with singular: {singular_entity}")
                return await query_entity(singular_entity, filter_str, select)
        logger.error(f"❌ Network error querying Dynamics 365: {str(e)}", exc_info=True)
        raise Exception(f"Network error while querying Dynamics 365: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error querying Dynamics 365: {str(e)}", exc_info=True)
        raise Exception(f"Unexpected error while querying Dynamics 365: {str(e)}")

async def update_entity(entity: str, entity_id: str, data: Dict[str, Any]) -> bool:
    access_token = await get_access_token()
    try:
        url = f"{DYNAMICS_CONFIG['resource_url']}/api/data/v9.2/{entity}({entity_id})"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }
        response = requests.patch(url, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        logger.info(f"✅ Updated entity {entity} with ID {entity_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error updating entity {entity}: {str(e)}", exc_info=True)
        raise Exception(f"Failed to update entity: {str(e)}")

async def create_entity(entity: str, data: Dict[str, Any]) -> str:
    access_token = await get_access_token()
    try:
        url = f"{DYNAMICS_CONFIG['resource_url']}/api/data/v9.2/{entity}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }
        response = requests.post(url, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        entity_id = response.headers.get("OData-EntityId", "").split("(")[-1].rstrip(")")
        logger.info(f"✅ Created entity {entity} with ID {entity_id}")
        return entity_id
    except Exception as e:
        logger.error(f"❌ Error creating entity {entity}: {str(e)}", exc_info=True)
        raise Exception(f"Failed to create entity: {str(e)}")

async def delete_entity(entity: str, entity_id: str) -> bool:
    access_token = await get_access_token()
    try:
        url = f"{DYNAMICS_CONFIG['resource_url']}/api/data/v9.2/{entity}({entity_id})"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }
        response = requests.delete(url, headers=headers, timeout=15)
        response.raise_for_status()
        logger.info(f"✅ Deleted entity {entity} with ID {entity_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error deleting entity {entity}: {str(e)}", exc_info=True)
        raise Exception(f"Failed to delete entity: {str(e)}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
async def discover_entities(prefix: str = "") -> List[str]:
    """
    Discover available entities in Dynamics 365, optionally filtered by prefix.
    
    Args:
        prefix: Optional prefix to filter entity names (e.g., 'new_')
        
    Returns:
        List of entity names
    """
    access_token = await get_access_token()
    try:
        # Query the EntityDefinitions endpoint to get all entity metadata
        url = f"{DYNAMICS_CONFIG['resource_url']}/api/data/v9.2/EntityDefinitions?$select=LogicalName,DisplayName"
        logger.info(f"Querying entity definitions from: {url}")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }
        
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        entities_data = response.json().get("value", [])
        logger.info(f"Retrieved {len(entities_data)} entity definitions")
        
        # Extract logical names and filter by prefix if provided
        entity_names = []
        for entity in entities_data:
            logical_name = entity.get("LogicalName", "")
            if not prefix or logical_name.startswith(prefix):
                entity_names.append(logical_name)
        
        logger.info(f"Found {len(entity_names)} entities" + 
                   (f" with prefix '{prefix}'" if prefix else ""))
        return entity_names
    except Exception as e:
        logger.error(f"❌ Error discovering entities: {str(e)}", exc_info=True)
        raise Exception(f"Failed to discover entities: {str(e)}")


# Synchronous wrappers for FunctionTool compatibility
def get_access_token_sync():
    result = asyncio.run(get_access_token())
    return str(result)

def query_entity_sync(entity, filter_str=None, select=None):
    """
    Synchronous wrapper for ``query_entity`` that is safe to call from both
    blocking **and** asynchronous contexts.

    The OpenAI Python function‑calling runtime (used by the Azure AI Projects
    `FunctionTool`) invokes synchronous functions in a standard, potentially
    already‑running, asyncio event‑loop.  Calling ``asyncio.run`` in this
    scenario raises the runtime error "asyncio.run() cannot be called from a
    running event loop" and causes the tool invocation to fail.  To make the
    helper robust we:

    1. Keep the public, synchronous signature unchanged (the agent only sees a
       plain function).
    2. Detect whether an event loop is already running in the current thread.
    3. If **no** loop is running, we simply delegate to ``asyncio.run`` as
       before.
    4. If a loop *is* running, we apply `nest_asyncio` to allow re‑entrancy and
       then call ``asyncio.run``.  This avoids complex thread/loop juggling and
       works reliably inside frameworks such as Chainlit.
    """

    # Sanitize the `$select` list: never include the GUID field as it bloats the
    # response and is rarely useful for the agent.
    if select:
        # Split on commas, trim whitespace, and drop any empty segments to avoid
        # generating an invalid "$select" list such as "field1,field2," which
        # is rejected by the Dynamics 365 OData endpoint with a 400 error.
        fields = [
            f.strip()
            for f in select.split(',')
            if f.strip() and f.strip() != 'new_classmemberid'
        ]

        # Re‑assemble the cleaned list only if we still have at least one field
        # to request. If the list is empty we simply omit the $select clause
        # altogether which instructs the server to return the default columns.
        select = ','.join(fields) if fields else None

    try:
        # Fast‑path: we are *not* inside an event loop.
        result = asyncio.run(query_entity(entity, filter_str=filter_str, select=select))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" not in str(e):
            # An unrelated error – re‑raise.
            raise

        # Fallback‑path: an event loop is already running. Enable nested loops
        # via `nest_asyncio` and retry.
        import nest_asyncio

        nest_asyncio.apply()
        result = asyncio.run(query_entity(entity, filter_str=filter_str, select=select))

    # Trim result size for easier downstream processing.
    # Remove internal GUID fields that must never reach the LLM
    def _strip_guid_fields(item):
        if isinstance(item, dict):
            item.pop("new_classmemberid", None)
        return item

    if isinstance(result, list):
        result = [_strip_guid_fields(r) for r in result]
        if len(result) > 5:
            result = result[:5]
    elif isinstance(result, dict):
        result = _strip_guid_fields(result)

    import json as _json
    # Azure AI Projects FunctionTool expects a **string** output, not a raw
    # Python object. Convert to JSON so the assistant can inspect it with
    # tool_call.json_output in its reasoning step.
    try:
        return _json.dumps(result)
    except Exception:
        # Fallback: convert via str() to ensure we always meet the required
        # schema even for non‑serialisable objects.
        return str(result)


def update_entity_sync(entity, entity_id, data):
    """
    Update a Dynamics 365 entity record.

    Args:
        entity (str): The logical name of the entity (e.g., 'new_classmembers').
        entity_id (str): The unique identifier (GUID) of the record to update.
        data (dict): Dictionary of fields and values to update.
    Returns:
        str: 'True' if update succeeded, otherwise raises an exception.
    """
    result = asyncio.run(update_entity(entity, entity_id, data))
    return str(result)


def create_entity_sync(entity, data):
    """
    Create a new Dynamics 365 entity record.

    Args:
        entity (str): The logical name of the entity (e.g., 'new_classmembers').
        data (dict): Dictionary of fields and values for the new record.
    Returns:
        str: The unique identifier (GUID) of the created record.
    """
    result = asyncio.run(create_entity(entity, data))
    return str(result)


def delete_entity_sync(entity, entity_id):
    result = asyncio.run(delete_entity(entity, entity_id))
    return str(result)


def test_azure_search_query():
    """Test Azure Search connectivity and sample data."""
    import os, requests
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    headers = {"api-key": api_key}
    # List indexes
    r = requests.get(f"{endpoint}/indexes?api-version=2021-04-30-Preview", headers=headers)
    print("Indexes:", r.json())
    # Try a simple search
    r = requests.get(f"{endpoint}/indexes/{index_name}/docs?api-version=2021-04-30-Preview&search=*", headers=headers)
    print("Sample docs:", r.json())


def test_sas_url():
    """Test SAS URL generation and blob access."""
    from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
    from datetime import datetime, timedelta
    import os, requests
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "lucyrag"
    # Example blob name; update as needed
    blob_name = "Alight Solutions/Data Files/Alight Solutions Class List.xlsx"
    account_name = blob_service_client.account_name
    account_key = blob_service_client.credential.account_key
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1),
    )
    url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    print("SAS URL:", url)
    # Try to fetch the blob
    r = requests.get(url)
    print("Status:", r.status_code)
    print("Content (first 200 bytes):", r.content[:200])


async def get_entity_metadata(entity: str) -> dict:
    access_token = await get_access_token()
    url = f"{DYNAMICS_CONFIG['resource_url']}/api/data/v9.2/EntityDefinitions(LogicalName='{entity}')/Attributes"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_entity_metadata_sync(entity):
    """
    Retrieve metadata (field definitions) for a Dynamics 365 entity.

    Args:
        entity (str): The logical name of the entity (e.g., 'new_classmembers').
    Returns:
        str: JSON string of the entity's metadata (field definitions).
    """
    return json.dumps(asyncio.run(get_entity_metadata(entity)))


async def query_related_entity(primary_entity: str, primary_filter: str, related_entity: str, 
                              relationship_attr: str, select: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Query a related entity through a relationship attribute.
    
    Args:
        primary_entity: The main entity to query (e.g., 'new_classmembers')
        primary_filter: Filter for the primary entity
        related_entity: The related entity to fetch (e.g., 'new_memberdisbursements')
        relationship_attr: The attribute defining the relationship
        select: Fields to select from the related entity
    """
    # First query primary entity to get IDs
    primary_results = await query_entity(primary_entity, primary_filter, select="new_classmemberid")
    
    if not primary_results:
        logger.warning(f"No {primary_entity} found with filter: {primary_filter}")
        return []
    
    # Get IDs from primary results
    primary_ids = [r["new_classmemberid"] for r in primary_results if "new_classmemberid" in r]
    
    if not primary_ids:
        logger.warning(f"No IDs found in {primary_entity} results")
        return []
    
    # Build filter for related entity
    id_filters = [f"{relationship_attr} eq {id}" for id in primary_ids]
    related_filter = " or ".join(id_filters)
    
    # Query related entity
    related_results = await query_entity(related_entity, related_filter, select)
    
    return related_results


def query_related_entity_sync(primary_entity, primary_filter, related_entity, 
                            relationship_attr, select=None):
    """
    Synchronous wrapper for query_related_entity.
    
    Args:
        primary_entity: The main entity to query (e.g., 'new_classmembers')
        primary_filter: Filter for the primary entity
        related_entity: The related entity to fetch (e.g., 'new_memberdisbursements')
        relationship_attr: The attribute defining the relationship
        select: Fields to select from the related entity
        
    Returns:
        JSON string with related entity results
    """
    try:
        # Fast‑path: we are *not* inside an event loop.
        result = asyncio.run(query_related_entity(
            primary_entity, primary_filter, related_entity, relationship_attr, select
        ))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" not in str(e):
            # An unrelated error – re‑raise.
            raise

        # Fallback‑path: an event loop is already running. Enable nested loops
        # via `nest_asyncio` and retry.
        import nest_asyncio

        nest_asyncio.apply()
        result = asyncio.run(query_related_entity(
            primary_entity, primary_filter, related_entity, relationship_attr, select
        ))
    
    # Ensure we return JSON string to meet the function tool requirements
    import json as _json
    try:
        return _json.dumps(result)
    except Exception:
        return str(result)

async def discover_entity_relationships(entity: str) -> Dict[str, List[str]]:
    """
    Discover relationships between entities by analyzing metadata.
    
    Args:
        entity: The entity to discover relationships for
    
    Returns:
        Dictionary mapping relationship names to related entities
    """
    metadata = await get_entity_metadata(entity)
    relationships = {}
    
    # Parse metadata for OneToMany, ManyToOne relationships
    # This is a simplified version - actual implementation would depend on
    # Dynamics 365 metadata structure
    for attribute in metadata.get('value', []):
        if attribute.get('AttributeType') == 'Lookup':
            target_entity = attribute.get('Targets', [])
            if target_entity:
                relationships[attribute.get('LogicalName')] = target_entity
    
    return relationships


def discover_entity_relationships_sync(entity):
    """
    Synchronous wrapper for discover_entity_relationships.
    
    Args:
        entity: The entity to discover relationships for
        
    Returns:
        JSON string with relationship mapping
    """
    try:
        # Fast‑path: we are *not* inside an event loop.
        result = asyncio.run(discover_entity_relationships(entity))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" not in str(e):
            # An unrelated error – re‑raise.
            raise

        # Fallback‑path: an event loop is already running. Enable nested loops
        # via `nest_asyncio` and retry.
        import nest_asyncio

        nest_asyncio.apply()
        result = asyncio.run(discover_entity_relationships(entity))
    
    # Ensure we return JSON string to meet the function tool requirements
    import json as _json
    try:
        return _json.dumps(result)
    except Exception:
        return f"Error converting relationships to JSON: {str(result)}"


def setup_dynamics_functions():
    """
    Returns a list of Dynamics 365 sync functions for use as FunctionTool(s):
    - query_entity_sync: Query records.
    - update_entity_sync: Update a record.
    - create_entity_sync: Create a record.
    - get_entity_metadata_sync: Discover entity fields/metadata.
    - query_related_entity_sync: Query records across relationships.
    - discover_entity_relationships_sync: Discover entity relationships.
    - discover_entities_sync: List available entities.
    - execute_complex_query_sync: Execute a complex OData query.
    - check_human_availability_sync: Check if human agents are available.
    - send_handoff_notification_email_sync: Send email about handoff request.
    Returns:
        list: List of function objects.
    """
    return [
        query_entity_sync,
        update_entity_sync,
        create_entity_sync,
        delete_entity_sync,
        get_entity_metadata_sync,
        query_related_entity_sync,
        discover_entity_relationships_sync,
        discover_entities_sync,
        execute_complex_query_sync,
        check_human_availability_sync,
        send_handoff_notification_email_sync,
    ]


def construct_incident_filter(incident_id: str, case_name: str) -> str:
    """
    Construct a valid OData filter string for Dynamics 365 'incidents' entity.
    Args:
        incident_id (str): The GUID of the incident/case.
        case_name (str): The value for new_fullname (case name).
    Returns:
        str: OData filter string, e.g. \
            "incidentid eq <guid> and new_fullname eq 'case'"
    """
    # Ensure GUID is not quoted, but string values are single-quoted
    filter_str = (
        f"incidentid eq {incident_id} and new_fullname eq '{case_name}'"
    )
    logger.info(f"[OData] Constructed incident filter: {filter_str}")
    return filter_str

def debug_log_function(func):
    """Decorator to add detailed logging to Dynamics 365 functions"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        
        logger.debug(f"Calling {func.__name__}({signature})")
        
        try:
            result = await func(*args, **kwargs)
            
            # Log result summary without overwhelming logs
            if isinstance(result, list):
                result_summary = f"{len(result)} items"
                if result and len(result) > 0:
                    sample = result[0]
                    if isinstance(sample, dict):
                        keys = list(sample.keys())
                        result_summary += f", keys: {keys}"
            elif isinstance(result, dict):
                keys = list(result.keys())
                result_summary = f"dict with keys: {keys}"
            else:
                result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                
            logger.debug(f"{func.__name__} returned: {result_summary}")
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
            
    return wrapper

def build_odata_query(
    primary_entity: str,
    primary_filter: str = None,
    expand_relations: Optional[List[Dict[str, str]]] = None,
    select: Optional[str] = None,
    top: Optional[int] = None
) -> str:
    """
    Build a complex OData query with expansion of related entities.
    
    Args:
        primary_entity: Main entity to query
        primary_filter: Filter for the primary entity
        expand_relations: List of relations to expand, e.g. [{"relation": "new_related_entity", "select": "field1,field2"}]
        select: Fields to select from primary entity
        top: Maximum number of records to return
        
    Returns:
        OData query URL (without base URL)
    """
    url = f"/api/data/v9.2/{primary_entity}"
    params = []
    
    if primary_filter:
        params.append(f"$filter={primary_filter}")
    
    if select:
        params.append(f"$select={select}")
    
    if expand_relations:
        expand_parts = []
        for rel in expand_relations:
            relation = rel.get("relation")
            rel_select = rel.get("select")
            if relation:
                if rel_select:
                    expand_parts.append(f"{relation}($select={rel_select})")
                else:
                    expand_parts.append(relation)
        
        if expand_parts:
            params.append(f"$expand={','.join(expand_parts)}")
    
    if top is not None and top > 0:
        params.append(f"$top={top}")
    
    if params:
        url += "?" + "&".join(params)
    
    return url

@debug_log_function
async def execute_complex_query(odata_query: str) -> List[Dict[str, Any]]:
    """
    Execute a complex OData query against Dynamics 365.
    
    Args:
        odata_query: OData query URL (without base URL)
        
    Returns:
        Query results
    """
    access_token = await get_access_token()
    try:
        # Ensure we build a valid absolute URL regardless of how *odata_query* is formatted.
        base_url = DYNAMICS_CONFIG["resource_url"].rstrip("/")
        # If caller forgot leading slash, add one. Avoid duplicate slashes.
        odata_path = odata_query.lstrip("/")
        url = f"{base_url}/{odata_path}"
        logger.info(f"Executing complex OData query: {url}")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }
        
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        result = response.json().get("value", [])
        
        logger.info(f"Complex query returned {len(result)} results")
        return result
    except Exception as e:
        logger.error(f"Error executing complex OData query: {str(e)}", exc_info=True)
        raise

def execute_complex_query_sync(odata_query: str) -> str:
    """
    Synchronous wrapper for execute_complex_query.
    
    Args:
        odata_query: OData query URL (without base URL)
        
    Returns:
        JSON string with query results
    """
    try:
        # Fast‑path: we are *not* inside an event loop.
        result = asyncio.run(execute_complex_query(odata_query))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" not in str(e):
            # An unrelated error – re‑raise.
            raise

        # Fallback‑path: an event loop is already running. Enable nested loops
        # via `nest_asyncio` and retry.
        import nest_asyncio

        nest_asyncio.apply()
        result = asyncio.run(execute_complex_query(odata_query))
    
    # Trim result size for easier downstream processing
    if isinstance(result, list) and len(result) > 5:
        result = result[:5]
    
    # Ensure we return JSON string to meet the function tool requirements
    import json as _json
    try:
        return _json.dumps(result)
    except Exception as e:
        logger.error(f"Error converting complex query results to JSON: {str(e)}")
        return f"Error converting results to JSON: {str(e)}"

# Human Handoff Functions
async def check_human_availability(timeout=60):
    """
    Check if any human agents are available to assist.
    
    Args:
        timeout (int): How long to wait for a response (seconds)
        
    Returns:
        tuple: (is_available: bool, agent_name: str)
    """
    logger.info("Checking for available human agents")
    
    try:
        # First try Microsoft Teams notification if configured
        teams_webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
        if teams_webhook_url:
            return await check_teams_availability(teams_webhook_url, timeout)
        else:
            logger.info("Teams webhook URL not configured, skipping Teams notification")
            return False, None
    except Exception as e:
        logger.error(f"Error checking human availability: {str(e)}", exc_info=True)
        return False, None

@debug_log_function
async def check_teams_availability(webhook_url, timeout=60):
    """
    Check agent availability via Microsoft Teams webhook.
    
    Args:
        webhook_url (str): Microsoft Teams webhook URL
        timeout (int): Timeout in seconds
        
    Returns:
        tuple: (is_available: bool, agent_name: str)
    """
    try:
        import aiohttp
        import uuid
        import json
        
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Create Teams message with adaptive card
        message = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "body": [
                            {
                                "type": "TextBlock",
                                "size": "Medium",
                                "weight": "Bolder",
                                "text": "AI Agent Lucy Needs Assistance"
                            },
                            {
                                "type": "TextBlock",
                                "text": "Are you available for a live transfer? Class Member is waiting.",
                                "wrap": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.Submit",
                                "title": "Yes",
                                "data": {
                                    "response": "yes",
                                    "requestId": request_id
                                }
                            },
                            {
                                "type": "Action.Submit",
                                "title": "No",
                                "data": {
                                    "response": "no",
                                    "requestId": request_id
                                }
                            }
                        ],
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "version": "1.2"
                    }
                }
            ]
        }
        
        # Send the message to Teams
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status != 200:
                    logger.error(f"Failed to send Teams notification: {response.status}")
                    return False, None
                    
                response_data = await response.json()
                logger.debug(f"Teams notification sent: {response_data}")
        
        # Wait for a response (polling approach) - in production, would use a webhook for responses
        # For this implementation, we'll simulate a response after a short delay
        await asyncio.sleep(1)  # Simulate processing time
        
        # In a real implementation, we would check for responses to our specific requestId
        # For demo purposes, return success after a few seconds (to be replaced with real logic)
        is_available = False
        agent_name = None
        
        # Here you would implement actual Teams response checking
        # This is a placeholder that randomly determines if someone is available
        if os.getenv("DEMO_MODE", "false").lower() == "true":
            import random
            is_available = random.random() > 0.5  # 50% chance of availability
            agent_name = "Demo Agent" if is_available else None
            
            if is_available:
                logger.info(f"Agent {agent_name} is available via Teams")
            else:
                logger.info("No agents available via Teams")
                
        return is_available, agent_name
    
    except Exception as e:
        logger.error(f"Error checking Teams availability: {str(e)}", exc_info=True)
        return False, None

async def send_handoff_notification_email(conversation_id, user_info, message, is_callback=False):
    """
    Send an email notification about a handoff request.
    
    Args:
        conversation_id (str): Unique ID for the conversation
        user_info (dict): User information
        message (str): Message from the user
        is_callback (bool): Whether this is a callback request
        
    Returns:
        bool: Success status
    """
    try:
        # Get email configuration
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        recipient_email = "agentsupport@apexclassaction.com"
        
        if not all([smtp_server, smtp_port, sender_email, sender_password]):
            logger.error("Email configuration incomplete. Cannot send handoff notification.")
            return False
        
        # Create message
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg["Subject"] = f"{'Callback' if is_callback else 'Handoff'} Request: {conversation_id}"
        msg["From"] = sender_email
        msg["To"] = recipient_email
        
        # Prepare user info text
        user_info_text = "\n".join([f"{k}: {v}" for k, v in user_info.items() if v])
        
        # Create email body
        agent_portal_url = os.getenv("AGENT_PORTAL_URL", "http://localhost:8000")
        body = f"""
        A class member has requested {'a callback' if is_callback else 'human assistance'}.
        
        Conversation ID: {conversation_id}
        
        User Information:
        {user_info_text}
        
        Message:
        {message}
        
        {f'Click here to join the conversation: {agent_portal_url}/agent/conversation/{conversation_id}' if not is_callback else 'Please contact the user at your earliest convenience.'}
        """
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        import smtplib
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            
        logger.info(f"Handoff notification email sent for conversation {conversation_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending handoff notification email: {str(e)}", exc_info=True)
        return False

def check_human_availability_sync(timeout=60):
    """
    Synchronous wrapper for check_human_availability
    
    Args:
        timeout (int): How long to wait for a response (seconds)
        
    Returns:
        str: JSON string with result
    """
    try:
        is_available, agent_name = asyncio.run(check_human_availability(timeout))
        return json.dumps({
            "is_available": is_available,
            "agent_name": agent_name
        })
    except Exception as e:
        logger.error(f"Error in check_human_availability_sync: {str(e)}", exc_info=True)
        return json.dumps({
            "is_available": False,
            "agent_name": None,
            "error": str(e)
        })

def send_handoff_notification_email_sync(conversation_id, user_info, message, is_callback=False):
    """
    Synchronous wrapper for send_handoff_notification_email
    
    Args:
        conversation_id (str): Unique ID for the conversation
        user_info (dict): User information
        message (str): Message from the user
        is_callback (bool): Whether this is a callback request
        
    Returns:
        str: Success status as string
    """
    try:
        result = asyncio.run(send_handoff_notification_email(
            conversation_id, user_info, message, is_callback
        ))
        return str(result).lower()
    except Exception as e:
        logger.error(f"Error in send_handoff_notification_email_sync: {str(e)}", exc_info=True)
        return f"false"

# Function to discover entities is already defined at line 197

def discover_entities_sync(prefix=None):
    """
    Synchronous wrapper for discover_entities.
    
    Args:
        prefix: Optional prefix to filter entities (e.g., 'new_')
        
    Returns:
        JSON string with the list of entity names
    """
    try:
        # Fast‑path: we are *not* inside an event loop.
        result = asyncio.run(discover_entities(prefix))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" not in str(e):
            # An unrelated error – re‑raise.
            raise

        # Fallback‑path: an event loop is already running. Enable nested loops
        # via `nest_asyncio` and retry.
        import nest_asyncio

        nest_asyncio.apply()
        result = asyncio.run(discover_entities(prefix))
    
    # Ensure we return JSON string to meet the function tool requirements
    import json as _json
    try:
        return _json.dumps(result)
    except Exception:
        return f"Error converting entity list to JSON: {str(result)}"

def setup_handoff_functions():
    """
    Returns a list of human handoff functions for use as FunctionTool(s)
    
    Returns:
        list: List of function objects
    """
    return [
        check_human_availability_sync,
        send_handoff_notification_email_sync,
    ]
