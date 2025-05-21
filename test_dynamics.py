import asyncio
import logging
import os
import sys
import requests
from dotenv import load_dotenv
from user_functions import (
    get_access_token,
    query_entity,
    DYNAMICS_CONFIG
)

# Force immediate console output 
print("Starting test_dynamics.py script...")
print(f"Python version: {sys.version}")
sys.stdout.flush()

# Define a function to discover entities since it's not in user_functions.py
async def discover_entities(prefix=None):
    """
    Discover available entities in Dynamics 365.
    
    Args:
        prefix: Optional prefix to filter entities (e.g., 'new_' for
               custom entities)
        
    Returns:
        List of entity names
    """
    try:
        print(f"Fetching access token for entity discovery...")
        sys.stdout.flush()
        access_token = await get_access_token()
        
        # Query metadata to get entity list
        print(f"Querying entity definitions...")
        sys.stdout.flush()
        url = (f"{DYNAMICS_CONFIG['resource_url']}"
               "/api/data/v9.2/EntityDefinitions")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Extract entity names from response
        print(f"Processing entity response...")
        sys.stdout.flush()
        entities = []
        for entity in response.json().get("value", []):
            entity_name = entity.get("LogicalName", "")
            if prefix is None or entity_name.startswith(prefix):
                entities.append(entity_name)
        
        return entities
    except Exception as e:
        print(f"ERROR in discover_entities: {str(e)}")
        sys.stdout.flush()
        logging.error(f"Error discovering entities: {str(e)}")
        return []

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug handler to ensure output is visible
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# Print key environment info (no credentials)
def print_env_info():
    print("\n===== ENVIRONMENT INFO =====")
    # Check critical environment variables
    print(f"Dynamics Resource URL set: {bool(DYNAMICS_CONFIG.get('resource_url'))}")
    if DYNAMICS_CONFIG.get('resource_url'):
        # Log only the domain, not full URL with potential sensitive path info
        url_parts = DYNAMICS_CONFIG['resource_url'].split('/')
        if len(url_parts) >= 3:
            domain = url_parts[2]  # Get domain from URL
            print(f"Dynamics domain: {domain}")
    
    print(f"Tenant ID set: {bool(DYNAMICS_CONFIG.get('tenant_id'))}")
    print(f"Client ID set: {bool(DYNAMICS_CONFIG.get('client_id'))}")
    print(f"Client Secret set: {bool(DYNAMICS_CONFIG.get('client_secret'))}")
    
    # Additional env vars that might be helpful
    print(f"Python path: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print("===== END ENVIRONMENT INFO =====\n")
    sys.stdout.flush()

# Load environment variables
load_dotenv()
print_env_info()

async def test_connection():
    """Test the basic Dynamics 365 connection by getting an access token"""
    print("Testing Dynamics 365 connection...")
    sys.stdout.flush()
    try:
        token = await get_access_token()
        # Only log token length, never the token itself
        token_len = len(token) if token else 0
        print(f"✅ Successfully obtained access token (length: {token_len})")
        sys.stdout.flush()
        logger.info("✅ Successfully obtained access token")
        return True
    except Exception as e:
        print(f"❌ Failed to get access token: {str(e)}")
        sys.stdout.flush()
        logger.error(f"❌ Failed to get access token: {str(e)}")
        return False

async def test_entity_discovery():
    """Test entity discovery to see what entities are available"""
    print("Testing entity discovery...")
    sys.stdout.flush()
    try:
        # Get all entities
        all_entities = await discover_entities()
        print(f"✅ Found {len(all_entities)} entities total")
        sys.stdout.flush()
        logger.info(f"✅ Found {len(all_entities)} entities total")
        
        # Get only custom entities (usually prefixed with 'new_')
        custom_entities = await discover_entities(prefix="new_")
        print(f"✅ Found {len(custom_entities)} custom entities")
        sys.stdout.flush()
        logger.info(f"✅ Found {len(custom_entities)} custom entities")
        
        # Log the entities found
        print("Custom entities:")
        custom_entities_sorted = sorted(custom_entities)
        for entity in custom_entities_sorted[:20]:  # Limit to first 20 to avoid flooding
            print(f"  - {entity}")
        if len(custom_entities_sorted) > 20:
            print(f"  ... plus {len(custom_entities_sorted) - 20} more entities")
        sys.stdout.flush()
        
        # Look specifically for class action related entities
        print("\nSearching for class action related entities:")
        class_action_entities = [e for e in custom_entities if "class" in e.lower()]
        for entity in class_action_entities:
            print(f"  - {entity}")
        sys.stdout.flush()
        
        return True
    except Exception as e:
        print(f"❌ Failed to discover entities: {str(e)}")
        sys.stdout.flush()
        logger.error(f"❌ Failed to discover entities: {str(e)}")
        return False

async def test_problem_case():
    """Test the specific case that's failing in production"""
    print("\nTesting the problematic case query...")
    sys.stdout.flush()
    case_id = "309ea91c-a4d2-ef11-8eea-6045bdfec0f2"
    
    # Try different entity names
    entity_variations = [
        "new_classactioncase", 
        "new_classactioncases",
        "classactioncase", 
        "classactioncases"
    ]
    
    # Try different filter syntaxes
    filter_variations = [
        f"new_classactioncaseid eq '{case_id}'",  # Original with quotes
        f"new_classactioncaseid eq {case_id}",    # Without quotes
        # GUID format with guid prefix
        # Break string to meet line length requirements
        "_new_classactioncaseid_value eq "
        f"guid'{case_id}'",
        # _value with quotes
        (f"_new_classactioncaseid_value eq "
         f"'{case_id}'")
    ]
    
    success = False
    
    print(f"Testing case ID: {case_id}")
    # Log Dynamics URL in a way that won't exceed line length
    url = DYNAMICS_CONFIG.get('resource_url', 'Not Set')
    print(f"Dynamics URL: {url}")
    sys.stdout.flush()
    logger.info(f"Testing case ID: {case_id}")
    logger.info(f"Dynamics URL: {url}")
    
    # First, try all entity name variations with the original filter
    for entity in entity_variations:
        try:
            print(f"Trying entity: {entity}")
            sys.stdout.flush()
            logger.info(f"Trying entity: {entity}")
            result = await query_entity(entity, filter_variations[0])
            print(f"✅ Success with entity '{entity}'!")
            sys.stdout.flush()
            logger.info(f"✅ Success with entity '{entity}'!")
            logger.info(f"Results: {result}")
            success = True
        except Exception as e:
            print(f"❌ Failed with entity '{entity}': {str(e)}")
            sys.stdout.flush()
            logger.error(f"❌ Failed with entity '{entity}': {str(e)}")
    
    # If still no success, try all filter variations with both entity names
    if not success:
        print("\nTrying all filter variations...\n")
        sys.stdout.flush()
        for entity in entity_variations:
            for i, filter_expr in enumerate(filter_variations):
                try:
                    print(f"Trying entity: {entity} with filter variation #{i+1}")
                    print(f"Filter: {filter_expr}")
                    sys.stdout.flush()
                    logger.info(f"Trying entity: {entity}")
                    logger.info("Trying filter...")
                    
                    # Execute query
                    result = await query_entity(
                        entity=entity,
                        filter_str=filter_expr
                    )
                    # Short messages to avoid line length issues
                    print(f"✅ Success! Entity '{entity}' with filter #{i+1} works!")
                    sys.stdout.flush()
                    logger.info(f"✅ Match: {entity}")
                    logger.info("Filter works!")
                    # Log result count, not full content
                    result_len = 0
                    if isinstance(result, list):
                        result_len = len(result)
                    else:
                        result_len = 1
                    print(f"Found {result_len} result(s)")
                    sys.stdout.flush()
                    logger.info(f"Found {result_len} result(s)")
                    success = True
                    break
                except Exception as e:
                    # Print full error details to console
                    print(f"❌ Failed with entity '{entity}' and filter: {filter_expr}")
                    print(f"Error details: {e}")
                    sys.stdout.flush()
                    
                    # Keep logs clean with shortened messages
                    err_prefix = f"❌ Failed with entity '{entity}'"
                    err_filter = filter_expr[:15] + "..."
                    logger.error(f"{err_prefix}:")
                    logger.error(f"Filter: '{err_filter}'")
                    logger.error(f"Error details: {e}")
                    continue
            if success:
                break
    
    if not success:
        print("❌ All entity and filter combinations failed!")
        sys.stdout.flush()
    
    return success


async def test_related_entities():
    """Test querying entities that might be related to class action cases"""
    print("\nTesting related entities...")
    sys.stdout.flush()
    related_entities = [
        "new_classmember", 
        "new_classmembers",
        "new_classactioncase", 
        "new_classactioncases",
        "incident"
    ]
    
    for entity in related_entities:
        try:
            print(f"Querying top records from {entity}")
            sys.stdout.flush()
            logger.info(f"Querying top records from {entity}")
            # Use $top in the OData filter string to limit results
            result = await query_entity(entity, "$top=2")
            if result:
                print(f"✅ Success querying {entity}, found {len(result)} records")
                if len(result) > 0:
                    print(f"Sample fields: {list(result[0].keys())[:5]}")
                sys.stdout.flush()
                logger.info(
                    f"✅ Success querying {entity}, "
                    f"found {len(result)} records"
                )
                logger.info(f"Sample fields: {list(result[0].keys())[:5]}")
            else:
                print(f"⚠️ No records found in {entity}")
                sys.stdout.flush()
                logger.info(f"⚠️ No records found in {entity}")
        except Exception as e:
            print(f"❌ Failed to query {entity}: {str(e)}")
            sys.stdout.flush()
            logger.error(f"❌ Failed to query {entity}: {str(e)}")


async def main():
    """Run all tests"""
    print("====== Starting Dynamics 365 diagnostics ======")
    sys.stdout.flush()
    logger.info("====== Starting Dynamics 365 diagnostics ======")
    
    if not await test_connection():
        print("❌ Failed the basic connection test - aborting remaining tests")
        sys.stdout.flush()
        logger.error("❌ Failed the basic connection test - "
                     "aborting remaining tests")
        return
    
    print("\n====== Testing entity discovery ======")
    sys.stdout.flush()
    logger.info("\n====== Testing entity discovery ======")
    await test_entity_discovery()
    
    print("\n====== Testing problematic case query ======")
    sys.stdout.flush()
    logger.info("\n====== Testing problematic case query ======")
    await test_problem_case()
    
    print("\n====== Testing related entities ======")
    sys.stdout.flush()
    logger.info("\n====== Testing related entities ======")
    await test_related_entities()
    
    print("\n====== Testing complete ======")
    sys.stdout.flush()
    logger.info("\n====== Testing complete ======")

if __name__ == "__main__":
    asyncio.run(main())
