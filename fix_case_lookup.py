#!/usr/bin/env python3
"""
Focused test script to diagnose the specific 404 error with case ID lookup.
Tests multiple entity variations and query syntaxes to find the working combination.
"""
import asyncio
import os
import sys
import requests
from dotenv import load_dotenv
from user_functions import get_access_token

# Load environment variables
load_dotenv()

# Get Dynamics 365 configuration from environment
DYNAMICS_CONFIG = {
    "tenant_id": os.getenv("D365_TENANT_ID"),
    "client_id": os.getenv("D365_CLIENT_ID"),
    "client_secret": os.getenv("D365_CLIENT_SECRET"),
    "resource_url": os.getenv("D365_RESOURCE_URL"),
}

# The case ID that's failing in production
PROBLEM_CASE_ID = "309ea91c-a4d2-ef11-8eea-6045bdfec0f2"


async def direct_query(url, access_token):
    """Execute a direct API query to Dynamics 365.
    
    This bypasses the abstraction in user_functions.py to give
    us more control over the exact request and response.
    """
    print(f"Direct query to: {url}")
    sys.stdout.flush()
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
        "Prefer": "odata.include-annotations=*"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        # Print both status and detailed info
        print(f"Response status: {response.status_code}")
        
        # Check if we can get JSON response
        try:
            json_data = response.json()
            value_count = len(json_data.get("value", []))
            print(f"Response contains {value_count} values")
            
            if value_count > 0:
                # Print first item keys to analyze the schema
                print(f"First item keys: {list(json_data['value'][0].keys())}")
                
                # If there's a case ID field, print it to verify
                case_id_fields = [
                    k for k in json_data['value'][0].keys() 
                    if "caseid" in k.lower() or "id" in k.lower()
                ]
                if case_id_fields:
                    print(f"Potential ID fields found: {case_id_fields}")
                    for field in case_id_fields[:3]:  # Limit to first 3
                        print(f"  {field}: {json_data['value'][0][field]}")
                        
        except ValueError:
            print("Response is not valid JSON")
            if response.content:
                print(f"Response content: {response.content[:300]}")
                
        # Always raise for HTTP errors to trigger the exception handler
        response.raise_for_status()
        return response.json().get("value", [])
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response body: {e.response.text[:300]}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


async def list_entities(access_token, search="class"):
    """List entities that match a search term."""
    base_url = DYNAMICS_CONFIG["resource_url"].rstrip("/")
    url = f"{base_url}/api/data/v9.2/EntityDefinitions"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0"
    }
    
    try:
        print(f"Fetching entity definitions containing '{search}'...")
        sys.stdout.flush()
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        entities = []
        for entity in response.json().get("value", []):
            name = entity.get("LogicalName", "")
            display_name = entity.get("DisplayName", {}).get("UserLocalizedLabel", {}).get("Label", "")
            
            if search.lower() in name.lower() or search.lower() in display_name.lower():
                entities.append({
                    "name": name,
                    "display": display_name
                })
        
        # Sort and print the found entities
        entities.sort(key=lambda x: x["name"])
        print(f"\nFound {len(entities)} entities with '{search}' in the name:")
        for entity in entities:
            print(f"  {entity['name']} - {entity['display']}")
        sys.stdout.flush()
        
        return entities
    except Exception as e:
        print(f"Error listing entities: {str(e)}")
        return []


async def test_case_variations():
    """Test different combinations to find the working entity and filter syntax."""
    print("Obtaining access token...")
    sys.stdout.flush()
    
    try:
        access_token = await get_access_token()
        print("✅ Successfully obtained access token\n")
        sys.stdout.flush()
        
        # First, identify relevant entity names
        entities = await list_entities(access_token, "class")
        
        # These are the entity variations we'll test
        entity_variations = [
            "new_classactioncase",
            "new_classactioncases", 
            "incidents",          # Common entity for cases
            "incident"            # Common entity for cases (singular)
        ]
        
        # Add any discovered entities that might be relevant
        for entity in entities:
            name = entity["name"]
            if name not in entity_variations and (
                "case" in name.lower() or 
                "class" in name.lower() or 
                "action" in name.lower()
            ):
                entity_variations.append(name)
        
        # Test with direct entity access first (no filter)
        base_url = DYNAMICS_CONFIG["resource_url"].rstrip("/")
        
        print("\nTesting direct entity access (getting first few records):")
        for entity in entity_variations:
            # Craft URL to get top 2 records
            url = f"{base_url}/api/data/v9.2/{entity}?$top=2"
            result = await direct_query(url, access_token)
            if result is not None and result:
                print(f"✅ Successfully accessed entity: {entity}\n")
            else:
                print(f"❌ Failed to access entity: {entity}\n")
        
        # These are the filter variations we'll test
        id_field_variations = [
            "new_classactioncaseid", 
            "_new_classactioncaseid_value",
            "incidentid",              # Standard field for incidents
            "_incidentid_value",       # Standard field for incidents
            "new_class_action_case_id",
            "id"                       # Generic ID field
        ]
        
        print("\nTesting case ID lookup with different filter syntaxes:")
        for entity in entity_variations:
            for id_field in id_field_variations:
                # Test with quotes around the GUID
                filter_with_quotes = f"{id_field} eq '{PROBLEM_CASE_ID}'"
                url = f"{base_url}/api/data/v9.2/{entity}?$filter={filter_with_quotes}"
                result = await direct_query(url, access_token)
                if result is not None and result:
                    print(f"✅ Success! Entity: {entity}, ID field: {id_field} with quotes\n")
                
                # Test without quotes around the GUID
                filter_without_quotes = f"{id_field} eq {PROBLEM_CASE_ID}"
                url = f"{base_url}/api/data/v9.2/{entity}?$filter={filter_without_quotes}"
                result = await direct_query(url, access_token)
                if result is not None and result:
                    print(f"✅ Success! Entity: {entity}, ID field: {id_field} without quotes\n")
                
                # Test with guid' prefix
                filter_with_guid = f"{id_field} eq guid'{PROBLEM_CASE_ID}'"
                url = f"{base_url}/api/data/v9.2/{entity}?$filter={filter_with_guid}"
                result = await direct_query(url, access_token)
                if result is not None and result:
                    print(f"✅ Success! Entity: {entity}, ID field: {id_field} with guid prefix\n")
        
        print("\nTesting complete!")
        sys.stdout.flush()
    
    except Exception as e:
        print(f"Error in test_case_variations: {str(e)}")
        sys.stdout.flush()


async def main():
    """Main entry point."""
    print("====== Starting Dynamics 365 Case Lookup Diagnostics ======")
    print(f"Targeting case ID: {PROBLEM_CASE_ID}")
    
    # Print environment info
    print("\n===== ENVIRONMENT INFO =====")
    print(f"Dynamics Resource URL set: {bool(DYNAMICS_CONFIG.get('resource_url'))}")
    if DYNAMICS_CONFIG.get('resource_url'):
        url_parts = DYNAMICS_CONFIG['resource_url'].split('/')
        if len(url_parts) >= 3:
            domain = url_parts[2]
            print(f"Dynamics domain: {domain}")
    
    print(f"Tenant ID set: {bool(DYNAMICS_CONFIG.get('tenant_id'))}")
    print(f"Client ID set: {bool(DYNAMICS_CONFIG.get('client_id'))}")
    print(f"Client Secret set: {bool(DYNAMICS_CONFIG.get('client_secret'))}")
    print(f"Current directory: {os.getcwd()}")
    print("===== END ENVIRONMENT INFO =====\n")
    sys.stdout.flush()
    
    await test_case_variations()


if __name__ == "__main__":
    asyncio.run(main())
