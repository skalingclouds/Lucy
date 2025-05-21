import asyncio
import json
from user_functions import query_entity

async def test_case_lookup():
    """Test that we can look up the case that was previously failing."""
    print("===== Testing case lookup with updated entity mapping =====")
    
    # This is the case ID that was failing in the error logs
    case_id = "309ea91c-a4d2-ef11-8eea-6045bdfec0f2"
    
    # Test all different ways to query the same case
    test_scenarios = [
        # Entity name, field name, expected success
        ("incidents", "incidentid", True),                    # Direct query with correct names
        ("new_classactioncase", "new_classactioncaseid", True),  # Should map both entity and field
        ("new_classactioncases", "new_classactioncaseid", True)  # Should map both entity and field
    ]
    
    for entity_name, field_name, expected_success in test_scenarios:
        print(f"\nTesting entity: {entity_name} with field: {field_name}")
        
        filter_str = f"{field_name} eq {case_id}"
            
        try:
            result = await query_entity(entity_name, filter_str)
            if result:
                print(f"✅ Success! Found case with ID {case_id}")
                # Print a few fields to confirm it's the right record
                first_result = result[0]
                print(f"Title: {first_result.get('title', 'N/A')}")
                print(f"Case Number: {first_result.get('ticketnumber', 'N/A')}")
                # Other fields that might be useful for debugging
                field_names = list(first_result.keys())[:5]  # First 5 field names
                print(f"Available fields (first 5): {field_names}")
                
                if not expected_success:
                    print("⚠️ Warning: Query succeeded but we expected it to fail")
            else:
                print(f"❌ No results found for {entity_name} with filter: {filter_str}")
                if expected_success:
                    print("⚠️ Warning: Query failed but we expected it to succeed")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if expected_success:
                print("⚠️ Warning: Query failed but we expected it to succeed")

if __name__ == "__main__":
    asyncio.run(test_case_lookup())
