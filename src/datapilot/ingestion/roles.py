from typing import Dict, Any

def infer_roles(profile: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    roles = {}
    for col_name, info in profile.items():
        name_lower = col_name.lower()
        semantic = info.get("semantic_type")
        
        # Simple heuristics for role mapping
        if semantic == "date" and "date" not in roles.values():
            roles[col_name] = "date"
        elif "price" in name_lower or "arpu" in name_lower:
            roles[col_name] = "price"
        elif "quantity" in name_lower or "volume" in name_lower or "qty" in name_lower:
            roles[col_name] = "quantity"
        elif "revenue" in name_lower or "sales" in name_lower or "mrr" in name_lower:
            roles[col_name] = "metric"
        elif "customer" in name_lower and "id" in name_lower:
            roles[col_name] = "customer_id"
        elif semantic == "id" and "customer_id" not in roles.values():
            roles[col_name] = "customer_id"
        elif "region" in name_lower or "country" in name_lower:
            roles[col_name] = "region"
        elif "product" in name_lower or "item" in name_lower:
            roles[col_name] = "product"
        elif "channel" in name_lower or "source" in name_lower:
            roles[col_name] = "channel"
        elif semantic == "category":
            roles[col_name] = "segment"
        elif semantic == "metric" and "metric" not in roles.values():
            roles[col_name] = "metric"
            
    return roles
