resource "dynatrace_management_zone" "this" {
  name        = var.name
  description = var.description

  rule {
    type        = var.rule_type
    condition   = var.rule_condition
    value       = var.rule_value
  }

  # Add other configuration details as needed
}

output "mz_id" {
  value = dynatrace_management_zone.this.id
}

variable "name" {
  type        = string
  description = "The name of the management zone."
}

variable "description" {
  type        = string
  description = "Description of the management zone."
  default     = ""
}

variable "rule_type" {
  type        = string
  description = "The type of rule to apply."
}

variable "rule_condition" {
  type        = string
  description = "The condition for the rule."
}

variable "rule_value" {
  type        = string
  description = "The value for the rule."
}

resource "dynatrace_http_monitor" "synthetic" {
  for_each = { for test in var.synthetic_tests : test.name => test }

  name        = each.value.name
  frequency   = each.value.frequency
  enabled     = each.value.enabled
  locations   = each.value.locations
  tags        = each.value.tags

  # Optionally add dynamic blocks if scripts or other nested configuration is needed
}

output "monitor_ids" {
  value = { for name, monitor in dynatrace_http_monitor.synthetic : name => monitor.id }
}

variable "synthetic_tests" {
  type = list(object({
    name       = string
    frequency  = number
    enabled    = optional(bool, true)
    locations  = optional(list(string), [])
    tags       = optional(list(string), [])
  }))
  description = "List of synthetic monitoring tests to deploy."
}

#!/usr/bin/env python3
"""
compare_resources.py

A detailed Python script that detects if two resource definitions (in JSON format)
are the same. It supports ignoring specific keys and outputs detailed differences.

Usage:
    python compare_resources.py resource1.json resource2.json [--ignore id,timestamp]
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple, Union

def load_json_file(filename: str) -> Any:
    """Loads JSON data from a file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file '{filename}': {e}")
        sys.exit(1)

def normalize_resource(
    resource: Any, ignore_keys: List[str] = None
) -> Any:
    """
    Recursively normalizes the resource for comparison.
    
    - If resource is a dictionary, it sorts keys and removes any keys that should be ignored.
    - If resource is a list, it sorts the list items (if possible) or normalizes each item.
    """
    if ignore_keys is None:
        ignore_keys = []
    
    if isinstance(resource, dict):
        normalized = {}
        for key, value in resource.items():
            if key in ignore_keys:
                continue
            normalized[key] = normalize_resource(value, ignore_keys)
        # Return a dict sorted by keys for consistency
        return dict(sorted(normalized.items()))
    
    elif isinstance(resource, list):
        # Normalize each item in the list
        normalized_list = [normalize_resource(item, ignore_keys) for item in resource]
        try:
            # Try to sort the list if items are comparable
            return sorted(normalized_list, key=lambda x: json.dumps(x, sort_keys=True))
        except Exception:
            # Otherwise, return the normalized list as is
            return normalized_list
    else:
        return resource

def deep_diff(
    res1: Any, res2: Any, path: str = ""
) -> List[str]:
    """
    Recursively compares two normalized resources.
    
    Returns a list of differences with a description of where the difference occurs.
    """
    diffs = []
    if type(res1) != type(res2):
        diffs.append(f"Type mismatch at '{path}': {type(res1).__name__} != {type(res2).__name__}")
        return diffs

    if isinstance(res1, dict):
        # Check keys in both dictionaries
        all_keys = set(res1.keys()).union(res2.keys())
        for key in all_keys:
            new_path = f"{path}.{key}" if path else key
            if key not in res1:
                diffs.append(f"Key '{new_path}' missing in first resource")
            elif key not in res2:
                diffs.append(f"Key '{new_path}' missing in second resource")
            else:
                diffs.extend(deep_diff(res1[key], res2[key], new_path))
    elif isinstance(res1, list):
        # Compare list lengths
        if len(res1) != len(res2):
            diffs.append(f"List length mismatch at '{path}': {len(res1)} != {len(res2)}")
        # Compare each item in order
        for index, (item1, item2) in enumerate(zip(res1, res2)):
            new_path = f"{path}[{index}]"
            diffs.extend(deep_diff(item1, item2, new_path))
    else:
        if res1 != res2:
            diffs.append(f"Value mismatch at '{path}': {res1} != {res2}")
    return diffs

def resources_are_same(resource1: Any, resource2: Any, ignore_keys: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Normalizes and compares two resources.
    
    Returns a tuple (is_same, differences). is_same is True if the resources match.
    """
    norm1 = normalize_resource(resource1, ignore_keys)
    norm2 = normalize_resource(resource2, ignore_keys)
    differences = deep_diff(norm1, norm2)
    return (len(differences) == 0, differences)

def main():
    parser = argparse.ArgumentParser(
        description="Detect if two resource definitions (JSON files) are the same."
    )
    parser.add_argument("resource1", help="Path to first JSON resource file")
    parser.add_argument("resource2", help="Path to second JSON resource file")
    parser.add_argument(
        "--ignore",
        type=lambda s: s.split(","),
        default="",
        help="Comma-separated list of keys to ignore (e.g. id,timestamp)",
    )
    args = parser.parse_args()

    # Load the resources from files
    res1 = load_json_file(args.resource1)
    res2 = load_json_file(args.resource2)
    
    # Convert ignore argument to list (filter out empty strings)
    ignore_keys = [key.strip() for key in args.ignore if key.strip()]
    
    same, diffs = resources_are_same(res1, res2, ignore_keys)
    if same:
        print("The two resources are the same (after normalization).")
    else:
        print("The resources differ:")
        for diff in diffs:
            print(f"  - {diff}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
split_tfstate.py

This utility reads a Terraform state file (tfstate) and splits it into separate files
based on resource type. You can optionally specify a particular resource type to extract.
This is useful when you want to divide a monolithic state (such as one generated by a Dynatrace export)
into smaller state files, each containing only a subset of resources.

Usage:
    python split_tfstate.py --input state.tfstate --output-dir output_states
    python split_tfstate.py --input state.tfstate --output-dir output_states --resource-type dynatrace_slo

Arguments:
    --input         Path to the input tfstate file.
    --output-dir    Directory where the output tfstate files will be written.
    --resource-type (Optional) If provided, only resources of this type will be extracted into one file.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from collections import defaultdict

def load_tfstate(filename: str) -> dict:
    """Load and return the JSON tfstate from the given file."""
    try:
        with open(filename, "r") as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"Error loading tfstate file '{filename}': {e}")
        sys.exit(1)

def filter_module_resources(module: dict, resource_type_filter: str = None) -> dict:
    """
    Given a module dictionary from the tfstate, filter its 'resources' dictionary
    to only include resources matching the resource_type_filter (if provided).
    Returns a new module dictionary with the filtered resources.
    """
    if "resources" not in module:
        return {}

    filtered_resources = {}
    for res_key, res_val in module["resources"].items():
        # Each resource should have a "type" field
        res_type = res_val.get("type")
        if resource_type_filter:
            if res_type == resource_type_filter:
                filtered_resources[res_key] = res_val
        else:
            filtered_resources[res_key] = res_val

    return filtered_resources

def split_state_by_resource_type(state: dict) -> dict:
    """
    Splits the given tfstate into multiple state dictionaries, one per resource type.
    Returns a dictionary mapping resource_type to a new tfstate dict.
    """
    # Create a dict to hold new state per resource type
    state_by_type = defaultdict(lambda: deepcopy(state))
    
    # We assume the tfstate has a "modules" key which is a list.
    # We iterate over each module and then each resource.
    # For each resource, we remove it from all states except the one for its type.
    modules = state.get("modules", [])
    
    # Initialize new modules for each resource type:
    # For each resource type, we want to build a list of modules (with same metadata as original)
    grouped_modules = defaultdict(list)
    
    for module in modules:
        # For each module, check its resources
        if "resources" not in module:
            continue
        for res_key, res_val in module["resources"].items():
            res_type = res_val.get("type")
            # Make a shallow copy of the module, but only with the resource for this type.
            mod_copy = deepcopy(module)
            mod_copy["resources"] = {res_key: res_val}
            grouped_modules[res_type].append(mod_copy)
    
    # For each resource type, create a new state dict
    new_states = {}
    for res_type, modules_list in grouped_modules.items():
        new_state = deepcopy(state)
        new_state["modules"] = modules_list
        new_states[res_type] = new_state

    return new_states

def filter_state_by_resource_type(state: dict, resource_type_filter: str) -> dict:
    """
    Given a tfstate dict and a resource type filter, returns a new state dict that
    contains only modules with resources matching the filter.
    """
    new_state = deepcopy(state)
    new_modules = []
    for module in state.get("modules", []):
        filtered = filter_module_resources(module, resource_type_filter)
        if filtered:
            mod_copy = deepcopy(module)
            mod_copy["resources"] = filtered
            new_modules.append(mod_copy)
    new_state["modules"] = new_modules
    return new_state

def write_state_file(state: dict, filename: str):
    """Write the given state dict to the specified filename in JSON format."""
    try:
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)
        print(f"State file written: {filename}")
    except Exception as e:
        print(f"Error writing state file '{filename}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Split a Terraform tfstate file into separate state files per resource type."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input tfstate file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the output tfstate files will be written.",
    )
    parser.add_argument(
        "--resource-type",
        help="(Optional) If provided, only extract resources of this type into a single state file.",
    )
    args = parser.parse_args()

    # Load the state file
    state = load_tfstate(args.input)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resource_type:
        # Filter state for the given resource type
        filtered_state = filter_state_by_resource_type(state, args.resource_type)
        output_filename = os.path.join(args.output_dir, f"state_{args.resource_type}.tfstate")
        write_state_file(filtered_state, output_filename)
    else:
        # Split the state into multiple files, one per resource type
        states_by_type = split_state_by_resource_type(state)
        for res_type, state_dict in states_by_type.items():
            # Sanitize resource type for file name usage
            safe_res_type = res_type.replace(":", "_")
            output_filename = os.path.join(args.output_dir, f"state_{safe_res_type}.tfstate")
            write_state_file(state_dict, output_filename)

if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
split_tfstate.py

This utility reads a Terraform state file (tfstate) and splits it into separate files
based on resource type and/or a substring match in the resource name.
This is useful when you want to divide a monolithic state (such as one generated by a Dynatrace export)
into smaller state files for subsets of your resources.

Usage:
    # Split by resource type (all types get their own file)
    python split_tfstate.py --input state.tfstate --output-dir output_states

    # Filter to only include resources of a given type and that have "central" in their name
    python split_tfstate.py --input state.tfstate --output-dir output_states --resource-type dynatrace_slo --name-filter central
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from collections import defaultdict

def load_tfstate(filename: str) -> dict:
    """Load and return the JSON tfstate from the given file."""
    try:
        with open(filename, "r") as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"Error loading tfstate file '{filename}': {e}")
        sys.exit(1)

def filter_module_resources(
    module: dict,
    resource_type_filter: str = None,
    name_filter: str = None,
) -> dict:
    """
    Given a module dictionary from the tfstate, filter its 'resources' dictionary
    to only include resources matching the resource_type_filter and containing the
    name_filter substring (if provided). Returns a new dictionary with the filtered resources.
    """
    if "resources" not in module:
        return {}

    filtered_resources = {}
    for res_key, res_val in module["resources"].items():
        # Check resource type if a filter is provided.
        res_type = res_val.get("type")
        if resource_type_filter and res_type != resource_type_filter:
            continue
        # Check if the resource key (name) contains the substring (case-insensitive).
        if name_filter and name_filter.lower() not in res_key.lower():
            continue
        filtered_resources[res_key] = res_val

    return filtered_resources

def filter_state_by_criteria(
    state: dict, resource_type_filter: str = None, name_filter: str = None
) -> dict:
    """
    Given a tfstate dict, returns a new state dict that contains only modules with resources
    matching the given resource_type_filter and/or name_filter.
    """
    new_state = deepcopy(state)
    new_modules = []
    for module in state.get("modules", []):
        filtered = filter_module_resources(module, resource_type_filter, name_filter)
        if filtered:
            mod_copy = deepcopy(module)
            mod_copy["resources"] = filtered
            new_modules.append(mod_copy)
    new_state["modules"] = new_modules
    return new_state

def split_state_by_resource_type(state: dict, name_filter: str = None) -> dict:
    """
    Splits the given tfstate into multiple state dictionaries, one per resource type.
    Optionally, only resources whose keys contain name_filter are considered.
    Returns a dictionary mapping resource_type to a new tfstate dict.
    """
    # Group modules by resource type.
    grouped_modules = defaultdict(list)
    
    for module in state.get("modules", []):
        if "resources" not in module:
            continue
        for res_key, res_val in module["resources"].items():
            # If a name filter is provided, skip resources not matching it.
            if name_filter and name_filter.lower() not in res_key.lower():
                continue
            res_type = res_val.get("type")
            # Make a shallow copy of the module with only this resource.
            mod_copy = deepcopy(module)
            mod_copy["resources"] = {res_key: res_val}
            grouped_modules[res_type].append(mod_copy)
    
    new_states = {}
    for res_type, modules_list in grouped_modules.items():
        new_state = deepcopy(state)
        new_state["modules"] = modules_list
        new_states[res_type] = new_state

    return new_states

def write_state_file(state: dict, filename: str):
    """Write the given state dict to the specified filename in JSON format."""
    try:
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)
        print(f"State file written: {filename}")
    except Exception as e:
        print(f"Error writing state file '{filename}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Split a Terraform tfstate file into separate state files by resource type and/or resource name substring."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input tfstate file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the output tfstate files will be written.",
    )
    parser.add_argument(
        "--resource-type",
        help="(Optional) If provided, only extract resources of this type into a single state file.",
    )
    parser.add_argument(
        "--name-filter",
        help="(Optional) If provided, only extract resources whose names (keys) contain this substring.",
    )
    args = parser.parse_args()

    # Load the state file
    state = load_tfstate(args.input)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resource_type or args.name_filter:
        # Filter state by the provided criteria.
        filtered_state = filter_state_by_criteria(
            state, args.resource_type, args.name_filter
        )
        # Create a filename that reflects the applied filters.
        type_part = args.resource_type if args.resource_type else "all"
        name_part = args.name_filter if args.name_filter else "all"
        output_filename = os.path.join(
            args.output_dir, f"state_{type_part}_{name_part}.tfstate"
        )
        write_state_file(filtered_state, output_filename)
    else:
        # No filters provided: split the state into files per resource type.
        states_by_type = split_state_by_resource_type(state)
        for res_type, state_dict in states_by_type.items():
            safe_res_type = res_type.replace(":", "_")
            output_filename = os.path.join(
                args.output_dir, f"state_{safe_res_type}.tfstate"
            )
            write_state_file(state_dict, output_filename)

if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
state_merger.py

This utility merges multiple Terraform state files into a single state file.
It assumes that each input state file is in JSON format and has a "modules" key.
The merged state will contain a union of the modules from all input files.

Usage:
    python state_merger.py --input state1.tfstate state2.tfstate ... --output merged.tfstate
"""

import argparse
import json
import sys

def load_state(filename: str) -> dict:
    """Load and return a JSON tfstate from the given file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state file '{filename}': {e}")
        sys.exit(1)

def merge_states(state_files: list) -> dict:
    """
    Merge multiple tfstate dictionaries.
    This simple merger uses the first state's metadata and concatenates all modules.
    """
    if not state_files:
        sys.exit("No state files provided for merging.")
    
    merged_state = load_state(state_files[0])
    merged_modules = merged_state.get("modules", [])
    
    # Iterate over remaining state files and merge their modules.
    for filename in state_files[1:]:
        state = load_state(filename)
        modules = state.get("modules", [])
        merged_modules.extend(modules)
    
    merged_state["modules"] = merged_modules
    # Optionally, you could update the serial or version fields here if needed.
    return merged_state

def write_state(state: dict, filename: str):
    """Write the merged state to the given filename in JSON format."""
    try:
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Merged state written to: {filename}")
    except Exception as e:
        print(f"Error writing merged state file '{filename}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Terraform state files into one unified state file."
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="Paths to input tfstate files to merge."
    )
    parser.add_argument(
        "--output", required=True, help="Path for the merged output tfstate file."
    )
    args = parser.parse_args()

    merged = merge_states(args.input)
    write_state(merged, args.output)

if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
state_validator.py

This utility compares a Terraform state file with a live configuration file (both in JSON)
and outputs differences to detect drift.

Usage:
    python state_validator.py --state state.tfstate --live live_config.json [--ignore id,timestamp]
"""

import argparse
import json
import sys
from copy import deepcopy

def load_json(filename: str) -> dict:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        sys.exit(1)

def normalize(obj, ignore_keys=None):
    """Recursively normalize a JSON structure."""
    if ignore_keys is None:
        ignore_keys = []
    if isinstance(obj, dict):
        return {k: normalize(v, ignore_keys) for k, v in sorted(obj.items()) if k not in ignore_keys}
    elif isinstance(obj, list):
        return sorted(normalize(item, ignore_keys) for item in obj)
    else:
        return obj

def deep_diff(obj1, obj2, path=""):
    """Recursively compare two objects and return list of differences."""
    diffs = []
    if type(obj1) != type(obj2):
        diffs.append(f"Type mismatch at {path}: {type(obj1).__name__} != {type(obj2).__name__}")
        return diffs
    if isinstance(obj1, dict):
        keys = set(obj1.keys()).union(obj2.keys())
        for k in keys:
            new_path = f"{path}.{k}" if path else k
            if k not in obj1:
                diffs.append(f"Key {new_path} missing in first object")
            elif k not in obj2:
                diffs.append(f"Key {new_path} missing in second object")
            else:
                diffs.extend(deep_diff(obj1[k], obj2[k], new_path))
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            diffs.append(f"List length mismatch at {path}: {len(obj1)} != {len(obj2)}")
        for index, (i, j) in enumerate(zip(obj1, obj2)):
            diffs.extend(deep_diff(i, j, f"{path}[{index}]"))
    else:
        if obj1 != obj2:
            diffs.append(f"Value mismatch at {path}: {obj1} != {obj2}")
    return diffs

def main():
    parser = argparse.ArgumentParser(
        description="Validate Terraform state against live configuration to detect drift."
    )
    parser.add_argument("--state", required=True, help="Path to the tfstate file.")
    parser.add_argument("--live", required=True, help="Path to the live configuration file (JSON).")
    parser.add_argument("--ignore", type=lambda s: s.split(","), default=[], help="Comma-separated list of keys to ignore.")
    args = parser.parse_args()

    state = load_json(args.state)
    live = load_json(args.live)

    norm_state = normalize(state, ignore_keys=args.ignore)
    norm_live = normalize(live, ignore_keys=args.ignore)

    differences = deep_diff(norm_state, norm_live)
    if differences:
        print("Drift detected! Differences:")
        for diff in differences:
            print(f"  - {diff}")
        sys.exit(1)
    else:
        print("No drift detected: state and live configuration match.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
state_inventory_reporter.py

This utility reads a Terraform state file (tfstate) and prints an inventory report:
- Total resource count per resource type.
- List of resource keys for each type.

Usage:
    python state_inventory_reporter.py --input state.tfstate
"""

import argparse
import json
import sys
from collections import defaultdict

def load_state(filename: str) -> dict:
    """Load and return the tfstate JSON."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state file '{filename}': {e}")
        sys.exit(1)

def report_inventory(state: dict):
    """
    Process the state and print a report of resources per type.
    Assumes state has a "modules" list with each module containing a "resources" dict.
    """
    inventory = defaultdict(list)
    modules = state.get("modules", [])
    
    for module in modules:
        resources = module.get("resources", {})
        for res_key, res_val in resources.items():
            res_type = res_val.get("type", "unknown")
            inventory[res_type].append(res_key)
    
    print("Inventory Report:")
    for res_type, keys in inventory.items():
        print(f"\nResource Type: {res_type}")
        print(f"Count: {len(keys)}")
        for key in keys:
            print(f"  - {key}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate an inventory report from a Terraform state file."
    )
    parser.add_argument("--input", required=True, help="Path to the tfstate file.")
    args = parser.parse_args()

    state = load_state(args.input)
    report_inventory(state)

if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
config_diff_tool.py

This utility compares two Terraform state files (tfstate) and outputs a human-friendly diff.
It reports differences in the configuration (modules, resources, and values).

Usage:
    python config_diff_tool.py --state1 state1.tfstate --state2 state2.tfstate [--ignore id,timestamp]
"""

import argparse
import json
import sys
from copy import deepcopy

def load_state(filename: str) -> dict:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state file '{filename}': {e}")
        sys.exit(1)

def normalize(obj, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = []
    if isinstance(obj, dict):
        return {k: normalize(v, ignore_keys) for k, v in sorted(obj.items()) if k not in ignore_keys}
    elif isinstance(obj, list):
        return sorted(normalize(item, ignore_keys) for item in obj)
    else:
        return obj

def deep_diff(obj1, obj2, path=""):
    diffs = []
    if type(obj1) != type(obj2):
        diffs.append(f"Type mismatch at '{path}': {type(obj1).__name__} vs {type(obj2).__name__}")
        return diffs
    if isinstance(obj1, dict):
        all_keys = set(obj1.keys()).union(obj2.keys())
        for k in all_keys:
            new_path = f"{path}.{k}" if path else k
            if k not in obj1:
                diffs.append(f"Key '{new_path}' missing in first state")
            elif k not in obj2:
                diffs.append(f"Key '{new_path}' missing in second state")
            else:
                diffs.extend(deep_diff(obj1[k], obj2[k], new_path))
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            diffs.append(f"List length mismatch at '{path}': {len(obj1)} vs {len(obj2)}")
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            diffs.extend(deep_diff(item1, item2, f"{path}[{i}]"))
    else:
        if obj1 != obj2:
            diffs.append(f"Value mismatch at '{path}': {obj1} != {obj2}")
    return diffs

def main():
    parser = argparse.ArgumentParser(
        description="Compare two Terraform state files and report differences."
    )
    parser.add_argument("--state1", required=True, help="Path to the first tfstate file.")
    parser.add_argument("--state2", required=True, help="Path to the second tfstate file.")
    parser.add_argument("--ignore", type=lambda s: s.split(","), default=[], help="Comma-separated keys to ignore.")
    args = parser.parse_args()

    state1 = load_state(args.state1)
    state2 = load_state(args.state2)

    norm1 = normalize(state1, ignore_keys=args.ignore)
    norm2 = normalize(state2, ignore_keys=args.ignore)

    differences = deep_diff(norm1, norm2)
    if differences:
        print("Differences found between the two states:")
        for diff in differences:
            print(f"  - {diff}")
        sys.exit(1)
    else:
        print("The two state files are identical (after normalization).")

if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
backup_restore.py

This utility automates backing up Terraform state files and restoring them.
It can copy a tfstate file from a source directory to a backup directory with a timestamp,
and restore a backup file to a target location.

Usage:
    # To backup a state file:
    python backup_restore.py backup --state state.tfstate --backup-dir backups

    # To restore a backup file:
    python backup_restore.py restore --backup-file backups/state.tfstate.20250407T153000 --target state.tfstate
"""

import argparse
import os
import shutil
from datetime import datetime

def backup_state(state_path: str, backup_dir: str):
    """Backup the state file to the backup directory with a timestamp."""
    if not os.path.isfile(state_path):
        print(f"State file '{state_path}' does not exist.")
        return
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base_name = os.path.basename(state_path)
    backup_filename = f"{base_name}.{timestamp}"
    backup_path = os.path.join(backup_dir, backup_filename)
    shutil.copy2(state_path, backup_path)
    print(f"Backup created: {backup_path}")

def restore_state(backup_file: str, target_path: str):
    """Restore the backup file to the target path."""
    if not os.path.isfile(backup_file):
        print(f"Backup file '{backup_file}' does not exist.")
        return
    shutil.copy2(backup_file, target_path)
    print(f"Restored backup from '{backup_file}' to '{target_path}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Automate backup and restore of Terraform state files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup_parser = subparsers.add_parser("backup", help="Backup a state file")
    backup_parser.add_argument("--state", required=True, help="Path to the state file to backup")
    backup_parser.add_argument("--backup-dir", required=True, help="Directory to store backup files")

    restore_parser = subparsers.add_parser("restore", help="Restore a backup file")
    restore_parser.add_argument("--backup-file", required=True, help="Path to the backup file to restore")
    restore_parser.add_argument("--target", required=True, help="Target path for restoring the state file")

    args = parser.parse_args()

    if args.command == "backup":
        backup_state(args.state, args.backup_dir)
    elif args.command == "restore":
        restore_state(args.backup_file, args.target)

if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
resource_renamer.py

This utility renames resources within a Terraform state file based on a mapping.
The mapping can be provided as a JSON file containing key-value pairs, where keys
are substrings to search for in resource names and values are the replacements.

Usage:
    python resource_renamer.py --state state.tfstate --mapping mapping.json --output new_state.tfstate
"""

import argparse
import json
import os
import sys
from copy import deepcopy

def load_json_file(filename: str) -> dict:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        sys.exit(1)

def rename_resources_in_module(module: dict, mapping: dict) -> dict:
    """Rename resource keys in the module based on mapping."""
    if "resources" not in module:
        return module
    new_resources = {}
    for res_key, res_val in module["resources"].items():
        new_key = res_key
        for old, new in mapping.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_resources[new_key] = res_val
    module_copy = deepcopy(module)
    module_copy["resources"] = new_resources
    return module_copy

def rename_resources(state: dict, mapping: dict) -> dict:
    """Apply renaming to each module in the state."""
    new_state = deepcopy(state)
    new_modules = []
    for module in state.get("modules", []):
        new_modules.append(rename_resources_in_module(module, mapping))
    new_state["modules"] = new_modules
    return new_state

def main():
    parser = argparse.ArgumentParser(
        description="Rename resource keys in a Terraform state file based on a mapping."
    )
    parser.add_argument("--state", required=True, help="Path to the input tfstate file.")
    parser.add_argument("--mapping", required=True, help="Path to the JSON mapping file.")
    parser.add_argument("--output", required=True, help="Path to the output tfstate file.")
    args = parser.parse_args()

    state = load_json_file(args.state)
    mapping = load_json_file(args.mapping)

    new_state = rename_resources(state, mapping)
    try:
        with open(args.output, "w") as f:
            json.dump(new_state, f, indent=2)
        print(f"Renamed state file written to: {args.output}")
    except Exception as e:
        print(f"Error writing output file '{args.output}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
#!/usr/bin/env python3
"""
terraform_manager.py

A module that defines a Terraform class to manage plans, validation,
and reporting for each management zone independently when deployed.

Usage example:
    Suppose you have three management zones with Terraform configurations
    located in separate directories (e.g., "mz-east", "mz-west", "mz-central").

    You can use the Terraform class as follows:

        from terraform_manager import Terraform

        mz_dirs = {
            "East": "/path/to/mz-east",
            "West": "/path/to/mz-west",
            "Central": "/path/to/mz-central"
        }

        for mz_name, dir_path in mz_dirs.items():
            tf = Terraform(mz_name, dir_path)
            ret, validate_output = tf.validate()
            print(f"Validation output for {mz_name}:\n{validate_output}\n")
            ret, plan_output = tf.plan()
            print(f"Plan output for {mz_name}:\n{plan_output}\n")
            report = tf.report()
            print(f"Report for {mz_name}:\n{report}\n")
"""

import subprocess
import os
import sys

class Terraform:
    """
    A class to manage Terraform operations (plan, validate, report)
    for a given management zone (represented by its Terraform working directory).
    """
    def __init__(self, mz_name: str, working_dir: str):
        """
        Initialize the Terraform manager.

        Args:
            mz_name (str): The name of the management zone.
            working_dir (str): The directory containing the Terraform configuration for this zone.
        """
        self.mz_name = mz_name
        self.working_dir = working_dir
        
        if not os.path.isdir(self.working_dir):
            raise ValueError(f"Working directory '{self.working_dir}' does not exist.")

    def _run_command(self, command: list) -> (int, str):
        """
        Run a command in the Terraform working directory.

        Args:
            command (list): The command and arguments to run.

        Returns:
            tuple: A tuple (exit_code, output) where exit_code is an integer and output is the combined stdout/stderr.
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
            return result.returncode, result.stdout
        except Exception as e:
            return 1, f"Exception occurred: {str(e)}"

    def plan(self) -> (int, str):
        """
        Run 'terraform plan' in the working directory.

        Returns:
            tuple: (exit_code, output) from the terraform plan command.
        """
        print(f"Running 'terraform plan' for management zone: {self.mz_name}")
        # The -input=false and -no-color flags help automation
        ret, output = self._run_command(["terraform", "plan", "-input=false", "-no-color"])
        if ret != 0:
            print(f"'terraform plan' failed for {self.mz_name}.")
        return ret, output

    def validate(self) -> (int, str):
        """
        Run 'terraform validate' in the working directory.

        Returns:
            tuple: (exit_code, output) from the terraform validate command.
        """
        print(f"Running 'terraform validate' for management zone: {self.mz_name}")
        ret, output = self._run_command(["terraform", "validate", "-no-color"])
        if ret != 0:
            print(f"'terraform validate' failed for {self.mz_name}.")
        return ret, output

    def report(self) -> str:
        """
        Generate a simple report based on the output of 'terraform plan'.

        Returns:
            str: A human-readable summary report for this management zone.
        """
        ret, plan_output = self.plan()
        if ret != 0:
            return f"Plan failed for {self.mz_name}. Cannot generate report."

        # Look for a summary line in the plan output (e.g., "Plan: 2 to add, 0 to change, 1 to destroy.")
        summary = ""
        for line in plan_output.splitlines():
            if "Plan:" in line:
                summary = line.strip()
                break

        if not summary:
            summary = "No summary information found in plan output."
        
        report_text = (
            f"Management Zone: {self.mz_name}\n"
            f"Terraform Plan Summary: {summary}\n"
        )
        return report_text

# Example usage (if running this module directly):
if __name__ == "__main__":
    # Expecting directories to be passed as command-line arguments in the form: <mz_name>=<dir_path>
    # e.g.: python terraform_manager.py East=/path/to/mz-east West=/path/to/mz-west Central=/path/to/mz-central
    if len(sys.argv) < 2:
        print("Usage: python terraform_manager.py <mz_name1>=<dir_path1> [<mz_name2>=<dir_path2> ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        try:
            mz_name, dir_path = arg.split("=", 1)
        except ValueError:
            print(f"Invalid argument format: '{arg}'. Expected format: <mz_name>=<dir_path>")
            continue
        try:
            tf_manager = Terraform(mz_name, dir_path)
            print("="*60)
            print(f"Processing management zone: {mz_name}")
            ret_val, val_output = tf_manager.validate()
            print(f"Validate Output:\n{val_output}\n")
            ret_val, plan_output = tf_manager.plan()
            print(f"Plan Output:\n{plan_output}\n")
            print("Report:")
            print(tf_manager.report())
            print("="*60)
        except Exception as e:
            print(f"Error processing management zone '{mz_name}': {e}")
            

