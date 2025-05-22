# /core/templates.py
import re

def parse_template(template_content: str):
    parsed_sections = {}
    # Split by H2 headers (## Section Title), keeping the headers
    # This regex captures the '## Header' line itself.
    parts = re.split(r'(^## .*$)', template_content, flags=re.MULTILINE)

    current_title = None
    content_buffer = []

    for part_content in parts:
        part_stripped = part_content.strip()
        if not part_stripped: # Skip empty parts from split
            continue

        is_header = re.match(r'^## .*$', part_stripped, flags=re.MULTILINE)
        
        if is_header:
            # If we have a current_title and accumulated content from the previous section, save it
            if current_title and content_buffer:
                parsed_sections[current_title] = "".join(content_buffer).strip()
            
            # Start a new section
            current_title = part_stripped.replace("## ", "").strip() # Extract title
            content_buffer = [] # Reset buffer for the new section's content
            # Initialize section in dict; its content will be the joined content_buffer later
            parsed_sections[current_title] = "" 
        elif current_title:
            # This is content for the current_title (e.g., (AI: ...) lines, diagram placeholders)
            content_buffer.append(part_content) # Append raw part, stripping will happen when buffer is joined
        # else:
            # This is content before any "##" header has been encountered (e.g., an H1 document title)
            # or empty strings after filtering. We are currently ignoring such pre-header content for section generation.
            # print(f"DEBUG TEMPLATE PARSER: Ignoring pre-header or unassigned content: '{part_content[:100].strip()}...'")
            pass

    # Save the last accumulated section content
    if current_title and content_buffer:
        parsed_sections[current_title] = "".join(content_buffer).strip()
    
    # print(f"DEBUG: Parsed template sections and their template content keys: {list(parsed_sections.keys())}")
    # for title, content in parsed_sections.items():
    #     print(f"  Section: '{title}', Template Content Hint: '{content[:100].strip()}...'")
        
    return parsed_sections

def find_diagram_placeholders(section_content: str): # <--- THIS FUNCTION WAS MISSING
    # Example placeholder: <!-- DIAGRAM: type=architecture description="Overall System View" -->
    placeholders = []
    # Regex to find placeholders like <!-- DIAGRAM: key1="value1" key2="value2" ... -->
    # It looks for 'key="value"' pairs.
    for match in re.finditer(r'<!-- DIAGRAM:\s*(.*?)\s*-->', section_content):
        full_match = match.group(0)
        params_str = match.group(1).strip()
        
        params = {}
        # Regex to find key="value" pairs, allowing for spaces around '='
        # and quotes around value.
        param_pattern = re.compile(r'(\w+)\s*=\s*"(.*?)"')
        for param_match in param_pattern.finditer(params_str):
            key = param_match.group(1)
            value = param_match.group(2)
            params[key] = value
            
        if "type" in params and "description" in params:
             placeholders.append({"full_match": full_match, "params": params})
        # else: # Optional: print a warning if a DIAGRAM comment is malformed
        #     print(f"Warning: Malformed diagram placeholder found and ignored: {full_match}")
            
    return placeholders