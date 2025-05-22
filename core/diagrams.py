# /core/diagrams.py
from abc import ABC, abstractmethod
import requests
import json
from .models import BaseModelConnector # <--- ADD THIS LINE

class BaseDiagramGenerator(ABC):
    @abstractmethod
    def generate_diagram(self, context: str, diagram_type: str, description: str) -> str:
        """Generates diagram code/data based on context."""
        pass

class MermaidGenerator(BaseDiagramGenerator):
    def __init__(self, config, llm_connector: BaseModelConnector = None):
        self.llm_connector = llm_connector

    def generate_diagram(self, context: str, diagram_type: str, description: str) -> str:
        # Determine if it's a C4 diagram type
        is_c4_diagram = diagram_type.lower().startswith('c4')

        c4_example = ""
        if is_c4_diagram:
            # Provide a very explicit C4 example for Mermaid
            c4_example = """
        For C4 diagrams (like 'c4_context', 'c4_container', etc.), the Mermaid syntax MUST start with 'C4Context', 'C4Container', etc.
        Here is a simple example of a Mermaid C4Context diagram:
        ```mermaid
        C4Context
          title System Context for [System Name]
          Person(actor, "Actor Name", "Description of actor")
          System(currentSystem, "Current System", "Description of current system")
          Rel(actor, currentSystem, "Interacts with")
        ```
        Ensure your output for this C4 diagram strictly follows this Mermaid C4 structure.
        """

        prompt = f"""
        You are an expert diagram generator. Your SOLE task is to generate diagram code using **MERMAID SYNTAX ONLY**.
        You MUST NOT generate PlantUML, DOT, or any other diagramming language.
        The output MUST be a single, valid Mermaid code block, starting with ```mermaid and ending with ```.
        Do NOT include any explanations, apologies, or text outside this Mermaid code block.

        Diagram Type Requested: '{diagram_type}'
        Description: '{description}'
        {c4_example}
        Context to use for the diagram content:
        --- START CONTEXT ---
        {context}
        --- END CONTEXT ---

        Generate the Mermaid code now for the '{diagram_type}' diagram.
        """
        
        mermaid_code = self.llm_connector.generate_content(prompt)

        # Basic cleanup and validation
        output_lines = mermaid_code.strip().split('\n')
        
        # Remove potential "Here's the Mermaid code:" preamble if LLM adds it
        if output_lines and "mermaid" not in output_lines[0].lower() and output_lines[0].lower().startswith(("here", "sure", "okay", "certainly")):
            # Find the actual start of the mermaid block
            mermaid_start_index = -1
            for i, line in enumerate(output_lines):
                if line.strip().startswith("```mermaid"):
                    mermaid_start_index = i
                    break
            if mermaid_start_index != -1:
                output_lines = output_lines[mermaid_start_index:]
        
        cleaned_code = "\n".join(output_lines)

        if not cleaned_code.startswith("```mermaid"):
            cleaned_code = f"```mermaid\n{cleaned_code}"
        if not cleaned_code.endswith("```"):
            # Be careful not to add ``` if it's already ```mermaid ... ```            if cleaned_code.strip().endswith("```mermaid"): # Should not happen but as a safeguard
                 pass # It's already closed by the start
        elif not cleaned_code.strip().endswith("\n```"):
            cleaned_code = f"{cleaned_code.rstrip()}\n```"


        # Stronger check for PlantUML contamination
        if "@startuml" in cleaned_code or "!include" in cleaned_code:
            print(f"WARNING: LLM generated PlantUML-like syntax despite being asked for Mermaid for '{description}'. Attempting to return a placeholder.")
            # You could try a re-prompt here, or just return a clear error/placeholder
            return f"```mermaid\n%% ERROR: LLM failed to generate valid Mermaid syntax for {diagram_type}: {description}.\n%% Received PlantUML-like content instead.\ngraph TD\n  A[Generation Error for {description}]\n```"
            
        return cleaned_code

class MCPConnector(BaseDiagramGenerator):
    def __init__(self, config):
        self.server_url = config.get("mcp_server_url")
        if not self.server_url:
            raise ValueError("AWS Diagram MCP Server URL not configured.")

    def generate_diagram(self, context: str, diagram_type: str, description: str) -> str:
        # This is a hypothetical implementation based on MCP concepts
        # You'll need to adapt this based on the *actual* AWS Diagram MCP Server API spec
        mcp_payload = {
            "protocol_version": "1.0", # Example version
            "request_id": "some_unique_id", # Generate dynamically
            "context": {
                "text": context,
                "metadata": {
                    "diagram_type": diagram_type,
                    "description": description,
                    # Add any other relevant metadata the server expects
                }
            },
            "parameters": {
                "output_format": "mermaid" # Or 'svg', 'png' etc. - depends on server capabilities
            }
        }
        try:
            response = requests.post(
                f"{self.server_url}/generate", # Assuming an endpoint like /generate
                json=mcp_payload,
                timeout=30 # Set a reasonable timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            # Process the response based on the MCP server's spec
            if result.get("status") == "success" and result.get("diagram_data"):
                 # Assuming it returns Mermaid code directly if requested
                 if mcp_payload["parameters"]["output_format"] == "mermaid":
                     # Ensure it's properly formatted as a Markdown code block
                     code = result["diagram_data"]
                     if not code.strip().startswith("```mermaid"):
                         return f"```mermaid\n{code.strip()}\n```"
                     else:
                         return code
                 else:
                     # Handle other formats (e.g., return an image tag or link)
                     return f"<!-- Diagram generated by MCP: type={diagram_type}, format={result.get('format', 'unknown')} -->\n[Diagram: {description}]({result.get('diagram_url', '#')})" # Example
            else:
                error_message = result.get("error", "Unknown error from MCP server")
                print(f"MCP Server Error: {error_message}")
                raise ConnectionError(f"MCP Server failed: {error_message}")

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to AWS Diagram MCP Server at {self.server_url}: {e}")
            raise ConnectionError(f"Failed to connect to MCP Server: {e}") # Signal failure for fallback

class DiagramStrategy:
    @staticmethod
    def get_generator(method: str, config: dict, llm_connector: BaseModelConnector = None) -> BaseDiagramGenerator:
        if method == "mcp":
            try:
                # Ensure mcp_server_url is passed if it's part of the main config,
                # or that MCPConnector specifically looks for it in its passed 'config' dict.
                if not config.get("mcp_server_url"):
                    print("Warning: MCP Server URL not found in config for DiagramStrategy. MCP init might fail.")
                return MCPConnector(config)
            except Exception as e:
                print(f"MCP Connector initialization failed: {e}. Falling back to Mermaid.")
                # Fallback directly during instantiation failure
                return MermaidGenerator(config, llm_connector)
        elif method == "mermaid":
            return MermaidGenerator(config, llm_connector)
        else:
            raise ValueError(f"Unsupported diagram generation method: {method}")