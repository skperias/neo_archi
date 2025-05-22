# /app.py
import streamlit as st
from core.orchestrator import build_graph, HLDWorkflowState
from core.config import get_config
import os
import ollama
from ollama._types import ListResponse
import time
import requests
import traceback
import datetime
import hashlib
from pathlib import Path
import concurrent.futures
import markdown
from weasyprint import HTML

st.set_page_config(layout="wide")
st.title("AI-Powered High-Level Design (HLD) Generator 📄✨")

# --- Configuration ---
config = get_config()

# --- Helper function to get Ollama models (with caching and refined parsing) ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_ollama_models(base_url):
    """Fetches the list of available Ollama models by accessing attributes of ListResponse and Model objects."""
    model_names = []
    print(f"DEBUG: Attempting to fetch Ollama models from base_url: {base_url}")
    try:
        client = ollama.Client(host=base_url)
        list_response_obj = client.list()

        print(f"DEBUG: Raw response object from ollama.Client().list(): {list_response_obj}")
        print(f"DEBUG: Type of list_response_obj: {type(list_response_obj)}")

        if not isinstance(list_response_obj, ollama._types.ListResponse):
            st.sidebar.warning(f"Ollama response is not the expected ListResponse object. Type: {type(list_response_obj)}")
            print(f"ERROR: Ollama response is not a ListResponse object. Type: {type(list_response_obj)}")
            return []

        if not hasattr(list_response_obj, 'models') or not isinstance(list_response_obj.models, list):
            st.sidebar.warning("Ollama ListResponse object does not contain a valid 'models' list attribute.")
            print(f"ERROR: Ollama ListResponse object missing 'models' attribute, or 'models' is not a list.")
            return []

        models_list = list_response_obj.models
        print(f"DEBUG: Extracted models_list from ListResponse.models: {models_list}")

        if not models_list:
            st.sidebar.info("Ollama 'models' list is empty. No models to display. Ensure models are pulled via 'ollama pull <modelname>'.")
            print("DEBUG: The models_list from Ollama is empty.")
            return []

        for i, model_obj in enumerate(models_list):
            print(f"DEBUG: Processing model_obj item {i}: {model_obj}")
            print(f"DEBUG: Type of model_obj item {i}: {type(model_obj)}")

            # CORRECTED: Access the 'model' attribute instead of 'name'
            if hasattr(model_obj, 'model') and isinstance(model_obj.model, str):
                model_names.append(model_obj.model) # Use model_obj.model
                print(f"DEBUG: Successfully appended model name: {model_obj.model}")
            else:
                # Check for 'model' attribute specifically in debug message
                model_attr_val_debug = getattr(model_obj, 'model', 'Attribute "model" not found')
                print(f"DEBUG: model_obj item {i} ('{model_obj}') does not have a 'model' string attribute. 'model' attribute value: {model_attr_val_debug}")
        
        if not model_names:
            print("DEBUG: model_names list is empty after processing all model_obj items.")
            st.sidebar.warning("No valid model names could be extracted from the Ollama models list.")
            
        return model_names
        
    except requests.exceptions.ConnectionError as ce:
        st.sidebar.error(f"Ollama connection error at '{base_url}': {ce}. Is Ollama running?")
        print(f"ERROR in get_ollama_models (ConnectionError): {ce}")
        return []
    except ollama.ResponseError as ore:
        st.sidebar.error(f"Ollama API Response Error: {ore.status_code} - {ore.error}")
        print(f"ERROR in get_ollama_models (ollama.ResponseError): Status {ore.status_code}, Error: {ore.error}")
        return []
    except Exception as e:
        st.sidebar.warning(f"An unexpected error occurred while fetching Ollama models: {type(e).__name__} - {e}")
        print(f"ERROR in get_ollama_models (Unexpected): {type(e).__name__} - {e}\n{traceback.format_exc()}")
        return []

def load_templates():
    """Load available HLD templates from the templates directory."""
    templates_dir = Path("./templates")
    templates_dir.mkdir(exist_ok=True)
    
    templates = {}
    for template_file in templates_dir.glob("*.md"):
        try:
            with open(template_file, "r", encoding="utf-8") as f:
                content = f.read()
            templates[template_file.stem] = content
        except Exception as e:
            print(f"Error loading template {template_file}: {e}")
    
    # Always include the default template
    if "default" not in templates:
        try:
            with open("hld_template.md", "r", encoding="utf-8") as f:
                templates["default"] = f.read()
        except:
            templates["default"] = """## Introduction
(AI: Describe the purpose and scope of the system based on the requirements.)

## Architecture Overview
(AI: Provide a high-level overview of the chosen architecture.)
<!-- DIAGRAM: type=architecture description="High-Level System Architecture" -->

## Component Design
(AI: Break down the major components of the system.)
<!-- DIAGRAM: type=component description="Component Breakdown" -->

## Data Flow
(AI: Explain the flow of data through the system.)
<!-- DIAGRAM: type=sequence description="Main Data Flow" -->

## Security Considerations
(AI: Discuss security aspects of the design.)

## Scalability and Performance
(AI: Address how the system handles growth and performance requirements.)

## Implementation Roadmap
(AI: Suggest an implementation approach and timeline.)
"""
    
    return templates

def save_markdown_to_file(content, requirements):
    """Save the generated HLD to a markdown file in the output directory."""
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate a unique filename based on timestamp and requirements hash
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    req_hash = hashlib.md5(requirements.encode()).hexdigest()[:8]
    filename = f"hld_{timestamp}_{req_hash}.md"
    
    # Full path
    file_path = output_dir / filename
    
    # Save the content
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path

# --- Sidebar ---
st.sidebar.header("Configuration")

# Model Provider Selection
model_provider = st.sidebar.selectbox(
    "Select AI Model Provider",
    ("ollama", "bedrock", "hybrid"),
    index=2 if config.get("aws_access_key_id") and config.get("aws_secret_access_key") else 0,
    help="Choose 'hybrid' to automatically select the best model for each task based on complexity and cost"
)

# Model Specific Options
model_config_ui = {"provider": model_provider}

if model_provider == "ollama":
    model_config_ui["ollama_base_url"] = st.sidebar.text_input(
        "Ollama Base URL",
        value=config.get("ollama_base_url", "http://localhost:11434")
    )
    available_ollama_models = get_ollama_models(model_config_ui["ollama_base_url"])
    default_ollama_model_name = config.get("default_ollama_model", "llama3")

    if available_ollama_models:
        default_index = 0
        if default_ollama_model_name in available_ollama_models:
            try:
                default_index = available_ollama_models.index(str(default_ollama_model_name))
            except ValueError:
                print(f"DEBUG: Default Ollama model '{default_ollama_model_name}' not in fetched list. Using first model as default.")
        
        selected_ollama_model = st.sidebar.selectbox(
            "Select Ollama Model",
            options=available_ollama_models,
            index=default_index
        )
        model_config_ui["ollama_model_id"] = selected_ollama_model
    else:
        st.sidebar.info("No Ollama models found or could not fetch list. Please enter Model ID manually.")
        model_config_ui["ollama_model_id"] = st.sidebar.text_input(
            "Ollama Model ID (Manual)",
            value=str(default_ollama_model_name)
        )

elif model_provider == "bedrock":
    model_config_ui["aws_region"] = st.sidebar.text_input("AWS Region", value=config.get("aws_region", "us-east-1"))
    st.sidebar.info("Ensure AWS credentials are configured via .env file or environment variables.")
    
    bedrock_model_options = {
        "Claude 3 Sonnet (Anthropic)": {
            "id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "family": "anthropic",
            "description": "Powerful reasoning model with strong architecture design capabilities"
        },
        "Claude 3 Haiku (Anthropic)": {
            "id": "anthropic.claude-3-haiku-20240307-v1:0",
            "family": "anthropic",
            "description": "Fast, efficient model with good reasoning at lower cost"
        },
        "Llama 3 8B Instruct (Meta)": {
            "id": "meta.llama3-8b-instruct-v1:0",
            "family": "meta",
            "description": "Open model with good performance for text generation"
        },
        "Titan Text G1 - Express (Amazon)": {
            "id": "amazon.titan-text-express-v1",
            "family": "amazon",
            "description": "Amazon's general purpose text model"
        }
    }

    # Then use this code to select and display model information:
    selected_model_name = st.sidebar.selectbox(
        "Select Bedrock Model",
        options=list(bedrock_model_options.keys()),
        index=0  # Default to first model
    )
    selected_model = bedrock_model_options[selected_model_name]
    model_config_ui["bedrock_model_id"] = selected_model["id"]
    model_config_ui["bedrock_model_family"] = selected_model["family"]

    st.sidebar.markdown(f"**Model Info:** {selected_model['description']}")

elif model_provider == "hybrid":
    st.sidebar.info("Hybrid mode will intelligently select between Ollama and Bedrock models based on task complexity and cost optimization.")
    
    # Configure both model providers
    # Ollama config
    with st.sidebar.expander("Ollama Configuration"):
        model_config_ui["ollama_base_url"] = st.text_input(
            "Ollama Base URL",
            value=config.get("ollama_base_url", "http://localhost:11434")
        )
        available_ollama_models = get_ollama_models(model_config_ui["ollama_base_url"])
        default_ollama_model_name = config.get("default_ollama_model", "llama3")
        
        if available_ollama_models:
            default_index = 0
            if default_ollama_model_name in available_ollama_models:
                try:
                    default_index = available_ollama_models.index(str(default_ollama_model_name))
                except ValueError:
                    pass
            selected_ollama_model = st.selectbox(
                "Select Ollama Model",
                options=available_ollama_models,
                index=default_index
            )
            model_config_ui["ollama_model_id"] = selected_ollama_model
        else:
            st.info("No Ollama models found. Please enter Model ID manually.")
            model_config_ui["ollama_model_id"] = st.text_input(
                "Ollama Model ID (Manual)",
                value=str(default_ollama_model_name)
            )
    
    # Bedrock config
    with st.sidebar.expander("Bedrock Configuration"):
        model_config_ui["aws_region"] = st.text_input(
            "AWS Region", 
            value=config.get("aws_region", "us-east-1")
        )
        bedrock_model_options = {
            "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "Llama 3 8B Instruct (Meta)": "meta.llama3-8b-instruct-v1:0",
            "Titan Text G1 - Express (Amazon)": "amazon.titan-text-express-v1",
        }
        selected_model_name = st.selectbox(
            "Select Bedrock Model",
            options=list(bedrock_model_options.keys()),
            index=0
        )
        model_config_ui["bedrock_model_id"] = bedrock_model_options[selected_model_name]
    
    # Hybrid strategy selection
    hybrid_strategies = {
        "Cost Optimized": "cost-optimized",
        "Quality Optimized": "quality-optimized",
        "Balanced": "balanced"
    }
    selected_strategy = st.sidebar.selectbox(
        "Select Hybrid Strategy",
        options=list(hybrid_strategies.keys()),
        index=0,
        help="Cost Optimized: Uses Ollama when possible. Quality Optimized: Uses Bedrock when possible. Balanced: Makes the best choice for each task."
    )
    model_config_ui["hybrid_strategy"] = hybrid_strategies[selected_strategy]

# Diagram Generation Method
diagram_method = st.sidebar.radio(
    "Select Diagram Generation Method",
    ("Mermaid", "AWS MCP"),
    index=0,
    key="diagram_toggle",
    help="Choose 'Mermaid' for embedded script diagrams. Choose 'AWS MCP' to use a configured AWS Diagram MCP Server (requires URL)."
)

diagram_config_ui = {"method": diagram_method.lower()}
if diagram_method == "AWS MCP":
    diagram_config_ui["mcp_server_url"] = st.sidebar.text_input(
        "AWS Diagram MCP Server URL",
        value=config.get("mcp_server_url", ""),
        help="Enter the full URL of your running AWS Diagram MCP Server instance."
    )
    if not diagram_config_ui.get("mcp_server_url"):
        st.sidebar.warning("AWS MCP Server URL is required for this option.")

# External Context Documents (RAG Integration)
st.sidebar.markdown("---")
st.sidebar.header("External Context (Optional)")

uploaded_files = st.sidebar.file_uploader(
    "Upload reference documents", 
    accept_multiple_files=True,
    help="Upload existing documentation to provide additional context for generation."
)

# Process uploaded files
context_documents = []
if uploaded_files:
    for file in uploaded_files:
        try:
            content = file.read().decode('utf-8')
            context_documents.append({
                'filename': file.name,
                'content': content
            })
            st.sidebar.success(f"Loaded: {file.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to read {file.name}: {e}")

# --- Main Area ---
st.header("1. Define Requirements")
user_requirements = st.text_area(
    "Enter the high-level requirements for your system:",
    height=150,
    placeholder="e.g., Build a scalable e-commerce platform with user authentication, product catalog, shopping cart, and order processing."
)

# Template selection
st.header("2. Select or Create Template")

templates = load_templates()
template_option = st.selectbox(
    "Select a template:",
    options=["default"] + [name for name in templates.keys() if name != "default"],
    index=0
)

# Load the selected template
template_content = templates[template_option]

# Allow editing
template_content = st.text_area(
    "Edit the HLD Template (Markdown with `## Sections` and `<!-- DIAGRAM: type=... description=\"...\" -->` placeholders):",
    value=template_content,
    height=400
)

# Option to save as a new template
with st.expander("Save as New Template"):
    new_template_name = st.text_input("Template Name:")
    if st.button("Save Template") and new_template_name:
        templates_dir = Path("./templates")
        templates_dir.mkdir(exist_ok=True)
        
        filename = f"{new_template_name.lower().replace(' ', '_')}.md"
        with open(templates_dir / filename, "w", encoding="utf-8") as f:
            f.write(template_content)
        
        st.success(f"Template saved as {filename}")

st.header("3. Generate HLD")

if st.button("Generate Document", key="generate_hld_button"):
    if not user_requirements:
        st.error("Please enter the system requirements.")
    elif diagram_method == "AWS MCP" and not diagram_config_ui.get("mcp_server_url"):
        st.error("Please enter the AWS Diagram MCP Server URL when using the AWS MCP option.")
    elif model_provider == "ollama" and not model_config_ui.get("ollama_model_id"):
        st.error("Please select or enter an Ollama Model ID.")
    else:
        # Create progress indicators
        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        # Initial status update
        status_container.info("Initializing generation process...")
        progress_container.text("Preparing to generate HLD...")
        
        with st.spinner("Generating HLD... This may take a while..."):
            try:
                final_model_config = {**config, **model_config_ui}
                final_diagram_config = {**config, **diagram_config_ui}

                initial_state_input = {
                    "user_requirements": user_requirements,
                    "model_provider": model_provider,
                    "model_config": final_model_config,
                    "diagram_method": diagram_method.lower(),
                    "diagram_config": final_diagram_config,
                    "template_content": template_content,
                    "parsed_sections": {}, 
                    "generation_context": [],
                    "generated_sections": {},
                    "final_hld": None,
                    "error_message": None,
                }
                
                # Add context documents if available
                if context_documents:
                    initial_state_input["context_documents"] = context_documents

                total_start_time = time.time()
                print("\n--- Starting HLD Generation ---")
                
                # Import the direct executor
                from core.direct_executor import DirectHLDExecutor
                
                # Create and use the direct executor instead of LangGraph
                executor = DirectHLDExecutor()
                final_result_state = None
                
                # Process the generator results
                for result in executor.execute(initial_state_input):
                    # Update progress based on result status
                    if result['status'] == 'in_progress':
                        # Update progress indicators
                        current_section = result.get('current_section', '')
                        completed = result.get('completed_sections', 0)
                        total = result.get('total_sections', 1)  # Avoid division by zero
                        
                        # Update status
                        status_container.info(f"Generating section: {current_section}")
                        progress_container.text(f"Progress: {completed}/{total} sections")
                        
                        # Update progress bar
                        progress_value = completed / total if total > 0 else 0
                        progress_bar.progress(progress_value)
                    elif result['status'] == 'success':
                        # Final result
                        final_result_state = result['state']
                        status_container.success("HLD generation completed successfully!")
                        progress_container.text(f"Completed: {result.get('completed_sections', 0)}/{result.get('total_sections', 0)} sections")
                        progress_bar.progress(1.0)
                    else:
                        # Error or failure
                        status_container.error("Generation process failed or produced incomplete results.")
                        final_result_state = result.get('state', {})
                
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                print(f"--- Finished HLD Generation ---")
                print(f"Total time taken: {total_duration:.2f} seconds")

                st.header("Generated High-Level Design")
                
                if final_result_state and 'final_hld' in final_result_state and final_result_state['final_hld']:
                    # Display the HLD in the UI
                    st.markdown(final_result_state["final_hld"], unsafe_allow_html=True)
                    
                    # Save to markdown file
                    output_file_path = save_markdown_to_file(
                        final_result_state["final_hld"], 
                        user_requirements
                    )
                    
                    # Provide download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as Markdown",
                            data=final_result_state["final_hld"],
                            file_name=os.path.basename(output_file_path),
                            mime="text/markdown",
                        )
                    
                    with col2:
                        st.success(f"HLD saved to: {output_file_path}")
                elif final_result_state and final_result_state.get("error_message"):
                    st.error(f"Generation failed: {final_result_state['error_message']}")
                elif final_result_state and 'generated_sections' in final_result_state and final_result_state['generated_sections']:
                    # We have generated sections but final_hld wasn't set - try to compile manually
                    st.warning("Final document compilation had issues. Showing manually compiled results...")
                    
                    # Do an emergency manual compilation
                    manual_parts = []
                    for title, content in final_result_state['generated_sections'].items():
                        manual_parts.append(f"## {title}\n")
                        manual_parts.append(content)
                        manual_parts.append("\n\n")
                    
                    manual_hld = "".join(manual_parts)
                    
                    # Display and save the manual compilation
                    st.markdown(manual_hld, unsafe_allow_html=True)
                    output_file_path = save_markdown_to_file(manual_hld, user_requirements)
                    
                    # Download button for manual compilation
                    st.download_button(
                        label="Download as Markdown",
                        data=manual_hld,
                        file_name=os.path.basename(output_file_path),
                        mime="text/markdown",
                    )
                    
                    st.success(f"Emergency compiled HLD saved to: {output_file_path}")
                else:
                    st.error("Generation process completed but no content was produced.")
                    
                    # Show debug info if available
                    if final_result_state:
                        with st.expander("Debug Information"):
                            st.write("State keys:", list(final_result_state.keys()))
                            
                            if 'parsed_sections' in final_result_state:
                                st.write("Template had these sections:", list(final_result_state['parsed_sections'].keys()))
                            
                            if 'generated_sections' in final_result_state:
                                st.write("Generated content for these sections:", list(final_result_state['generated_sections'].keys()))

            except Exception as e:
                st.error(f"A critical error occurred during the HLD generation process: {type(e).__name__} - {e}")
                st.code(traceback.format_exc())