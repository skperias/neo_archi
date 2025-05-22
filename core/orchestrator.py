# /core/orchestrator.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional, Callable, Any
from .models import ModelFactory, BaseModelConnector
from .diagrams import DiagramStrategy, BaseDiagramGenerator
from .templates import parse_template, find_diagram_placeholders
import time
import re
import traceback
import concurrent.futures
import copy

class HLDWorkflowState(TypedDict):
    user_requirements: str
    model_provider: str  # 'ollama', 'bedrock', or 'hybrid'
    model_config: dict   # Specific model ID, API keys/URL etc.
    diagram_method: str  # 'mermaid' or 'mcp'
    diagram_config: dict # MCP URL etc.
    template_content: str
    parsed_sections: Dict[str, str]  # Section title -> Raw template content from template
    generation_context: List[str]    # History of generated content for context
    generated_sections: Dict[str, str]  # Section title -> Generated content + diagrams
    final_hld: Optional[str]
    error_message: Optional[str]
    current_section: Optional[str]   # Currently processing section (for progress tracking)
    context_documents: Optional[List[Dict[str, str]]]  # External context documents
    generation_statistics: Optional[Dict[str, Any]]  # Statistics about the generation process

# --- Helper functions for diagram generation ---

def generate_diagrams_parallel(diagram_placeholders, diagram_generator, next_section_title, user_requirements, generated_text):
    """Generate diagrams in parallel using ThreadPoolExecutor."""
    diagram_results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for placeholder in diagram_placeholders:
            # Submit diagram generation task to the executor
            future = executor.submit(
                generate_single_diagram,
                diagram_generator,
                placeholder,
                next_section_title,
                user_requirements,
                generated_text
            )
            futures.append((future, placeholder))
        
        # Collect results as they complete
        for future, placeholder in futures:
            try:
                diagram_code = future.result()
                diagram_results.append((placeholder, diagram_code, None))
            except Exception as e:
                diagram_results.append((placeholder, None, e))
    
    return diagram_results

def generate_single_diagram(diagram_generator, placeholder, next_section_title, user_requirements, generated_text):
    """Generate a single diagram."""
    diagram_context = f"Section: {next_section_title}\nRequirements: {user_requirements}\nSection Content (so far):\n{generated_text}"
    return diagram_generator.generate_diagram(
        context=diagram_context,
        diagram_type=placeholder['params']['type'],
        description=placeholder['params']['description']
    )

# --- LangGraph Nodes ---

def setup_state(state: HLDWorkflowState) -> HLDWorkflowState:
    """Initializes connectors and parses the template."""
    try:
        state['parsed_sections'] = parse_template(state['template_content'])
        state['generated_sections'] = {}  # Initialize as empty dict
        state['generation_context'] = [f"User Requirements: {state['user_requirements']}"]
        
        # Add context from external documents if available
        if state.get('context_documents'):
            context_summary = []
            for doc in state.get('context_documents'):
                context_summary.append(f"External Document: {doc['filename']}")
            
            if context_summary:
                state['generation_context'].append("External context documents available: " + ", ".join(context_summary))
        
        # Initialize statistics tracking
        state['generation_statistics'] = {
            "ollama_calls": 0,
            "bedrock_calls": 0,
            "section_times": {},
            "total_estimated_cost": 0.0
        }
        
        print("State setup complete.")
        return state
    except Exception as e:
        error_msg = f"Setup failed: {type(e).__name__} - {e}\n{traceback.format_exc()}"
        state['error_message'] = error_msg
        print(f"ERROR: {error_msg}")
        return state

def generate_section_content(state: HLDWorkflowState) -> HLDWorkflowState:
    """Generates content for the next section using the selected LLM."""
    section_start_time = time.time()
    next_section_title = None  # Initialize here for broader scope in error handling

    try:
        model_provider = state['model_provider']
        model_config = state['model_config']
        llm_connector = ModelFactory.get_connector(model_provider, model_config)

        processed_sections = list(state['generated_sections'].keys())
        all_sections = list(state['parsed_sections'].keys())

        # Find the next section title
        for title_key in all_sections:
            if title_key not in processed_sections:
                next_section_title = title_key
                break

        if not next_section_title:
            return state

        # Update current section for progress tracking
        state['current_section'] = next_section_title
        
        print(f"Generating content for section: '{next_section_title}'...")
        section_template_content_hint = state['parsed_sections'][next_section_title]
        context_str = "\n".join(state['generation_context'])
        diagram_placeholders = find_diagram_placeholders(section_template_content_hint)
        diagram_info = "\n".join([f"- {p['params']['type']} diagram: {p['params']['description']}" for p in diagram_placeholders])

        # Determine task type based on section content
        task_type = "general"
        if "architecture" in next_section_title.lower() or "design" in next_section_title.lower():
            task_type = "architecture_reasoning"
        elif "security" in next_section_title.lower():
            task_type = "security_analysis"
        elif "component" in next_section_title.lower() or "module" in next_section_title.lower():
            task_type = "component_design"
        elif "introduction" in next_section_title.lower() or "overview" in next_section_title.lower():
            task_type = "summary"

        # Add context from external documents if available
        external_context = ""
        if state.get('context_documents'):
            external_context = "Relevant external documentation:\n\n"
            for doc in state.get('context_documents', []):
                # For each document, include a meaningful excerpt
                # Try to find relevant content based on section title
                relevant_excerpt = extract_relevant_content(doc['content'], next_section_title, 1000)
                external_context += f"Document: {doc['filename']}\n{relevant_excerpt}\n\n"

        prompt = f"""
        You are an AI assistant generating a High-Level Design (HLD) document.
        You are currently generating the content for the section titled: '{next_section_title}'.
        **IMPORTANT: Do NOT repeat the section title (e.g., '## {next_section_title}') in your output.**
        Generate only the body/content for this section based on the instructions and context.

        Use the following context from previous sections and user requirements:
        --- CONTEXT ---
        {context_str}
        --- END CONTEXT ---

        The user requirements are: {state['user_requirements']}
        
        {external_context if external_context else ""}

        The content for this section should be guided by the following instruction/placeholders from the template:
        Template instruction/placeholders: "{section_template_content_hint}"
        """
        if diagram_info:
            prompt += f"\nThis section will include the following diagrams (you should only write the textual content for the section; the diagrams will be added separately based on these descriptions):\n{diagram_info}"

        # For hybrid connector, pass the task_type
        generation_start = time.time()
        if model_provider == "hybrid" and hasattr(llm_connector, "generate_content") and "task_type" in llm_connector.generate_content.__code__.co_varnames:
            generated_text = llm_connector.generate_content(prompt, task_type=task_type)
            print(f"Using hybrid connector with task_type: {task_type}")
            
            # Update statistics
            if hasattr(llm_connector, "get_usage_stats"):
                usage_stats = llm_connector.get_usage_stats()
                state['generation_statistics']["ollama_calls"] = usage_stats['ollama_calls']
                state['generation_statistics']["bedrock_calls"] = usage_stats['bedrock_calls']
                state['generation_statistics']["total_estimated_cost"] = usage_stats['total_estimated_cost']
        else:
            generated_text = llm_connector.generate_content(prompt)
            
            # Update basic statistics
            if model_provider == "ollama":
                state['generation_statistics']["ollama_calls"] += 1
            elif model_provider == "bedrock":
                state['generation_statistics']["bedrock_calls"] += 1
                
        generation_end = time.time()
        generation_time = generation_end - generation_start
        
        # Track section time
        state['generation_statistics']["section_times"][next_section_title] = {
            "content_generation": generation_time
        }

        # --- Strip the section title if the LLM still includes it ---
        lines = generated_text.strip().split('\n')
        if lines:
            first_line_stripped = lines[0].strip()
            normalized_first_line = re.sub(r'^#+\s*', '', first_line_stripped).strip()
            normalized_section_title = re.sub(r'^#+\s*', '', next_section_title).strip()

            if normalized_first_line.lower() == normalized_section_title.lower():
                print(f"INFO: LLM included section title ('{first_line_stripped}') in its output for section '{next_section_title}'. Stripping it.")
                generated_text = "\n".join(lines[1:]).strip()
        # --- End title stripping ---

        section_output_parts = [generated_text]  # Start with LLM text
        diagram_method = state['diagram_method']
        diagram_config = state['diagram_config']
        diagram_strategy = DiagramStrategy()
        
        # For hybrid connector, we might want to use a different model for diagrams
        if model_provider == "hybrid" and hasattr(llm_connector, "select_model"):
            # Get a specialized connector for diagram generation
            diagram_connector, model_name = llm_connector.select_model(prompt, task_type="diagram_generation")
            print(f"Selected {model_name} specifically for diagram generation")
            llm_for_mermaid = diagram_connector if diagram_method == 'mermaid' else None
        else:
            llm_for_mermaid = llm_connector if diagram_method == 'mermaid' else None

        # Get the diagram generator
        diagram_generator = diagram_strategy.get_generator(diagram_method, diagram_config, llm_for_mermaid)

        # Use parallel diagram generation if there are multiple diagrams
        if len(diagram_placeholders) > 1:
            print(f"  Generating {len(diagram_placeholders)} diagrams in parallel...")
            diagram_start_time = time.time()
            
            # Generate diagrams in parallel
            diagram_results = generate_diagrams_parallel(
                diagram_placeholders, 
                diagram_generator,
                next_section_title, 
                state['user_requirements'], 
                generated_text
            )
            
            # Process results
            for placeholder, diagram_code, error in diagram_results:
                if diagram_code:
                    section_output_parts.append(f"\n\n{diagram_code}\n\n")
                    print(f"    Diagram '{placeholder['params']['description']}' generated successfully.")
                elif error:
                    if isinstance(error, ConnectionError) and diagram_method == 'mcp':
                        # Try Mermaid fallback for MCP failures
                        try:
                            print(f"    MCP connection failed for '{placeholder['params']['description']}'. Attempting Mermaid fallback.")
                            mermaid_gen = DiagramStrategy.get_generator('mermaid', diagram_config, llm_connector)
                            diagram_context = f"Section: {next_section_title}\nRequirements: {state['user_requirements']}\nSection Content (so far):\n{generated_text}"
                            fallback_code = mermaid_gen.generate_diagram(
                                context=diagram_context,
                                diagram_type=placeholder['params']['type'],
                                description=placeholder['params']['description']
                            )
                            section_output_parts.append(f"\n\n{fallback_code}\n\n")
                            print(f"    Diagram '{placeholder['params']['description']}' (Mermaid fallback) generated successfully.")
                        except Exception as fallback_e:
                            err_msg = f"Mermaid fallback failed for '{placeholder['params']['description']}': {type(fallback_e).__name__} - {fallback_e}"
                            print(f"  {err_msg}")
                            section_output_parts.append(f"\n\n<!-- Diagram generation failed: {err_msg} -->\n")
                    else:
                        err_msg = f"Diagram generation failed for '{placeholder['params']['description']}': {type(error).__name__} - {error}"
                        print(f"  {err_msg}")
                        section_output_parts.append(f"\n\n<!-- Diagram generation failed: {err_msg} -->\n")
            
            diagram_end_time = time.time()
            state['generation_statistics']["section_times"][next_section_title]["diagram_generation"] = diagram_end_time - diagram_start_time
            print(f"  All diagrams generated in {diagram_end_time - diagram_start_time:.2f} seconds.")
            
        else:
            # Process diagrams sequentially for a single diagram (simpler)
            for placeholder in diagram_placeholders:
                diag_start_time = time.time()
                print(f"  Generating diagram: '{placeholder['params']['description']}' using {diagram_method}...")
                try:
                    # Context for diagram should be the generated text for the section so far, plus overall reqs
                    diagram_context = f"Section: {next_section_title}\nRequirements: {state['user_requirements']}\nSection Content (so far):\n{generated_text}"
                    diagram_code = diagram_generator.generate_diagram(
                        context=diagram_context,
                        diagram_type=placeholder['params']['type'],
                        description=placeholder['params']['description']
                    )
                    section_output_parts.append(f"\n\n{diagram_code}\n\n")  # Append diagram
                    diag_end_time = time.time()
                    print(f"    Diagram '{placeholder['params']['description']}' generated in {diag_end_time - diag_start_time:.2f} seconds.")
                except ConnectionError as mcp_error:
                    print(f"  MCP connection failed for '{placeholder['params']['description']}': {mcp_error}. Attempting Mermaid fallback.")
                    try:
                        mermaid_gen = DiagramStrategy.get_generator('mermaid', diagram_config, llm_connector)
                        diagram_context = f"Section: {next_section_title}\nRequirements: {state['user_requirements']}\nSection Content (so far):\n{generated_text}"
                        diagram_code = mermaid_gen.generate_diagram(
                            context=diagram_context,
                            diagram_type=placeholder['params']['type'],
                            description=placeholder['params']['description']
                        )
                        section_output_parts.append(f"\n\n{diagram_code}\n\n")
                        diag_end_time = time.time()
                        print(f"    Diagram '{placeholder['params']['description']}' (Mermaid fallback) generated in {diag_end_time - diag_start_time:.2f} seconds.")
                    except Exception as fallback_e:
                        err_msg = f"Mermaid fallback failed for '{placeholder['params']['description']}': {type(fallback_e).__name__} - {fallback_e}"
                        print(f"  {err_msg}")
                        section_output_parts.append(f"\n\n<!-- Diagram generation failed: {err_msg} -->\n")
                except Exception as diag_e:
                    err_msg = f"Diagram generation failed for '{placeholder['params']['description']}': {type(diag_e).__name__} - {diag_e}"
                    print(f"  {err_msg}")
                    section_output_parts.append(f"\n\n<!-- Diagram generation failed: {err_msg} -->\n")
                
                # Track diagram generation time in stats
                diag_end_time = time.time()
                if "diagram_generation" not in state['generation_statistics']["section_times"][next_section_title]:
                    state['generation_statistics']["section_times"][next_section_title]["diagram_generation"] = 0
                
                state['generation_statistics']["section_times"][next_section_title]["diagram_generation"] += diag_end_time - diag_start_time
        
        final_section_content = "".join(section_output_parts)
        state['generated_sections'][next_section_title] = final_section_content
        state['generation_context'].append(f"## {next_section_title}\n{final_section_content}")
        section_end_time = time.time()
        section_duration = section_end_time - section_start_time
        print(f"Finished section: '{next_section_title}' in {section_duration:.2f} seconds.")
        
        # Track total section time
        state['generation_statistics']["section_times"][next_section_title]["total"] = section_duration
        
        # If using hybrid connector, log stats
        if model_provider == "hybrid" and hasattr(llm_connector, "get_usage_stats"):
            usage_stats = llm_connector.get_usage_stats()
            print(f"Current usage statistics:")
            print(f"  - Ollama calls: {usage_stats['ollama_calls']}")
            print(f"  - Bedrock calls: {usage_stats['bedrock_calls']}")
            print(f"  - Total estimated cost: ${usage_stats['total_estimated_cost']:.6f}")
            
        return state

    except Exception as e:
        error_msg = f"Content generation failed for section {next_section_title or 'Unknown'}: {type(e).__name__} - {e}\n{traceback.format_exc()}"
        state['error_message'] = error_msg
        print(f"ERROR: {error_msg}")
        section_end_time = time.time()
        section_duration = section_end_time - section_start_time
        print(f"Section '{next_section_title or 'Unknown'}' failed after {section_duration:.2f} seconds.")
        return state

def compile_document(state: HLDWorkflowState) -> HLDWorkflowState:
    """Combines all generated sections into the final HLD."""
    print("Compiling final document...")
    # Clear current section for progress reporting
    state['current_section'] = "Compiling document"
    
    if state.get('error_message') and not state.get('generated_sections'):
        state['final_hld'] = f"## Generation Failed Early\n\nError: {state['error_message']}"
        print("Document compilation finished (with early error state).")
        return state

    # Debug information
    print(f"Compiling document with {len(state.get('parsed_sections', {}))} template sections")
    print(f"Content generated for {len(state.get('generated_sections', {}))} sections")
    
    # List all section titles for debugging
    print("Template sections:", list(state.get('parsed_sections', {}).keys()))
    print("Generated sections:", list(state.get('generated_sections', {}).keys()))

    final_doc_parts = []
    # Ensure sections are added in the order defined by the template (parsed_sections keys)
    for title in state.get('parsed_sections', {}).keys():
        final_doc_parts.append(f"## {title}\n")  # Add the section title
        if title in state.get('generated_sections', {}):
            content = state['generated_sections'][title]
            final_doc_parts.append(content)
            print(f"Added content for section '{title}' ({len(content)} chars)")
        else:
            # Handle sections that might have been skipped due to errors or if generation stopped early
            final_doc_parts.append("\n<!-- Content generation skipped or failed for this section. -->")
            print(f"No content for section '{title}'")
        final_doc_parts.append("\n")  # Add a newline for spacing between sections

    # Create the final HLD content
    final_content = "\n".join(final_doc_parts).strip()
    print(f"Final document compiled with total length: {len(final_content)} chars")
    
    # Explicit assignment to the state dict
    state['final_hld'] = final_content
    
    # Double-check final_hld was set
    if 'final_hld' in state and state['final_hld']:
        print("final_hld successfully set in state")
    else:
        print("WARNING: final_hld is not set properly!")
    
    if state.get('error_message'):
        state['final_hld'] += f"\n\n## Generation Issues Encountered\n\nAn error occurred during the process: {state['error_message']}"

    # Add usage statistics for hybrid model
    model_provider = state.get('model_provider', '')
    if model_provider == "hybrid":
        if 'generation_statistics' in state and state['generation_statistics']:
            stats = state['generation_statistics']
            state['final_hld'] += f"\n\n## Generation Statistics\n\n"
            state['final_hld'] += f"- Ollama calls: {stats.get('ollama_calls', 0)}\n"
            state['final_hld'] += f"- Bedrock calls: {stats.get('bedrock_calls', 0)}\n"
            state['final_hld'] += f"- Total estimated cost: ${stats.get('total_estimated_cost', 0.0):.6f}\n"
            
            print(f"Final usage statistics:")
            print(f"  - Ollama calls: {stats.get('ollama_calls', 0)}")
            print(f"  - Bedrock calls: {stats.get('bedrock_calls', 0)}")
            print(f"  - Total estimated cost: ${stats.get('total_estimated_cost', 0.0):.6f}")

    print(f"Document compilation complete. State keys: {list(state.keys())}")
    return state

# --- LangGraph Workflow Definition ---

def should_continue_generation(state: HLDWorkflowState) -> str:
    """Determines if there are more sections to generate."""
    if state.get('error_message'):
        # If an error occurred, we might still want to compile what we have,
        # or we might want to stop entirely. For now, let's proceed to compile.
        print("Error detected during generation, proceeding to compilation.")
        return "compile"

    processed_sections_count = len(state['generated_sections'])
    total_sections_count = len(state['parsed_sections'])

    if processed_sections_count < total_sections_count:
        return "generate_section"
    else:
        print("All sections generated, proceeding to compilation.")
        return "compile"

def extract_relevant_content(document_content, section_title, max_chars=1000):
    """Extract content from document that seems relevant to the section being generated."""
    # Simplistic approach: look for sections with similar titles or keywords
    section_keywords = section_title.lower().split()
    
    # Try to find a section in the document with a similar title
    lines = document_content.split('\n')
    start_line = 0
    best_section_score = 0
    best_section_start = 0
    
    # Look for section headers in the document
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):  # Markdown header
            # Count how many keywords from the section title appear in this header
            score = sum(1 for keyword in section_keywords if keyword.lower() in line.lower())
            if score > best_section_score:
                best_section_score = score
                best_section_start = i
    
    # If we found a good match, extract content from that section
    if best_section_score > 0:
        # Extract from the matched section header
        content = []
        i = best_section_start
        
        # Include the header
        content.append(lines[i])
        i += 1
        
        # Add lines until we hit the next header or reach max length
        total_chars = len(lines[best_section_start])
        while i < len(lines) and total_chars < max_chars:
            if lines[i].strip().startswith('#') and i > best_section_start + 1:
                break  # Stop at the next header
            content.append(lines[i])
            total_chars += len(lines[i])
            i += 1
            
        return "\n".join(content)
    
    # If no good section match, look for paragraphs with relevant keywords
    relevant_paragraphs = []
    current_paragraph = []
    total_chars = 0
    
    for line in lines:
        if not line.strip():  # Empty line marks paragraph boundary
            if current_paragraph:
                para_text = "\n".join(current_paragraph)
                # Check if any keywords appear in this paragraph
                if any(keyword in para_text.lower() for keyword in section_keywords):
                    relevant_paragraphs.append(para_text)
                    total_chars += len(para_text) + 1  # +1 for newline
                
                if total_chars >= max_chars:
                    break
                    
                current_paragraph = []
        else:
            current_paragraph.append(line)
    
    # Don't forget the last paragraph
    if current_paragraph and total_chars < max_chars:
        para_text = "\n".join(current_paragraph)
        if any(keyword in para_text.lower() for keyword in section_keywords):
            relevant_paragraphs.append(para_text)
    
    # If we found relevant paragraphs, return them
    if relevant_paragraphs:
        return "\n\n".join(relevant_paragraphs)
    
    # If we couldn't find anything relevant, just return the beginning of the document
    return document_content[:max_chars] + "..."

def build_graph(progress_callback: Optional[Callable] = None):
    """Build the LangGraph workflow with optional progress tracking."""
    workflow = StateGraph(HLDWorkflowState)

    # Add nodes
    workflow.add_node("setup", setup_state)
    workflow.add_node("generate_section", generate_section_content)
    workflow.add_node("compile", compile_document)
    
    # Add progress tracking node if callback provided
    if progress_callback:
        workflow.add_node("track_progress", progress_callback)

    # Set entry point
    workflow.set_entry_point("setup")

    # Define edges
    if progress_callback:
        workflow.add_edge("setup", "track_progress")
        workflow.add_edge("track_progress", "generate_section")
        
        # Add conditional edges with progress tracking
        workflow.add_conditional_edges(
            "generate_section",
            should_continue_generation,
            {
                "generate_section": "track_progress",  # Loop back through progress tracker
                "compile": "compile"                  # Move to compile if done or error
            }
        )
    else:
        # Standard edges without progress tracking
        workflow.add_edge("setup", "generate_section")
        
        workflow.add_conditional_edges(
            "generate_section",
            should_continue_generation,
            {
                "generate_section": "generate_section",  # Loop back if more sections
                "compile": "compile"                    # Move to compile if done or error
            }
        )
    
    workflow.add_edge("compile", END)  # End after compilation

    app = workflow.compile()
    print("LangGraph workflow compiled.")
    return app