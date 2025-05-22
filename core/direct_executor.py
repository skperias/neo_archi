# direct_executor.py
# Place this in your core directory

import copy
from typing import Dict, Any, List, Optional
import time

# Import the node functions from your orchestrator
from .orchestrator import (
    setup_state, 
    generate_section_content, 
    compile_document, 
    should_continue_generation,
    HLDWorkflowState
)

class DirectHLDExecutor:
    """
    A direct executor that bypasses LangGraph's state management issues.
    This executes the same workflow but handles the state directly.
    """
    
    def __init__(self, debug=True):
        self.debug = debug
        
    def _log(self, message):
        if self.debug:
            print(message)
    
    def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the HLD generation workflow directly."""
        # Make a fresh copy of the initial state to avoid reference issues
        state = copy.deepcopy(initial_state)
        
        # Step 1: Setup
        self._log("Starting HLD generation with direct executor...")
        self._log(f"Initial state keys: {list(state.keys())}")
        
        # Execute setup_state
        self._log("Executing setup phase...")
        state = setup_state(state)
        self._log(f"Setup complete. State has {len(state.get('parsed_sections', {}))} sections to generate.")
        
        # Step 2: Generate sections
        section_count = len(state.get('parsed_sections', {}))
        completed_sections = 0
        
        # Keep generating sections until done
        while True:
            # Check if we should continue or move to compilation
            next_action = should_continue_generation(state)
            if next_action == "compile":
                break
                
            # Generate the next section
            self._log(f"Generating section {completed_sections + 1}/{section_count}...")
            state = generate_section_content(state)
            completed_sections += 1
            
            # Update progress information if available
            if 'current_section' in state:
                self._log(f"Completed section: {state['current_section']}")
                
            # For status updates
            yield {
                'status': 'in_progress',
                'current_section': state.get('current_section', ''),
                'completed_sections': completed_sections,
                'total_sections': section_count,
                'state': None  # Don't include the full state in progress updates
            }
        
        # Step 3: Compilation
        self._log("All sections generated. Compiling document...")
        state = compile_document(state)
        
        # Verify final output
        if 'final_hld' in state and state['final_hld']:
            self._log(f"Document compilation successful - {len(state['final_hld'])} characters generated")
            result_status = 'success'
        else:
            self._log("WARNING: Document compilation completed but final_hld is not set!")
            result_status = 'failure'
        
        # Return the final state and status
        yield {
            'status': result_status,
            'state': state,
            'completed_sections': completed_sections,
            'total_sections': section_count
        }
