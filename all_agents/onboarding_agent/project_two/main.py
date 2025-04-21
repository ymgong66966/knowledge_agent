import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# # Add the all_agents directory to the Python path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# all_agents_dir = os.path.join(parent_dir, "all_agents")
# all_agents_dir = os.path.join(parent_dir, "onboarding_agent")
# sys.path.append(all_agents_dir)
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# sys.path.append(parent_dir)
from onboarding_agent.project_two.utils.onboarding import create_graph

graph = create_graph()
# graph = create_graph()
# Add the parent directory (all_agents) to the Python path


# from onboarding_agent.project_two.utils.onboarding import create_graph

# graph = create_graph()
