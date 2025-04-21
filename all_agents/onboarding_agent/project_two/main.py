import sys
import os.path

# Add the all_agents directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
all_agents_dir = os.path.join(parent_dir, "all_agents")
sys.path.append(all_agents_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(parent_dir)
from onboarding_agent.project_two.utils.onboarding import create_graph

graph = create_graph()
# graph = create_graph()
# Add the parent directory (all_agents) to the Python path


# from onboarding_agent.project_two.utils.onboarding import create_graph

# graph = create_graph()
