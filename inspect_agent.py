
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization

def inspect_ghost_agent():
    # Load the scenario directly
    scenario_path = Path("data/0007df76-c9a2-47aa-83bf-3b2b414109c9/scenario_0007df76-c9a2-47aa-83bf-3b2b414109c9.parquet")
    print(f"Loading from: {scenario_path}")
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    
    target_id = "79438"
    
    found = False
    for track in scenario.tracks:
        if track.track_id == target_id:
            found = True
            print(f"=== FOUND AGENT {target_id} ===")
            print(f"Type: {track.object_type}")
            print(f"Category: {track.category}")
            print(f"Total States: {len(track.object_states)}")
            
            # Print first few states
            for i, state in enumerate(track.object_states[:10]):
                print(f"Index {i} (Time={state.timestep}): Pos={state.position}, Heading={state.heading}")
            break
            
    if not found:
        print(f"Agent {target_id} NOT FOUND in scenario tracks!")
        
        # Check AV id
        print(f"Scenario ID: {scenario.scenario_id}")
        print(f"All Track IDs match? Checking first 5...")
        for t in scenario.tracks[:5]:
             print(f" - {t.track_id}")

if __name__ == "__main__":
    inspect_ghost_agent()
