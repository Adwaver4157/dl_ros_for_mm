import pickle

with open("dataset/sim/sim.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(len(loaded_data))
print(type(loaded_data[0]))
print(loaded_data[0].keys())
print(f"base_cmd size: {loaded_data[0]['base_cmd'].shape}")
print(f"arm_pose size: {loaded_data[0]['arm_pose'].shape}")
print(f"arm_action size: {loaded_data[0]['arm_action'].shape}")
