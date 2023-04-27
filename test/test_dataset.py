import pickle

with open("dataset/arm/arm.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(len(loaded_data))
print(type(loaded_data[0]))
print(loaded_data[0].keys())
print(f"head_images size: {loaded_data[0]['head_images'].shape}")
print(f"hand_images size: {loaded_data[0]['hand_images'].shape}")
print(f"joint_states size: {loaded_data[0]['joint_states'].shape}")
print(f"base_cmd size: {loaded_data[0]['base_cmd'].shape}")
print(f"arm_pose size: {loaded_data[0]['arm_pose'].shape}")
print(f"arm_action size: {loaded_data[0]['arm_action'].shape}")

print(f"arm_pose: {loaded_data[0]['arm_pose']}")
