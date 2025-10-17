import json
# Load Floorplan1_metadata.txt and filter objects to print out the ones with salientMaterials

with open("Floorplan1_metadata.json", "r") as f:
    metadata = json.load(f)

for obj in metadata["objects"]:
    if obj["objectType"] == "Vase":
        print("\n ############ \n")
        print(obj["objectId"], obj["moveable"])
        print("\n ############")
    if obj["salientMaterials"] and "Metal" in obj["salientMaterials"]:
        print(obj["objectId"], obj["salientMaterials"])
