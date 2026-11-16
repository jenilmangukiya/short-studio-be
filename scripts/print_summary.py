import json
from pathlib import Path

with open("storage/20fa862f-3b07-4f9d-9c1f-eb542d8cdb3a/complete_generation_summary.json") as f:
    data = json.load(f)

sb = data.get("storyboard", {}).get("storyboard", {})
scenes = sb.get("scenes", [])
beats = sb.get("beats", [])
images_items = data.get("images", {}).get("images", {}).get("items", [])

img_by_id = {it.get("id"): it for it in images_items}

print(f"Total Scenes: {len(scenes)}")
print(f"Total Beats: {len(beats)}")
print(f"Total Images: {len(images_items)}")

for idx, b in enumerate(beats, start=1):
    bid = b.get("beat_id")
    it = img_by_id.get(bid)
    
    eb = b.get("entity_bindings", {})
    cam = b.get("camera", {})
    cs = b.get("character_state")
    
    print(f"\n==================== BEAT {idx} (ID: {bid}) ====================")
    print("Scene:", b.get("scene_index"), "Order:", b.get("order"), "Est Dur:", b.get("duration_est_seconds"), "s")
    print("Narration:", repr(b.get("narration_slice")))
    print("Transition:", b.get("transition_from_previous"))
    print("Entities: char=" + str(eb.get("character_id")) + ", env=" + str(eb.get("environment_id")) + ", concept=" + str(eb.get("concept_id")) + ", objs=" + str(eb.get("object_ids")))
    print("Camera:", cam.get("shot_type"), "|", cam.get("camera_angle"), "| Focal:", cam.get("focal_point_description"))
    if cs:
        print("Character State: pose=" + repr(cs.get("pose_description")) + ", expr=" + repr(cs.get("expression_description")) + ", action=" + repr(cs.get("action_in_progress")))
    else:
        print("Character State: None")
    if it:
        print("Image File:", it.get("file"))
        print("Final Composed Prompt:", it.get("final_prompt"))
