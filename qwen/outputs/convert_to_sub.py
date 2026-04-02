import json
import csv

json_file = "/root/Desktop/workspace/sw/05_Accident_cvpr/08_qwen3_8B/outputs/debug_results.json"
csv_file = "/root/Desktop/workspace/sw/05_Accident_cvpr/08_qwen3_8B/outputs/submission.csv"

with open(json_file, "r") as f:
    data = json.load(f)

with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["path", "accident_time", "center_x", "center_y", "type"])
    for row in data:
        raw_path = row["path"]
        path_without_prefix = raw_path[5:] if len(raw_path) > 5 and raw_path[4] == '_' else raw_path
        csv_path = f"videos/{path_without_prefix}"
        writer.writerow([csv_path, row["time"], row["center_x"], row["center_y"], row["type"]])

print(f"Generated {csv_file}")
