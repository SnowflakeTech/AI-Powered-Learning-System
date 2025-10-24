import shutil
import os
from sat_question_generator import generate_sat_items, save_as_json

def build_sat_pipeline(n=10):
    print("🚀 Generating new SAT questions and IRT parameters...")
    items, irt = generate_sat_items(n, start_id=1, seed=42)
    tmp_items = "data/items_generated.json"
    tmp_irt = "data/irt_params_generated.json"

    # 1️⃣ Lưu tạm
    save_as_json(items, irt, tmp_items, tmp_irt)
    print(f"✅ Generated files: {tmp_items}, {tmp_irt}")

    # 2️⃣ Copy sang file chính thức
    shutil.copy(tmp_items, "data/items.json")
    shutil.copy(tmp_irt, "data/irt_params.json")
    print("✅ Copied to data/items.json and data/irt_params.json")

    # 3️⃣ Xác nhận
    if os.path.exists("sat_ai_core.py"):
        print("🎯 You can now run: python sat_ai_system.py")
    else:
        print("⚠️ Warning: sat_ai_core.py not found in current directory.")

if __name__ == "__main__":
    build_sat_pipeline(20)
