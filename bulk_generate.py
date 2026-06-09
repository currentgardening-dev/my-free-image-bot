import os
import sys
import time
import re
from huggingface_hub import InferenceClient

sys.stdout.reconfigure(encoding='utf-8')

def bulk_generate():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ Error: Missing HF_TOKEN.")
        sys.exit(1)

    if not os.path.exists("prompts.txt"):
        print("❌ Error: 'prompts.txt' file not found.")
        sys.exit(1)

    with open("prompts.txt", "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    total_prompts = len(prompts)
    if total_prompts == 0:
        print("⚠️ Warning: 'prompts.txt' is empty.")
        sys.exit(0)

    print(f"📋 Found {total_prompts} prompts in prompts.txt. Starting bulk generation...")

    output_dir = "bulk_images"
    os.makedirs(output_dir, exist_ok=True)

    client = InferenceClient(api_key=hf_token)

    for index, prompt in enumerate(prompts, start=1):
        print(f"\n🖼️ [{index}/{total_prompts}] Generating: '{prompt}'...")
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:30]).lower()
        output_file = os.path.join(output_dir, f"{index}_{clean_name}.png")

        try:
            image = client.text_to_image(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell"
            )
            image.save(output_file)
            print(f"✅ Saved to: {output_file}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed: {e}")
            time.sleep(5)

    print(f"\n🎉 Bulk generation complete! All images are saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    bulk_generate()
