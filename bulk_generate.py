import os
import sys
import time
import re
import requests
import io
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

def generate_image_free(prompt, model_id, hf_token):
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt}
    
    for attempt in range(4):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)
            
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            elif response.status_code == 503:
                try:
                    err_data = response.json()
                    estimated_time = err_data.get("estimated_time", 20)
                except Exception:
                    estimated_time = 20
                print(f"⌛ Model is loading on Hugging Face... waiting {estimated_time:.1f}s (attempt {attempt+1}/4)...")
                time.sleep(estimated_time)
            else:
                raise Exception(f"HTTP {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            if attempt == 3:
                raise e
            print(f"⚠️ Network warning: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    raise Exception("Model failed to load after multiple attempts.")

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

    model_id = "black-forest-labs/FLUX.1-schnell"

    for index, prompt in enumerate(prompts, start=1):
        print(f"\n🖼️ [{index}/{total_prompts}] Generating: '{prompt}'...")
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:30]).lower()
        output_file = os.path.join(output_dir, f"{index}_{clean_name}.png")

        try:
            image = generate_image_free(prompt, model_id, hf_token)
            image.save(output_file)
            print(f"✅ Saved to: {output_file}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed: {e}")
            time.sleep(5)

    print(f"\n🎉 Bulk generation complete! All images are saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    bulk_generate()
