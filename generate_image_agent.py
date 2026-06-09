import os
import re
import sys
import io
from huggingface_hub import InferenceClient
from github import Github

def main():
    # 1. Read configurations
    hf_token = os.getenv("HF_TOKEN") 
    github_token = os.getenv("GITHUB_TOKEN")
    issue_num = os.getenv("ISSUE_NUMBER")
    issue_body = os.getenv("ISSUE_BODY")
    issue_title = os.getenv("ISSUE_TITLE")
    repo_name = os.getenv("REPO_NAME")

    if not hf_token:
        print("❌ Error: Missing HF_TOKEN environment variable (Hugging Face Free Token).")
        sys.exit(1)

    if not all([github_token, issue_num, issue_body, repo_name]):
        print("❌ Error: Missing required GitHub variables. This script runs inside GitHub Actions.")
        sys.exit(1)

    # 2. Extract prompt
    match = re.search(r"\[PROMPT:\s*(.*?)\]", issue_body, re.IGNORECASE | re.DOTALL)
    if match:
        prompt = match.group(1).strip()
    else:
        prompt = issue_title.strip()

    print(f"🎨 Found Prompt: '{prompt}'")

    print("🔌 Contacting free Hugging Face image generator...")
    try:
        # 3. Initialize the Hugging Face client
        client = InferenceClient(api_key=hf_token)
        
        # 4. Generate the image using the high-quality Flux.1-schnell model
        image = client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )
        
        # Save image to bytes so we can commit it to GitHub
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        print("✅ Image generated successfully!")
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        post_issue_comment(github_token, repo_name, int(issue_num), f"❌ Free image generation failed: {e}")
        sys.exit(1)

    # 5. Save and commit image to your GitHub repository
    git_client = Github(github_token)
    repo = git_client.get_repo(repo_name)

    clean_prompt = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:30]).lower()
    file_path = f"assets/generated/{clean_prompt}_{issue_num}.png"
    commit_message = f"chore: auto-generate image for issue #{issue_num} [skip ci]"

    print(f"💾 Saving image to your repository path: '{file_path}'...")
    try:
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, commit_message, image_bytes, contents.sha)
            print("🔄 Existing file updated.")
        except Exception:
            repo.create_file(file_path, commit_message, image_bytes)
            print("🆕 New file created.")

        raw_url = f"https://raw.githubusercontent.com/{repo_name}/main/{file_path}"
        markdown_comment = (
            f"### 🎉 Image Generated Successfully (Free Plan)!\n\n"
            f"**Prompt Used:** *{prompt}*\n"
            f"**Saved to:** `{file_path}`\n\n"
            f"![Generated Image]({raw_url})"
        )
        post_issue_comment(github_token, repo_name, int(issue_num), markdown_comment)

        # Remove the trigger label
        issue = repo.get_issue(int(issue_num))
        issue.remove_from_labels("generate-image")
        print("🚀 Success!")

    except Exception as e:
        print(f"❌ Failed to upload image back to GitHub: {e}")
        post_issue_comment(github_token, repo_name, int(issue_num), f"❌ Failed to commit image to repo: {e}")
        sys.exit(1)

def post_issue_comment(token, repo_name, issue_num, text):
    try:
        git_client = Github(token)
        repo = git_client.get_repo(repo_name)
        issue = repo.get_issue(issue_num)
        issue.create_comment(text)
    except Exception as e:
        print(f"⚠️ Failed to post comment on issue: {e}")

if __name__ == "__main__":
    main()
