import os
import base64
import csv
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_image(image_path, semaphore):
    """Process a single image with concurrency control."""
    async with semaphore:
        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Generate caption
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image factually in one sentence, focusing on objects and actions."},

                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            return (os.path.basename(image_path), response.choices[0].message.content)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return (os.path.basename(image_path), None)

async def main():
    image_dir = "./data/eval6/empathy_dataset/resized"
    save_path = "simple_captions_batched.csv"
    
    # Limit concurrency to 5 requests at a time
    semaphore = asyncio.Semaphore(5)
    
    # Gather all image paths
    image_paths = [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Initialize CSV and write header
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Caption"])
    
    # Open CSV in append mode for incremental writes
    with open(save_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        buffer = []
        
        # Process tasks as they complete
        tasks = [process_image(path, semaphore) for path in image_paths]
        for future in asyncio.as_completed(tasks):
            filename, caption = await future
            if caption:
                buffer.append((filename, caption))
                print(f"Processed: {filename}")
                
                # Save every 5 images
                if len(buffer) >= 5:
                    writer.writerows(buffer)
                    file.flush()  # Force-write to disk
                    buffer.clear()
                    print(f"Saved batch of 5 captions to {save_path}")
        
        # Save remaining captions (if any)
        if buffer:
            writer.writerows(buffer)
            file.flush()
            print(f"Saved final batch of {len(buffer)} captions")

if __name__ == "__main__":
    asyncio.run(main())