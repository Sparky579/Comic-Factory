import os
import json
import base64
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Gemini client: {e}")

    def update_api_key(self, new_key: str):
        self.api_key = new_key
        # Update .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        try:
            # Read existing lines
            lines = []
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    lines = f.readlines()
            
            # Update or add GOOGLE_API_KEY
            key_found = False
            new_lines = []
            for line in lines:
                if line.startswith("GOOGLE_API_KEY="):
                    new_lines.append(f"GOOGLE_API_KEY={new_key}\n")
                    key_found = True
                else:
                    new_lines.append(line)
            
            if not key_found:
                new_lines.append(f"GOOGLE_API_KEY={new_key}\n")
                
            with open(env_path, 'w') as f:
                f.writelines(new_lines)
            
            # Update internal client
            self.client = genai.Client(api_key=new_key)
            return True
        except Exception as e:
            print(f"Error saving API key: {e}")
            return False

    def _format_dialogue(self, text: str) -> str:
        """
        Formats dialogue text into: (Character Name) "Spoken Text".
        1. Extracts character name from prefix (e.g. "Doraemon:").
        2. Ensures spoken text is wrapped in quotes.
        3. If no name found, just returns quoted text.
        """
        if not text:
            return ""
            
        import re
        
        # Pattern to find Name: Content
        # ^(.*?)([:：])\s*(.*)$
        match = re.match(r'^(.*?)([:：])\s*(.*)$', text, re.DOTALL)
        
        name = ""
        content = text
        
        if match:
            name = match.group(1).strip()
            content = match.group(3).strip()
            
        # Clean quotes from content first (to avoid double quoting)
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")) or \
           (content.startswith('“') and content.endswith('”')):
            content = content[1:-1]
            
        # Format
        formatted_content = f'"{content}"'
        
        if name:
            return f"({name}) {formatted_content}"
        else:
            return formatted_content

    async def generate_storyboard(self, prompt: str, reference_style: str = "", aspect_ratio: str = "16:9") -> dict:
        if not self.client:
            raise ValueError("API Key not set. Please configure the Google API Key first.")
            
        """
        Generates a structured storyboard using Gemini 3 Pro with high thinking level.
        """
        system_instruction = """
        You are an expert manga creator. Your task is to generate a detailed, structured storyboard for a manga based on the user's request.
        
        The output must be a valid JSON object with the following structure:
        {
            "title": "Manga Title",
            "total_pages": n,
            "pages": [
                {
                    "page_number": 1,
                    "layout_description": "Description of the page layout (e.g., 4-panel grid, dynamic layout)",
                    "panels": [
                        {
                            "panel_number": 1,
                            "description": "Visual description of the panel content. Be extremely detailed about character appearance, clothing, pose, expression, background, lighting, and camera angle. This description will be used to generate an image.",
                            "dialogue": "Character dialogue or sound effects (optional)",
                            "shot_type": "Close-up, Wide-shot, Full-body, etc."
                        },
                        ...
                    ]
                },
                ...
            ]
        }
        
        Guidelines:
        - Ensure the story flow is logical, engaging, and has a clear beginning, middle, and end.
        - The visual descriptions for each panel must be self-contained and descriptive enough for an image generator.
        - Use "thinking" to plan out the story arc and page layouts before generating the JSON.
        """
        
        full_prompt = f"""
        Request: {prompt}
        Reference Style: {reference_style}
        Aspect Ratio: {aspect_ratio}
        
        Generate the storyboard in JSON format as specified.
        """

        def _call_gemini():
            try:
                response = self.client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="high"),
                        response_mime_type="application/json",
                        system_instruction=system_instruction
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                print(f"Error generating storyboard: {e}")
                raise

        return await asyncio.to_thread(_call_gemini)

    async def generate_storyboard_stream(self, prompt: str, reference_style: str = "", aspect_ratio: str = "16:9"):
        """Streaming version that yields thinking chunks then final JSON."""
        if not self.client:
            raise ValueError("API Key not set.")
        
        system_instruction = """
        You are an expert manga creator. Generate a structured storyboard for a manga.
        Output a valid JSON with: title, total_pages, pages (each with page_number, layout_description, panels array).
        Each panel has: panel_number, description (detailed visual), dialogue, shot_type.
        """
        
        full_prompt = f"""
        Request: {prompt}
        Reference Style: {reference_style}
        Aspect Ratio: {aspect_ratio}
        
        Generate the storyboard in JSON format.
        """
        
        import asyncio
        
        result_queue = asyncio.Queue()
        # Capture the loop reference BEFORE starting the thread
        loop = asyncio.get_running_loop()
        
        def _stream_gemini_sync():
            """Synchronous streaming that puts results into queue"""
            full_text = ""
            try:
                for chunk in self.client.models.generate_content_stream(
                    model="gemini-3-pro-preview",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="high",
                            include_thoughts=True
                        ),
                        response_mime_type="application/json",
                        system_instruction=system_instruction
                    )
                ):
                    # Process chunk
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'thought') and part.thought:
                                loop.call_soon_threadsafe(
                                    result_queue.put_nowait,
                                    {"type": "thinking", "content": part.text}
                                )
                            elif hasattr(part, 'text') and part.text:
                                full_text += part.text
                                loop.call_soon_threadsafe(
                                    result_queue.put_nowait,
                                    {"type": "text", "content": part.text}
                                )
                
                # Parse final result
                try:
                    result = json.loads(full_text)
                    loop.call_soon_threadsafe(
                        result_queue.put_nowait,
                        {"type": "result", "content": result}
                    )
                except json.JSONDecodeError as e:
                    loop.call_soon_threadsafe(
                        result_queue.put_nowait,
                        {"type": "error", "content": f"JSON parse error: {e}"}
                    )
            except Exception as e:
                loop.call_soon_threadsafe(
                    result_queue.put_nowait,
                    {"type": "error", "content": str(e)}
                )
            finally:
                loop.call_soon_threadsafe(
                    result_queue.put_nowait,
                    {"type": "done"}
                )
        
        # Start the streaming in a thread
        loop.run_in_executor(self.executor, _stream_gemini_sync)
        
        # Yield items from queue as they arrive
        while True:
            item = await result_queue.get()
            if item.get("type") == "done":
                break
            yield item


    async def generate_panel_image(self, panel_prompt: str, style_reference: str = "", aspect_ratio: str = "16:9", image_size: str = "1K", initial_image_base64: str = None, style_reference_image_base64: str = None) -> str:
        if not self.client:
            raise ValueError("API Key not set. Please configure the Google API Key first.")

        """
        Generates an image for a specific panel using Gemini 3 Pro Image.
        Returns the base64 encoded image string.
        """
        contents = [f"{panel_prompt}\nStyle Reference: {style_reference}"]
        
        # Add Style Reference Image if provided
        if style_reference_image_base64:
             try:
                contents.append(
                    types.Part(
                         inline_data=types.Blob(
                            mime_type="image/png", 
                            data=base64.b64decode(style_reference_image_base64)
                         )
                     )
                )
             except Exception as e:
                 print(f"Error decoding style reference image: {e}")
                 # Log but continue? Or raise? proceeding might be safer to avoid hard crash
        
        # Add Initial Image (for Redraw/Img2Img) if provided
        if initial_image_base64:
            try:
                # Append the image as a Part
                contents.append(
                    types.Part(
                         inline_data=types.Blob(
                            mime_type="image/png", 
                            data=base64.b64decode(initial_image_base64)
                         )
                     )
                )
            except Exception as e:
                print(f"Error decoding initial image: {e}")
                raise ValueError("Invalid initial_image_base64")

        def _call_gemini_image():
            try:
                print(f"Generating with size: {image_size}")
                response = self.client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE'],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size=image_size
                        )
                    )
                )
                
                for part in response.parts:
                    if part.inline_data:
                        return base64.b64encode(part.inline_data.data).decode('utf-8')
                return None
            except Exception as e:
                print(f"Error generating image: {e}")
                # Return None instead of raising to allow other panels to succeed
                return None

        return await asyncio.get_running_loop().run_in_executor(self.executor, _call_gemini_image)

    async def generate_manga_page(self, page_data: dict, style_reference: str = "", aspect_ratio: str = "9:16", image_size: str = "2K", style_reference_image_base64: str = None, reference_prompt: str = None, additional_image_base64: str = None) -> str:
        if not self.client:
             raise ValueError("API Key not set.")

        # Construct Prompt for the entire page
        prompt_parts = []
        
        # Add reference prompt for style consistency (from first page)
        if reference_prompt:
            prompt_parts.append(f"[Style Reference from Page 1]: {reference_prompt}\n")
        
        # Check for custom prompt
        if page_data.get('_custom_prompt'):
            prompt_parts.append(f"Generate a full manga page based on this description: {page_data.get('_custom_prompt')}")
            prompt_parts.append(f"Art Style: {style_reference}")
        else:
            prompt_parts.extend([
                f"Generate a full manga page. Layout: {page_data.get('layout_description', 'Standard manga layout')}.",
                f"Art Style: {style_reference}",
                "\nPanels:"
            ])
            
            for panel in page_data.get('panels', []):
                 p_num = panel.get('panel_number')
                 desc = panel.get('description', '')
                 raw_dialogue = panel.get('dialogue', '')
                 formatted_dialogue = self._format_dialogue(raw_dialogue)
                 shot = panel.get('shot_type', '')
                 prompt_parts.append(f"- Panel {p_num}: {desc}. Shot: {shot}. Dialogue: {formatted_dialogue}")
        
        if additional_image_base64:
             prompt_parts.append("\nUser input [Image] as reference")

        prompt_parts.append("\nEnsure the output is a single cohesive manga page with panels separated by gutters.")
        prompt_parts.append("IMPORTANT: Dialogue is provided in format (Character) \"Text\". Make sure the speech bubble points to the correct Character, but ONLY write the Text inside the quotes into the bubble.")
        
        full_prompt = "\n".join(prompt_parts)
        print(f"DEBUG: Full Prompt for Page: {full_prompt[:200]}...") # Log first 200 chars
        
        contents = [full_prompt]

        if style_reference_image_base64:
             try:
                contents.append(
                    types.Part(
                         inline_data=types.Blob(
                            mime_type="image/png", 
                            data=base64.b64decode(style_reference_image_base64)
                         )
                     )
                )
             except Exception as e:
                 print(f"Error decoding style reference image for page: {e}")

        if additional_image_base64:
             try:
                contents.append(
                    types.Part(
                         inline_data=types.Blob(
                            mime_type="image/png", 
                            data=base64.b64decode(additional_image_base64)
                         )
                     )
                )
             except Exception as e:
                 print(f"Error decoding additional reference image for page: {e}")

        # --- DEBUG LOGGING ---
        print("\nDEBUG: === RAW INPUT TO GEMINI (BATCH) ===", flush=True)
        for i, item in enumerate(contents):
            if isinstance(item, str):
                print(f"Item {i} [TEXT]:\n{item}\n", flush=True)
            elif hasattr(item, 'inline_data') and item.inline_data:
                print(f"Item {i} [IMAGE]: Mime={item.inline_data.mime_type}, SizeBytes={len(item.inline_data.data)}", flush=True)
            else:
                print(f"Item {i} [UNKNOWN]: {type(item)}", flush=True)
        print("DEBUG: === END RAW INPUT (BATCH) ===\n", flush=True)
        # ---------------------

        def _call_gemini_page():
            try:
                print(f"Generating PAGE with size: {image_size}")
                response = self.client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE'],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio, # Page is usually vertical
                            image_size=image_size
                        )
                    )
                )
                
                for part in response.parts:
                    if part.inline_data:
                        return base64.b64encode(part.inline_data.data).decode('utf-8')
                return None
            except Exception as e:
                print(f"Error generating page: {e}")
                return None

        return await asyncio.get_running_loop().run_in_executor(self.executor, _call_gemini_page)

    async def generate_manga_page_stream(self, page_data: dict, style_reference: str = "", aspect_ratio: str = "9:16", image_size: str = "2K", style_reference_image_base64: str = None, reference_prompt: str = None, additional_image_base64: str = None):
        """Streaming version of page generation that yields thinking then image."""
        if not self.client:
            raise ValueError("API Key not set.")
        
        import asyncio
        
        print(f"DEBUG: Streaming Page {page_data.get('page_number')}. StyleRef: {bool(style_reference_image_base64)}, AdditionalRef: {bool(additional_image_base64)}, CustomPrompt: {'YES' if page_data.get('_custom_prompt') else 'NO'} ({page_data.get('_custom_prompt', '')[:50]}...)", flush=True)
        
        # Build prompt (same as non-streaming)
        
        # Build prompt
        # Build prompt
        prompt_parts = []
        # Add reference prompt for style consistency (from first page)
        # BUT only if we are NOT using a custom override prompt
        if reference_prompt and not page_data.get('_custom_prompt'):
            prompt_parts.append(f"[Style Reference from Page 1]: {reference_prompt}\n")
        
        # Check for custom full-page prompt (from user edit)
        if page_data.get('_custom_prompt'):
            prompt_parts.append(f"Generate a full manga page based on this description: {page_data.get('_custom_prompt')}")
            prompt_parts.append(f"Art Style: {style_reference}")
        else:
            # Default construction from panels
            prompt_parts.extend([
                f"Generate a full manga page. Layout: {page_data.get('layout_description', 'Standard manga layout')}.",
                f"Art Style: {style_reference}",
                "\nPanels:"
            ])
            
            for panel in page_data.get('panels', []):
                p_num = panel.get('panel_number')
                desc = panel.get('description', '')
                raw_dialogue = panel.get('dialogue', '')
                formatted_dialogue = self._format_dialogue(raw_dialogue)
                shot = panel.get('shot_type', '')
                prompt_parts.append(f"- Panel {p_num}: {desc}. Shot: {shot}. Dialogue: {formatted_dialogue}")
        
        if additional_image_base64:
             prompt_parts.append("\nUser input [Image] as reference")

        prompt_parts.append("\nEnsure the output is a single cohesive manga page with panels separated by gutters.")
        prompt_parts.append("IMPORTANT: Dialogue is provided in format (Character) \"Text\". Make sure the speech bubble points to the correct Character, but ONLY write the Text inside the quotes into the bubble.")
        full_prompt = "\n".join(prompt_parts)
        
        contents = [full_prompt]
        
        if style_reference_image_base64:
            try:
                contents.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=base64.b64decode(style_reference_image_base64)
                        )
                    )
                )
            except Exception as e:
                print(f"Error decoding style reference image for page: {e}")
        
        if additional_image_base64:
            try:
                contents.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=base64.b64decode(additional_image_base64)
                        )
                    )
                )
            except Exception as e:
                print(f"Error decoding additional reference image for page: {e}")
        
        # --- DEBUG LOGGING ---
        print("\nDEBUG: === RAW INPUT TO GEMINI ===", flush=True)
        for i, item in enumerate(contents):
            if isinstance(item, str):
                print(f"Item {i} [TEXT]:\n{item}\n", flush=True)
            elif hasattr(item, 'inline_data') and item.inline_data:
                print(f"Item {i} [IMAGE]: Mime={item.inline_data.mime_type}, SizeBytes={len(item.inline_data.data)}", flush=True)
            else:
                print(f"Item {i} [UNKNOWN]: {type(item)}", flush=True)
        print("DEBUG: === END RAW INPUT ===\n", flush=True)
        # ---------------------

        result_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        def _stream_gemini_page():
            image_data = None
            try:
                for chunk in self.client.models.generate_content_stream(
                    model="gemini-3-pro-image-preview",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE'],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size=image_size
                        )
                    )
                ):
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'thought') and part.thought and hasattr(part, 'text'):
                                loop.call_soon_threadsafe(
                                    result_queue.put_nowait,
                                    {"type": "thinking", "content": part.text}
                                )
                            elif hasattr(part, 'inline_data') and part.inline_data:
                                image_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                
                if image_data:
                    loop.call_soon_threadsafe(
                        result_queue.put_nowait,
                        {"type": "image", "content": image_data, "page_number": page_data.get('page_number')}
                    )
                else:
                    loop.call_soon_threadsafe(
                        result_queue.put_nowait,
                        {"type": "error", "content": "No image generated"}
                    )
            except Exception as e:
                loop.call_soon_threadsafe(
                    result_queue.put_nowait,
                    {"type": "error", "content": str(e)}
                )
            finally:
                loop.call_soon_threadsafe(
                    result_queue.put_nowait,
                    {"type": "done"}
                )
        
        loop.run_in_executor(self.executor, _stream_gemini_page)
        
        while True:
            item = await result_queue.get()
            if item.get("type") == "done":
                break
            yield item
