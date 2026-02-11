import json
import os
import argparse
from PIL import Image, ImageDraw

def apply_masks(image_path, json_path, output_path):
    """
    Applies colored masks to a comic page based on panel coordinates.
    """
    # 1. Load the image
    try:
        original_image = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    width, height = original_image.size
    print(f"Loaded image: {image_path} ({width}x{height})")

    # 2. Load the JSON data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {json_path}")
        return

    panels = data.get('panels', [])
    if not panels:
        print("No panels found in JSON.")
        return

    # 3. Create a transparent overlay for drawing masks
    overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Define a list of colors for the masks (RGBA)
    # Fill colors: Very transparent (alpha=30)
    # Border colors: Opaque (alpha=255) or semi-opaque
    
    base_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 165, 0),  # Orange
        (128, 0, 128)   # Purple
    ]

    # 4. Process each panel
    for i, panel in enumerate(panels):
        try:
            # Extract normalized coordinates
            x1_norm = panel['x1']
            y1_norm = panel['y1']
            x2_norm = panel['x2']
            y2_norm = panel['y2']

            # Convert to pixel coordinates
            left = int(x1_norm * width)
            upper = int(y1_norm * height)
            right = int(x2_norm * width)
            lower = int(y2_norm * height)

            # Validating coordinates
            left = max(0, left)
            upper = max(0, upper)
            right = min(width, right)
            lower = min(height, lower)

            if right <= left or lower <= upper:
                print(f"Warning: Invalid panel coordinates for panel {panel.get('index', i+1)}: ({left}, {upper}, {right}, {lower}). Skipping.")
                continue

            # Select a color based on index
            base_color = base_colors[i % len(base_colors)]
            
            # Fill color with low alpha (e.g., 30)
            fill_color = base_color + (30,)
            
            # Border color with high alpha (e.g., 200 or 255)
            outline_color = base_color + (255,)

            # Draw the mask on the overlay using ImageDraw
            # width=3 for a visible "highlight" border
            draw.rectangle([left, upper, right, lower], fill=fill_color, outline=outline_color, width=4)

            print(f"Applied mask for panel {panel.get('index', i+1)}")

        except KeyError as e:
            print(f"Warning: Missing coordinate key {e} in panel data. Skipping panel.")
        except Exception as e:
            print(f"Error processing panel {panel.get('index', i+1)}: {e}")

    # 5. Composite the overlay onto the original image
    combined = Image.alpha_composite(original_image, overlay)

    # 6. Save the result
    try:
        combined.save(output_path)
        print(f"Saved masked image to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply colored masks to comic panels based on JSON coordinates.")
    parser.add_argument("--image_path", required=True, help="Path to the input image file.")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file containing panel coordinates.")
    parser.add_argument("--output_path", default="masked_output.png", help="Path to save the output image.")

    args = parser.parse_args()

    apply_masks(args.image_path, args.json_path, args.output_path)
