"""
Dataset Generation Script
Creates a 500-image dataset with diverse content for CBIR testing
"""

import cv2
import numpy as np
from pathlib import Path
import random
import os

def create_dataset(output_dir='static/dataset', num_images=500):
    """
    Create a dataset with 500 diverse images for CBIR testing
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} test images in {output_dir}...")
    
    # Create organized subdirectories
    categories = ['geometric', 'natural', 'patterns', 'gradients', 'mixed']
    for cat in categories:
        Path(output_dir) .joinpath(cat).mkdir(exist_ok=True)
    
    images_per_category = num_images // len(categories)
    
    # Generate geometric shapes
    print("Generating geometric images...")
    for i in range(images_per_category):
        img = generate_geometric_image()
        filename = f"{output_dir}/geometric/img_{i:04d}.jpg"
        cv2.imwrite(filename, img)
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{images_per_category} geometric images")
    
    # Generate natural-like images
    print("Generating natural images...")
    for i in range(images_per_category):
        img = generate_natural_image()
        filename = f"{output_dir}/natural/img_{i:04d}.jpg"
        cv2.imwrite(filename, img)
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{images_per_category} natural images")
    
    # Generate pattern images
    print("Generating pattern images...")
    for i in range(images_per_category):
        img = generate_pattern_image()
        filename = f"{output_dir}/patterns/img_{i:04d}.jpg"
        cv2.imwrite(filename, img)
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{images_per_category} pattern images")
    
    # Generate gradient images
    print("Generating gradient images...")
    for i in range(images_per_category):
        img = generate_gradient_image()
        filename = f"{output_dir}/gradients/img_{i:04d}.jpg"
        cv2.imwrite(filename, img)
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{images_per_category} gradient images")
    
    # Generate mixed images
    print("Generating mixed images...")
    remaining = num_images - (images_per_category * len(categories))
    for i in range(images_per_category + remaining):
        choice = random.choice([
            generate_geometric_image,
            generate_natural_image,
            generate_pattern_image,
            generate_gradient_image,
            generate_mixed_image
        ])
        img = choice()
        filename = f"{output_dir}/mixed/img_{i:04d}.jpg"
        cv2.imwrite(filename, img)
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{images_per_category} mixed images")
    
    total_created = len(list(Path(output_dir).rglob('*.jpg')))
    print(f"\n✓ Successfully created {total_created} images in {output_dir}")
    return total_created

def generate_geometric_image(width=256, height=256):
    """Generate image with geometric shapes"""
    # Create background with random color
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bg_color = tuple(np.random.randint(50, 200, 3).tolist())
    img[:] = bg_color
    
    # Add random geometric shapes
    num_shapes = random.randint(3, 8)
    
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'line', 'triangle'])
        color = tuple(np.random.randint(0, 256, 3).tolist())
        
        if shape_type == 'circle':
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(10, 50)
            cv2.circle(img, center, radius, color, -1)
        
        elif shape_type == 'rectangle':
            pt1 = (random.randint(0, width-50), random.randint(0, height-50))
            pt2 = (pt1[0] + random.randint(20, 100), pt1[1] + random.randint(20, 100))
            cv2.rectangle(img, pt1, pt2, color, -1)
        
        elif shape_type == 'line':
            pt1 = (random.randint(0, width), random.randint(0, height))
            pt2 = (random.randint(0, width), random.randint(0, height))
            cv2.line(img, pt1, pt2, color, random.randint(1, 5))
        
        elif shape_type == 'triangle':
            pts = np.array([
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)],
                [random.randint(0, width), random.randint(0, height)]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
    
    return img

def generate_natural_image(width=256, height=256):
    """Generate image with natural-like patterns (clouds, noise)"""
    # Create base with gradient
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add color gradient
    for i in range(height):
        color_val = int(100 + 100 * i / height)
        img[i, :] = [color_val, color_val - 30, color_val - 60]
    
    # Add Perlin-like noise (using random gaussian)
    noise = np.random.normal(0, 30, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some blob-like shapes
    for _ in range(random.randint(3, 7)):
        center = (random.randint(0, width), random.randint(0, height))
        radius = random.randint(20, 60)
        color = tuple(np.random.randint(50, 200, 3).tolist())
        cv2.circle(img, center, radius, color, -1)
        # Blur to make it look more natural
        cv2.GaussianBlur(img, (7, 7), 0)
    
    return img

def generate_pattern_image(width=256, height=256):
    """Generate image with regular patterns"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bg_color = tuple(np.random.randint(30, 100, 3).tolist())
    img[:] = bg_color
    
    # Create pattern
    pattern_type = random.choice(['grid', 'dots', 'lines', 'checkerboard'])
    fg_color = tuple(np.random.randint(150, 256, 3).tolist())
    
    if pattern_type == 'grid':
        spacing = random.randint(20, 50)
        for i in range(0, width, spacing):
            cv2.line(img, (i, 0), (i, height), fg_color, 1)
        for i in range(0, height, spacing):
            cv2.line(img, (0, i), (width, i), fg_color, 1)
    
    elif pattern_type == 'dots':
        spacing = random.randint(20, 50)
        for i in range(0, width, spacing):
            for j in range(0, height, spacing):
                cv2.circle(img, (i, j), random.randint(3, 8), fg_color, -1)
    
    elif pattern_type == 'lines':
        spacing = random.randint(15, 40)
        for i in range(0, width, spacing):
            cv2.line(img, (i, 0), (i + height, height), fg_color, 2)
    
    elif pattern_type == 'checkerboard':
        square_size = random.randint(16, 32)
        for i in range(0, width, square_size):
            for j in range(0, height, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    cv2.rectangle(img, (i, j), (i + square_size, j + square_size), fg_color, -1)
    
    return img

def generate_gradient_image(width=256, height=256):
    """Generate image with color gradients"""
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    gradient_type = random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])
    
    color1 = np.array(np.random.randint(0, 256, 3), dtype=np.float32)
    color2 = np.array(np.random.randint(0, 256, 3), dtype=np.float32)
    
    if gradient_type == 'horizontal':
        for i in range(width):
            color = color1 + (color2 - color1) * (i / width)
            img[:, i] = np.clip(color, 0, 255)
    
    elif gradient_type == 'vertical':
        for j in range(height):
            color = color1 + (color2 - color1) * (j / height)
            img[j, :] = np.clip(color, 0, 255)
    
    elif gradient_type == 'diagonal':
        for i in range(width):
            for j in range(height):
                t = (i + j) / (width + height)
                color = color1 + (color2 - color1) * t
                img[j, i] = np.clip(color, 0, 255)
    
    elif gradient_type == 'radial':
        cx, cy = width // 2, height // 2
        max_dist = np.sqrt(cx**2 + cy**2)
        for i in range(width):
            for j in range(height):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                t = dist / max_dist
                color = color1 + (color2 - color1) * t
                img[j, i] = np.clip(color, 0, 255)
    
    return img.astype(np.uint8)

def generate_mixed_image(width=256, height=256):
    """Generate image combining multiple techniques"""
    choice = random.choice([
        lambda: blend_images(generate_gradient_image(), generate_geometric_image()),
        lambda: blend_images(generate_natural_image(), generate_pattern_image()),
        lambda: add_text_to_image(generate_geometric_image())
    ])
    return choice()

def blend_images(img1, img2, alpha=0.5):
    """Blend two images"""
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

def add_text_to_image(img):
    """Add text to image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"IMG_{random.randint(1000, 9999)}"
    color = tuple(np.random.randint(0, 256, 3).tolist())
    cv2.putText(img, text, (20, 240), font, 0.8, color, 2)
    return img

if __name__ == '__main__':
    import sys
    
    # Check command line arguments
    num_images = 500
    output_dir = 'static/dataset'
    
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except:
            print(f"Invalid number: {sys.argv[1]}")
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    create_dataset(output_dir, num_images)
