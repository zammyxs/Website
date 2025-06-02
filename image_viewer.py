import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
from screeninfo import get_monitors  # Import screeninfo to get screen size

# Image Variables
original_img = None
edited_img = None
current_img = None  # Track the current state for undo functionality

def open_image():
    """Open an image and display it."""
    global original_img, edited_img, current_img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if file_path:
        original_img = Image.open(file_path).convert("RGB")
        edited_img = original_img.copy()
        current_img = original_img.copy()  # Initialize current image
        update_display()
        enable_controls()  # Enable controls after loading the image

def update_display():
    """Update the displayed original and edited images."""
    if original_img is None or edited_img is None:
        return

    # Calculate display size (maintain aspect ratio but limit size)
    max_display_width = 300  # Maximum width for display
    max_display_height = 300  # Maximum height for display
    
    # Original image resizing
    original_width, original_height = original_img.size
    display_ratio = min(max_display_width / original_width, max_display_height / original_height)
    display_width = int(original_width * display_ratio)
    display_height = int(original_height * display_ratio)
    
    # Resize for display
    display_original = original_img.resize((display_width, display_height), Image.LANCZOS)
    display_edited = edited_img.resize((display_width, display_height), Image.LANCZOS)
    
    # Convert to PhotoImage
    img_tk_original = ImageTk.PhotoImage(display_original)
    img_tk_edited = ImageTk.PhotoImage(display_edited)
    
    # Update labels with new images
    original_label.config(image=img_tk_original)
    original_label.image = img_tk_original
    original_label.config(width=display_width, height=display_height)
    
    edited_label.config(image=img_tk_edited)
    edited_label.image = img_tk_edited
    edited_label.config(width=display_width, height=display_height)
    
    # Update labels
    original_text.config(text="Original")
    edited_text.config(text="Edited")

def convert_background(image, to_white=True):
    """Convert the background from black to white or vice versa."""
    img_cv = np.array(image)
    if to_white:
        # Convert black background to white
        img_cv[np.all(img_cv == [0, 0, 0], axis=-1)] = [255, 255, 255]
    else:
        # Convert white background to black
        img_cv[np.all(img_cv == [255, 255, 255], axis=-1)] = [0, 0, 0]
    return Image.fromarray(img_cv)

def detect_shapes():
    """Detect and highlight shapes in the image."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        img = np.array(original_img)
        # Convert to grayscale if it's not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Apply adaptive thresholding for better edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Invert the image if the background is dark
        mean_intensity = np.mean(gray)
        if mean_intensity < 128:
            gray = cv2.bitwise_not(gray)
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find only external contours to avoid inner duplicates
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on a copy of the image for visualization
        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        
        # Create a temporary copy of the current image with contours for visualization
        contour_img = Image.fromarray(img_contours)
        temp_img = edited_img.copy()
        edited_img = contour_img
        update_display()
        
        # Restore the original current image after display
        edited_img = temp_img
        
        # Initialize shape counters
        shape_counts = {"Circle": 0, "Square": 0, "Triangle": 0, "Rectangle": 0, 
                        "Pentagon": 0, "Hexagon": 0, "Heptagon": 0, "Star": 0,
                        "Semi-Circle": 0, "Quadrilateral": 0, "Diamond": 0, "Ellipse": 0}
        
        # Store individual shape details for reporting
        shape_details = []
        
        # Minimum area threshold - adjust based on your typical images
        min_area_threshold = 200
        
        for cnt in contours:
            # Filter out very small contours (noise)
            area = cv2.contourArea(cnt)
            if area < min_area_threshold:  
                continue
                
            # Calculate perimeter and use it to approximate the shape
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            num_vertices = len(approx)
            
            # Get bounding rectangle for aspect ratio calculation
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Detect shape based on number of vertices and other properties
            shape_name = "Unknown"
            
            if num_vertices == 8:
                # Check if the shape is approximately circular
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                if circularity > 0.85:
                    shape_name = "Circle"
                    shape_counts["Circle"] += 1
                else:
                    shape_name = "Ellipse"
                    shape_counts["Ellipse"] += 1
            
            # For polygons with 3-7 vertices
            if 3 <= num_vertices <= 7:
                if num_vertices == 3:
                    shape_name = "Triangle"
                    shape_counts["Triangle"] += 1
                    
                elif num_vertices == 4:
                    # Distinguish between squares, rectangles, and irregular quadrilaterals
                    area_ratio = area / (w * h)  # Area of contour / area of bounding rect
                    
                    if 0.9 <= aspect_ratio <= 1.1 :
                        shape_name = "Square"
                        shape_counts["Square"] += 1
                    else:
                        # Check angles to determine if it's a quadrilateral
                        angles = []
                        for i in range(4):
                            p1 = approx[i][0]
                            p2 = approx[(i + 1) % 4][0]
                            p3 = approx[(i + 2) % 4][0]
                            v1 = p1 - p2
                            v2 = p3 - p2
                            angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
                            angles.append(angle)
                        
                        if all(90 <= angle <= 90 for angle in angles):
                            shape_name = "Rectangle"
                            shape_counts["Rectangle"] += 1
                        elif all(35 <= angle <= 145 for angle in angles):
                            shape_name = "Diamond"
                            shape_counts["Diamond"] += 1
                        else:
                            shape_name = "Quadrilateral"
                            shape_counts["Quadrilateral"] += 1
                        
                elif num_vertices == 5:
                    shape_name = "Pentagon"
                    shape_counts["Pentagon"] += 1
                    
                elif num_vertices == 6:
                    shape_name = "Hexagon"
                    shape_counts["Hexagon"] += 1
                    
                elif num_vertices == 7:
                    shape_name = "Heptagon"
                    shape_counts["Heptagon"] += 1

        
            # For semi-circles
            elif num_vertices == 5 and aspect_ratio > 0.5:
                shape_name = "Ellipse"
                shape_counts["Ellipse"] += 1

            # For stars
            elif num_vertices >= 10:
                shape_name = "Star"
                shape_counts["Star"] += 1

            # Draw the shape name on the image
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img_contours, shape_name, (cx - 30, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Show the visualization image with shape names
        vis_img_pil = Image.fromarray(img_contours)
        temp_img = edited_img.copy()
        edited_img = vis_img_pil
        update_display()
        
        # Restore the original current image after display
        edited_img = temp_img
        
        # Format and display results
        total_shapes = sum(shape_counts.values())
        summary = f"Total shapes detected: {total_shapes}\n"
        
        if total_shapes > 0:
            # List counts by shape type
            counts_text = ", ".join([f"{v} {k}(s)" for k, v in shape_counts.items() if v > 0])
            results_text = f"{summary}{counts_text}"
            messagebox.showinfo("Shape Detection Results", results_text)
            print(results_text)
        else:
            messagebox.showinfo("Shape Detection Results", "No shapes detected. Try adjusting the image or contrast.")
            print("No shapes detected. Try adjusting the image or contrast.")

def detect_colors():
    """Detect and label colors in the image with improved background filtering."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        # Convert background from black to white
        img_cv = np.array(convert_background(original_img, to_white=True))
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Threshold to create a mask for non-white areas
        _, mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Use morphological operations to clean the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out very small or very large contours (likely noise or background)
        min_area = 100  # Minimum area to consider a shape
        max_area = img_cv.shape[0] * img_cv.shape[1] * 0.9  # Avoid large background regions
        contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        detected_colors = {}
        img_with_labels = img_cv.copy()

        for contour in contours:
            # Compute the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cX, cY = x + w // 2, y + h // 2
            
            # Sample a small area around the center to avoid edge artifacts
            sample_size = 5
            x1, y1 = max(0, cX - sample_size), max(0, cY - sample_size)
            x2, y2 = min(img_cv.shape[1], cX + sample_size), min(img_cv.shape[0], cY + sample_size)
            sample_region = img_cv[y1:y2, x1:x2]

            if sample_region.size > 0:
                # Compute average color of the sampled region
                avg_color = np.mean(sample_region, axis=(0, 1))
                r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

                # Skip white and black background
                if (r > 240 and g > 240 and b > 240) or (r < 15 and g < 15 and b < 15):
                    continue

                color_name = get_color_name(r, g, b)
                detected_colors[color_name] = detected_colors.get(color_name, 0) + 1

                # Draw contour and label color name
                cv2.drawContours(img_with_labels, [contour], -1, (0, 255, 0), 2)
                cv2.putText(img_with_labels, f"{color_name}: ({r},{g},{b})", (cX - 10, cY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Update the edited image
        edited_img = Image.fromarray(img_with_labels)
        update_display()

        # Display detected colors
        if detected_colors:
            results_text = f"Detected colors ({len(detected_colors)}): " + ", ".join(
                [f"{color} ({count})" for color, count in detected_colors.items()]
            )
            messagebox.showinfo("Color Detection Results", results_text)
        else:
            messagebox.showinfo("Color Detection Results", "No colors detected. Try adjusting the image or contrast.")

def get_color_name(r, g, b):
    """Return the name of the color based on RGB values with broader ranges."""
    # Print the RGB values for debugging
    print(f"Analyzing RGB: ({r}, g, {b})")
    
    # Red family
    if r > 180 and g < 120 and b < 120:
        return "Red"
    # Green family
    elif r < 120 and g > 120 and b < 120:
        return "Green"
    # Blue family
    elif r < 120 and g < 120 and b > 120:
        return "Blue"
    # Yellow family
    elif r > 180 and g > 180 and b < 100:
        return "Yellow"
    # Orange family
    elif r > 200 and g > 100 and g < 200 and b < 100:
        return "Orange"
    # Brown family
    elif r > 120 and r < 200 and g > 80 and g < 160 and b > 60 and b < 120:
     return "Brown"
    # White family
    elif r > 180 and g > 180 and b > 180:
        return "White"
    # # Black family
    # elif r < 60 and g < 60 and b < 60:
    #     return "Black"
    # Gray family
    elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30 and r > 60 and r < 180:
        return "Gray"
    # Purple family
    elif r > 80 and g < 120 and b > 80 and r < b + 60:
        return "Purple"
    # Cyan family
    elif r < 120 and g > 120 and b > 120:
        return "Cyan"
    # Pink family
    elif r > 180 and g < 180 and b > 120:
        return "Pink"
    # Light blue (for pentagon in image)
    elif r < 120 and g > 120 and b > 180:
        return "Light Blue"
    # Light green (for circle in image)
    elif r > 120 and g > 180 and b < 120:
        return "Light Green"
    # Light purple (for triangle in image)
    elif r > 120 and g < 120 and b > 180:
        return "Light Purple"
    # else:
    #     return "Unknown"

def show_histogram():
    """Display the RGB histogram of the currently loaded image."""
    if original_img:
        # Convert the image to a NumPy array
        img_array = np.array(original_img)

        # Split the channels
        if len(img_array.shape) == 3:  # RGB image
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        else:  # Grayscale image
            r = g = b = img_array

        # Plot the histograms
        plt.figure(figsize=(10, 6))
        plt.title("RGB Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.hist(r.ravel(), bins=256, color="red", alpha=0.5, label="Red")
        plt.hist(g.ravel(), bins=256, color="green", alpha=0.5, label="Green")
        plt.hist(b.ravel(), bins=256, color="blue", alpha=0.5, label="Blue")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()
    else:
        messagebox.showwarning("No Image", "Please load an image to view its histogram.")

def binary_image_projection():
    """Calculate and display the binary image with its horizontal and vertical projections."""
    if original_img:
        # Convert the image to grayscale and apply binary threshold
        img_cv = np.array(original_img.convert("L"))
        _, binary_img = cv2.threshold(img_cv, 128, 255, cv2.THRESH_BINARY)

        # Calculate horizontal and vertical projections
        horizontal_projection = np.sum(binary_img == 255, axis=1)  # Sum along rows
        vertical_projection = np.sum(binary_img == 255, axis=0)  # Sum along columns

        # Plot the binary image and projections
        plt.figure(figsize=(12, 8))

        # Binary image
        plt.subplot(2, 2, 1)
        plt.imshow(binary_img, cmap="gray")
        plt.title("Binary Image")
        plt.axis("off")

        # Horizontal projection
        plt.subplot(2, 2, 2)
        plt.plot(horizontal_projection, range(len(horizontal_projection)), color="blue")
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        plt.title("Horizontal Projection")
        plt.xlabel("Number of 1 Pixels")
        plt.ylabel("Row Index")

        # Vertical projection
        plt.subplot(2, 2, 4)
        plt.plot(range(len(vertical_projection)), vertical_projection, color="red")
        plt.title("Vertical Projection")
        plt.xlabel("Column Index")
        plt.ylabel("Number of 1 Pixels")

        plt.tight_layout()
        plt.show()
    else:
        messagebox.showwarning("No Image", "Please load an image to calculate projections.")

def apply_filter(filter_type):
    """Apply a selected filter to the image."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        img = original_img.copy()
        
        # Define available filters
        filters = {
            "Sepia": img.convert("RGB").point(lambda p: min(255, int(p * 1.2))),  # Example Sepia filter
            "Vivid": ImageEnhance.Color(img).enhance(2.0),  # Increase color saturation for vivid effect
            "Color Leak": img.filter(ImageFilter.CONTOUR),  # Example filter for "Color Leak"
            "Melbourne": img.filter(ImageFilter.DETAIL),  # Example filter for "Melbourne"
            "Amaro": ImageEnhance.Brightness(img).enhance(1.2),  # Example filter for "Amaro"
            "Nashville": ImageEnhance.Contrast(img).enhance(1.3),  # Example filter for "Nashville"
            "Fade Warm": ImageEnhance.Color(img).enhance(0.8),  # Example filter for "Fade Warm"
            "Invert": ImageOps.invert(img.convert("RGB")),
            "Darken": ImageEnhance.Brightness(img).enhance(0.7),
            "Grayscale": img.convert("L").convert("RGB"),  # Convert to grayscale and back to RGB
        }
        
        # Debugging: Print the selected filter type
        print(f"Applying filter: {filter_type}")
        
        # Apply the selected filter
        if filter_type in filters:
            edited_img = filters[filter_type]
            update_display()
        else:
            print(f"Filter '{filter_type}' not found.")

def adjust_rgb():
    """Adjust the RGB values of the image based on slider values."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        r = r_slider.get()
        g = g_slider.get()
        b = b_slider.get()
        
        # Create a copy of the original image
        img_cv = np.array(original_img)
        
        # Split the channels
        b_channel, g_channel, r_channel = cv2.split(img_cv)
        
        # Apply adjustments
        r_channel = np.clip(r_channel.astype(np.int16) + r, 0, 255).astype(np.uint8)
        g_channel = np.clip(g_channel.astype(np.int16) + g, 0, 255).astype(np.uint8)
        b_channel = np.clip(b_channel.astype(np.int16) + b, 0, 255).astype(np.uint8)
        
        # Merge channels back
        img_cv = cv2.merge((b_channel, g_channel, r_channel))
        
        edited_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        update_display()

def apply_threshold():
    """Apply a threshold to the image based on slider value and threshold type."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        # Get the threshold value and type
        threshold_value = threshold_slider.get()
        lower_threshold = lower_threshold_slider.get()
        upper_threshold = upper_threshold_slider.get()
        threshold_type = threshold_var.get()
        
        print(f"Threshold Value: {threshold_value}, Lower: {lower_threshold}, Upper: {upper_threshold}")
        
        # Convert the image to grayscale
        gray_image = original_img.convert("L")

        
        if threshold_type == "simple":
            # Simple binary threshold
            edited_img = gray_image.point(lambda x: 0 if x < threshold_value else 255).convert("RGB")
        elif threshold_type == "adaptive":
            # Adaptive threshold
            block_size = 11  # Size of a pixel neighborhood used to calculate threshold value
            C = 2  # Constant subtracted from the mean or weighted mean
            img_cv = np.array(gray_image)
            img_thresh = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, block_size, C)
            edited_img = Image.fromarray(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB))
        elif threshold_type == "average":
            # Average threshold
            edited_img = gray_image.point(lambda x: 0 if x < threshold_value else 255).convert("RGB")
        elif threshold_type == "dual":
            # Dual threshold
            edited_img = gray_image.point(
                lambda x: 0 if lower_threshold <= x <= upper_threshold else 255
            ).convert("RGB")
        else:
            # Default to simple threshold if type is not set
            edited_img = gray_image.point(lambda x: 0 if x < threshold_value else 255).convert("RGB")
        
        update_display()

def undo_changes():
    """Revert to the previous state of the image."""
    global edited_img, current_img  
    if current_img:
        edited_img = current_img.copy()
        update_display()

def reset_image():
    """Reset to the original image."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        edited_img = original_img.copy()
        update_display()

def rotate_image():
    """Rotate the image by the specified degree."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        # Get the rotation angle from the slider
        angle = rotation_slider.get()
        
        # Rotate the image
        rotated_img = original_img.rotate(angle, expand=True)
        edited_img = rotated_img
        update_display()

def mirror_image(direction):
    """Mirror the image either vertically or horizontally."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        if direction == "horizontal":
            mirrored_img = edited_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            mirrored_img = edited_img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return  # Invalid direction
        
        edited_img = mirrored_img
        update_display()

def translate_image():
    """Translate (shift) the image to specified (x, y) coordinates."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        # Get translation values from sliders
        x_shift = x_translation_slider.get()
        y_shift = y_translation_slider.get()
        
        # Convert the image to a NumPy array for translation
        img_cv = np.array(edited_img)
        rows, cols, _ = img_cv.shape
        
        # Create the translation matrix
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        
        # Apply the translation
        translated_img = cv2.warpAffine(img_cv, translation_matrix, (cols, rows))
        
        # Convert back to a PIL image
        edited_img = Image.fromarray(translated_img)
        update_display()

def apply_kernel():
    """Apply a kernel operation to the image and display the output."""
    global edited_img, current_img
    if original_img:
        try:
            # Save current state before modification
            current_img = edited_img.copy()

            # Get the kernel size and type
            kernel_size = int(kernel_size_var.get())
            if kernel_size % 2 == 0 or kernel_size < 1:
                raise ValueError("Kernel size must be an odd positive integer.")

            kernel_type = kernel_type_var.get()

            # Create the kernel
            if kernel_type == "average":
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            elif kernel_type == "gaussian":
                kernel = cv2.getGaussianKernel(kernel_size, -1)
                kernel = np.outer(kernel, kernel)
            elif kernel_type == "custom":
                custom_kernel = custom_kernel_var.get()
                custom_values = [float(x) for x in custom_kernel.split(",")]
                if len(custom_values) != kernel_size * kernel_size:
                    raise ValueError(f"Custom kernel must have {kernel_size * kernel_size} values.")
                kernel = np.array(custom_values, dtype=np.float32).reshape(kernel_size, kernel_size)
            else:
                raise ValueError("Invalid kernel type selected.")

            # Debugging: Print kernel details
            print(f"Kernel Type: {kernel_type}")
            print(f"Kernel Size: {kernel_size}")
            print(f"Kernel:\n{kernel}")

            # Apply the kernel to the image
            img_array = np.array(edited_img)
            if len(img_array.shape) == 2:  # Grayscale image
                filtered_array = cv2.filter2D(img_array, -1, kernel)
            else:  # RGB image
                filtered_array = cv2.filter2D(img_array, -1, kernel)

            # Convert the filtered array back to an image
            filtered_img = Image.fromarray(filtered_array)

            # Display the filtered image in a new window
            # Resize the filtered image for display
            display_width, display_height = 300, 300
            filtered_resized = filtered_img.resize((display_width, display_height), Image.LANCZOS)
            img_tk_filtered = ImageTk.PhotoImage(filtered_resized)

            # Display the filtered image in the main application
            edited_label.config(image=img_tk_filtered)
            edited_label.image = img_tk_filtered

            # Update the edited image
            edited_img = filtered_img
            update_display()
        except ValueError as e:
            tk.messagebox.showerror("Invalid Input", str(e))
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error details: {e}")
    else:
        tk.messagebox.showerror("No Image", "Please upload an image first.")

def apply_convolution(kernel):
    """Apply a convolution filter to the image."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        # Convert the image to a NumPy array
        img_array = np.array(edited_img)
        
        # Ensure the image is in the correct format (RGB or Grayscale)
        if len(img_array.shape) == 3:  # RGB image
            # Apply the kernel to each channel separately
            try:
                # Convert the image to a NumPy array
                img_array = np.array(edited_img)
                
                # Ensure the image is in the correct format (RGB or Grayscale)
                if len(img_array.shape) == 3:  # RGB image
                    # Apply the kernel to each channel separately
                    channels = cv2.split(img_array)
                    filtered_channels = [cv2.filter2D(channel, -1, np.array(kernel, dtype=np.float32)) for channel in channels]
                    filtered_array = cv2.merge(filtered_channels)
                else:  # Grayscale image
                    filtered_array = cv2.filter2D(img_array, -1, np.array(kernel, dtype=np.float32))
            except Exception as e:
                tk.messagebox.showerror("Error", f"Invalid kernel or image format: {str(e)}")
                return
            update_display()

# Enable Grayscale and Binary Buttons
def enable_controls():
    """Enable all controls after an image is loaded."""
    detect_shapes_button.config(state=tk.NORMAL)
    detect_colors_button.config(state=tk.NORMAL)
    grayscale_button.config(state=tk.NORMAL)  # Enable Grayscale button
    binary_button.config(state=tk.NORMAL)  # Enable Binary button
    apply_filter_button.config(state=tk.NORMAL)  # Enable the new filter apply button
    rgb_adjust_button.config(state=tk.NORMAL)
    threshold_button.config(state=tk.NORMAL)
    undo_button.config(state=tk.NORMAL)
    reset_button.config(state=tk.NORMAL)
    save_button.config(state=tk.NORMAL)
    r_slider.config(state=tk.NORMAL)
    g_slider.config(state=tk.NORMAL)
    b_slider.config(state=tk.NORMAL)
    threshold_slider.config(state=tk.NORMAL)
    lower_threshold_slider.config(state=tk.NORMAL)  # Enable lower threshold slider
    upper_threshold_slider.config(state=tk.NORMAL)  # Enable upper threshold slider
    threshold_type_simple.config(state=tk.NORMAL)
    threshold_type_adaptive.config(state=tk.NORMAL)
    threshold_type_average.config(state=tk.NORMAL)  # Enable average threshold radio button
    threshold_type_dual.config(state=tk.NORMAL)  # Enable dual threshold radio button
    enhancers_toggle_button.config(state=tk.NORMAL)
    edge_detection_button.config(state=tk.NORMAL)
    image_segmentation_button.config(state=tk.NORMAL)
    threshold_segmentation_button.config(state=tk.NORMAL)
    grabcut_segmentation_button.config(state=tk.NORMAL)
    apply_kernel_button.config(state=tk.NORMAL)  # Enable kernel button
    toggle_enhancers(enhancers_toggle.get())  # Ensure enhancers are enabled if the toggle is on

def save_image():
    """Save the edited image."""
    if edited_img:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            edited_img.save(file_path)
            messagebox.showinfo("Success", "Image saved successfully!")

def toggle_enhancers(state):
    """Enable or disable enhancers sliders based on toggle state."""
    new_state = tk.NORMAL if state else tk.DISABLED
    opacity_slider.config(state=new_state)
    brightness_slider.config(state=new_state)
    contrast_slider.config(state=new_state)
    saturation_slider.config(state=new_state)
    blur_slider.config(state=new_state)
    apply_enhancers_button.config(state=new_state)

def apply_enhancers():
    """Apply the selected enhancers to the image."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        img = original_img.copy()
        
        # Apply opacity
        opacity = opacity_slider.get() / 100
        img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), opacity)
        
        # Apply brightness
        brightness = brightness_slider.get() / 100
        img = ImageEnhance.Brightness(img).enhance(brightness)
        
        # Apply contrast
        contrast = contrast_slider.get() / 100
        img = ImageEnhance.Contrast(img).enhance(contrast)
        
        # Apply saturation
        saturation = saturation_slider.get() / 100
        img = ImageEnhance.Color(img).enhance(saturation)
        
        # Apply blur
        blur = blur_slider.get()
        img = img.filter(ImageFilter.GaussianBlur(blur))
        
        edited_img = img
        update_display()

def update_threshold_controls(*args):
    """Enable or disable sliders based on the selected threshold type."""
    threshold_type = threshold_var.get()
    if threshold_type == "average":
        threshold_slider.config(state=tk.NORMAL)
        lower_threshold_slider.config(state=tk.DISABLED)
        upper_threshold_slider.config(state=tk.DISABLED)
    elif threshold_type == "dual":
        threshold_slider.config(state=tk.DISABLED)
        lower_threshold_slider.config(state=tk.NORMAL)
        upper_threshold_slider.config(state=tk.NORMAL)
    else:
        # Default behavior for other types (e.g., simple, adaptive)
        threshold_slider.config(state=tk.NORMAL)
        lower_threshold_slider.config(state=tk.DISABLED)
        upper_threshold_slider.config(state=tk.DISABLED)

def apply_edge_detection():
    """Apply edge detection to the image."""
    global edited_img, current_img
    if original_img:
        current_img = edited_img.copy()
        img_cv = np.array(original_img.convert("L"))  # Convert to grayscale
        edges = cv2.Canny(img_cv, 100, 200)  # Apply Canny edge detection
        edited_img = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))  # Convert back to RGB
        update_display()

def apply_image_segmentation():
    """Apply basic image segmentation using K-means clustering."""
    global edited_img, current_img
    if original_img:
        current_img = edited_img.copy()
        img_cv = np.array(original_img)
        img_reshaped = img_cv.reshape((-1, 3))  # Reshape to a 2D array of pixels
        img_reshaped = np.float32(img_reshaped)

        # Define criteria and apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 4  # Number of clusters
        _, labels, centers = cv2.kmeans(img_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to uint8 and reshape to original image
        centers = np.uint8(centers)
        segmented_img = centers[labels.flatten()]
        segmented_img = segmented_img.reshape(img_cv.shape)

        edited_img = Image.fromarray(segmented_img)
        update_display()

def apply_grabcut_segmentation():
    """Apply GrabCut segmentation to the image."""
    global edited_img, current_img
    if original_img:
        current_img = edited_img.copy()
        img_cv = np.array(original_img)

        # Create a mask for GrabCut
        mask = np.zeros(img_cv.shape[:2], np.uint8)

        # Define a rectangle around the object to segment
        height, width = img_cv.shape[:2]
        rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))

        # Create background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Apply GrabCut
        cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Modify the mask to extract the foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        segmented_img = img_cv * mask2[:, :, np.newaxis]

        edited_img = Image.fromarray(segmented_img)
        update_display()

def apply_threshold_segmentation():
    """Apply threshold-based segmentation to the image."""
    global edited_img, current_img
    if original_img:
        current_img = edited_img.copy()
        img_cv = np.array(original_img)

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        # Apply a binary threshold
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Convert back to RGB for display
        segmented_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        edited_img = Image.fromarray(segmented_img)
        update_display()


# Create the main window
root = tk.Tk()
root.title("Image Editor")
root.geometry("800x700")
root.configure(bg="#f0f0f0")

# Center the window on the screen
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

center_window(root, 800, 700)

# Create a main frame to hold everything
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a canvas with scrollbar
canvas = tk.Canvas(main_frame, bg="#f0f0f0")
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")

# Configure the canvas
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Create a container frame for all content with fixed width
content_frame = tk.Frame(scrollable_frame, bg="#f0f0f0", width=750)
content_frame.pack(padx=20, pady=20)

# Header
header_frame = tk.Frame(content_frame, bg="#f0f0f0")
header_frame.pack(fill="x", pady=10)

title_label = tk.Label(header_frame, text="Image Editor", font=("Arial", 18, "bold"), bg="#f0f0f0")
title_label.pack()

# Open Image Button
open_button = tk.Button(content_frame, text="Open Image", command=open_image, 
                        font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
open_button.pack(pady=10)

# Image Display Frame
image_frame = tk.Frame(content_frame, bg="#f0f0f0")
image_frame.pack(pady=10)

# Original image
original_container = tk.Frame(image_frame, bg="#f0f0f0", padx=10, pady=5)
original_container.pack(side="left")

original_text = tk.Label(original_container, text="Original", font=("Arial", 12), bg="#f0f0f0")
original_text.pack()

original_label = tk.Label(original_container, bg="#e0e0e0", width=30, height=15)
original_label.pack()

# Edited image
edited_container = tk.Frame(image_frame, bg="#f0f0f0", padx=10, pady=5)
edited_container.pack(side="left")

edited_text = tk.Label(edited_container, text="Edited", font=("Arial", 12), bg="#f0f0f0")
edited_text.pack()

edited_label = tk.Label(edited_container, bg="#e0e0e0", width=30, height=15)
edited_label.pack()

# Section divider
ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

# Shape and Color Detection Frame
detection_frame = tk.Frame(content_frame, bg="#f0f0f0")
detection_frame.pack(fill="x", pady=10)

# Section title
detection_title = tk.Label(detection_frame, text="Detection Tools", font=("Arial", 14, "bold"), bg="#f0f0f0")
detection_title.pack(pady=5)

# Shape and Color Detection Buttons
detect_shapes_button = tk.Button(detection_frame, text="Detect Shapes", command=detect_shapes, 
                                font=("Arial", 12), bg="#FF9800", fg="white", padx=10, pady=5, state=tk.DISABLED)
detect_shapes_button.pack(side="left", padx=10, pady=5, expand=True)

detect_colors_button = tk.Button(detection_frame, text="Detect Colors", command=detect_colors, 
                                font=("Arial", 12), bg="#FF9800", fg="white", padx=10, pady=5, state=tk.DISABLED)
detect_colors_button.pack(side="left", padx=10, pady=5, expand=True)

# Section divider
ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

# Filters Section
filters_section = tk.Frame(content_frame, bg="#f0f0f0")
filters_section.pack(fill="x", pady=10)

filters_title = tk.Label(filters_section, text="Filters", font=("Arial", 14, "bold"), bg="#f0f0f0")
filters_title.pack(pady=5)

# Filters Frame - use Combobox instead of buttons
filters_frame = tk.Frame(filters_section, bg="#f0f0f0")
filters_frame.pack(pady=5)

filter_names = ["Sepia", "Vivid", "Color Leak", "Melbourne", "Amaro", 
                "Nashville", "Fade Warm", "Invert", "Darken", "Grayscale"]

selected_filter = tk.StringVar(value=filter_names[0])
filter_combobox = ttk.Combobox(filters_frame, textvariable=selected_filter, values=filter_names, state="readonly", width=20)
filter_combobox.grid(row=0, column=0, padx=5, pady=5)

apply_filter_button = tk.Button(filters_frame, text="Apply Filter", command=lambda: apply_filter(selected_filter.get()),
                               font=("Arial", 10), bg="#2196F3", fg="white", width=15, state=tk.DISABLED)
apply_filter_button.grid(row=0, column=1, padx=5, pady=5)

# Section divider
ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

# Adjustments Section
adjustments_section = tk.Frame(content_frame, bg="#f0f0f0")
adjustments_section.pack(fill="x", pady=10)

adjustments_title = tk.Label(adjustments_section, text="Adjustments", font=("Arial", 14, "bold"), bg="#f0f0f0")
adjustments_title.pack(pady=5)

# Create a notebook for tabs
adjustment_tabs = ttk.Notebook(adjustments_section)
adjustment_tabs.pack(fill="x", pady=5)

# RGB Tab
rgb_tab = tk.Frame(adjustment_tabs, bg="#f0f0f0", padx=10, pady=10)
adjustment_tabs.add(rgb_tab, text="RGB")

r_slider = tk.Scale(rgb_tab, from_=-100, to=100, orient=tk.HORIZONTAL, label="Red", 
                    length=300, bg="#f0f0f0", state=tk.DISABLED)
r_slider.pack(pady=5)

g_slider = tk.Scale(rgb_tab, from_=-100, to=100, orient=tk.HORIZONTAL, label="Green", 
                    length=300, bg="#f0f0f0", state=tk.DISABLED)
g_slider.pack(pady=5)

b_slider = tk.Scale(rgb_tab, from_=-100, to=100, orient=tk.HORIZONTAL, label="Blue", 
                    length=300, bg="#f0f0f0", state=tk.DISABLED)
b_slider.pack(pady=5)

rgb_adjust_button = tk.Button(rgb_tab, text="Apply RGB", command=adjust_rgb, 
                             font=("Arial", 10), bg="#2196F3", fg="white", state=tk.DISABLED)
rgb_adjust_button.pack(pady=5)

# Enhancers Tab
enhancers_tab = tk.Frame(adjustment_tabs, bg="#f0f0f0", padx=10, pady=10)
adjustment_tabs.add(enhancers_tab, text="Enhancers")

# Toggle for enhancers
enhancers_toggle = tk.IntVar(value=0)
enhancers_toggle_button = tk.Checkbutton(enhancers_tab, text="Enable Enhancers", 
                                        variable=enhancers_toggle, 
                                        command=lambda: toggle_enhancers(enhancers_toggle.get()), 
                                        bg="#f0f0f0", font=("Arial", 10), state=tk.DISABLED)
enhancers_toggle_button.pack(pady=5)

# Enhancer sliders
opacity_slider = tk.Scale(enhancers_tab, from_=0, to=100, orient=tk.HORIZONTAL, 
                         label="Opacity", length=300, bg="#f0f0f0", state=tk.DISABLED)
opacity_slider.pack(pady=5)

brightness_slider = tk.Scale(enhancers_tab, from_=0, to=200, orient=tk.HORIZONTAL, 
                            label="Brightness", length=300, bg="#f0f0f0", state=tk.DISABLED)
brightness_slider.pack(pady=5)

contrast_slider = tk.Scale(enhancers_tab, from_=0, to=200, orient=tk.HORIZONTAL, 
                          label="Contrast", length=300, bg="#f0f0f0", state=tk.DISABLED)
contrast_slider.pack(pady=5)

saturation_slider = tk.Scale(enhancers_tab, from_=0, to=200, orient=tk.HORIZONTAL, 
                            label="Saturation", length=300, bg="#f0f0f0", state=tk.DISABLED)
saturation_slider.pack(pady=5)

blur_slider = tk.Scale(enhancers_tab, from_=0, to=10, orient=tk.HORIZONTAL, 
                      label="Blur", length=300, bg="#f0f0f0", state=tk.DISABLED)
blur_slider.pack(pady=5)

apply_enhancers_button = tk.Button(enhancers_tab, text="Apply Enhancers", command=apply_enhancers, 
                                  font=("Arial", 10), bg="#2196F3", fg="white", state=tk.DISABLED)
apply_enhancers_button.pack(pady=5)

# Threshold Tab
threshold_tab = tk.Frame(adjustment_tabs, bg="#f0f0f0", padx=10, pady=10)
adjustment_tabs.add(threshold_tab, text="Threshold")

# Threshold type
threshold_type_frame = tk.Frame(threshold_tab, bg="#f0f0f0")
threshold_type_frame.pack(pady=5)

threshold_var = tk.StringVar(value="simple")
threshold_var.trace_add("write", update_threshold_controls)  # Bind the function to threshold_var changes

threshold_type_label = tk.Label(threshold_type_frame, text="Threshold Type:", bg="#f0f0f0")
threshold_type_label.pack(side="left", padx=5)

threshold_type_simple = tk.Radiobutton(threshold_type_frame, text="Simple", variable=threshold_var, 
                                      value="simple", bg="#f0f0f0", state=tk.DISABLED)
threshold_type_simple.pack(side="left", padx=5)

threshold_type_adaptive = tk.Radiobutton(threshold_type_frame, text="Adaptive", variable=threshold_var, 
                                        value="adaptive", bg="#f0f0f0", state=tk.DISABLED)
threshold_type_adaptive.pack(side="left", padx=5)

threshold_type_average = tk.Radiobutton(threshold_type_frame, text="Average", variable=threshold_var, 
                                        value="average", bg="#f0f0f0", state=tk.DISABLED)
threshold_type_average.pack(side="left", padx=5)

threshold_type_dual = tk.Radiobutton(threshold_type_frame, text="Dual", variable=threshold_var, 
                                     value="dual", bg="#f0f0f0", state=tk.DISABLED)
threshold_type_dual.pack(side="left", padx=5)

# Threshold sliders
threshold_slider = tk.Scale(threshold_tab, from_=0, to=255, orient=tk.HORIZONTAL, 
                           label="Threshold Value", length=300, bg="#f0f0f0", state=tk.DISABLED)
threshold_slider.pack(pady=5)

lower_threshold_slider = tk.Scale(threshold_tab, from_=0, to=255, orient=tk.HORIZONTAL, 
                                  label="Lower Threshold", length=300, bg="#f0f0f0", state=tk.DISABLED)
lower_threshold_slider.pack(pady=5)

upper_threshold_slider = tk.Scale(threshold_tab, from_=0, to=255, orient=tk.HORIZONTAL, 
                                  label="Upper Threshold", length=300, bg="#f0f0f0", state=tk.DISABLED)
upper_threshold_slider.pack(pady=5)

threshold_button = tk.Button(threshold_tab, text="Apply Threshold", command=apply_threshold, 
                            font=("Arial", 10), bg="#2196F3", fg="white", state=tk.DISABLED)
threshold_button.pack(pady=5)

# Segmentation Tab
segmentation_tab = tk.Frame(adjustment_tabs, bg="#f0f0f0", padx=10, pady=10)
adjustment_tabs.add(segmentation_tab, text="Segmentation")

# Segmentation Title
segmentation_title = tk.Label(segmentation_tab, text="Segmentation Methods", font=("Arial", 14, "bold"), bg="#f0f0f0")
segmentation_title.pack(pady=5)

# Edge Detection Button
edge_detection_button = tk.Button(segmentation_tab, text="Edge Detection", command=apply_edge_detection,
                                   font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.DISABLED)
edge_detection_button.pack(pady=5)

# Image Segmentation Button
image_segmentation_button = tk.Button(segmentation_tab, text="Image Segmentation", command=apply_image_segmentation,
                                       font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.DISABLED)
image_segmentation_button.pack(pady=5)

# Replace Semantic Segmentation Button
threshold_segmentation_button = tk.Button(segmentation_tab, text="Threshold Segmentation", command=apply_threshold_segmentation,
                                          font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.DISABLED)
threshold_segmentation_button.pack(pady=5)

# Replace Watershed Segmentation Button
grabcut_segmentation_button = tk.Button(segmentation_tab, text="GrabCut Segmentation", command=apply_grabcut_segmentation,
                                        font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.DISABLED)
grabcut_segmentation_button.pack(pady=5)

# Rotation Tab
rotation_tab = tk.Frame(adjustment_tabs, bg="#f0f0f0", padx=10, pady=10)
adjustment_tabs.add(rotation_tab, text="Rotation")

rotation_slider = tk.Scale(rotation_tab, from_=-180, to=180, orient=tk.HORIZONTAL, 
                           label="Rotation Angle (°)", length=300, bg="#f0f0f0", state=tk.NORMAL)
rotation_slider.pack(pady=5)

rotate_button = tk.Button(rotation_tab, text="Rotate Image", command=rotate_image, 
                          font=("Arial", 10), bg="#2196F3", fg="white", state=tk.NORMAL)
rotate_button.pack(pady=5)

# Add Mirror Buttons
mirror_horizontal_button = tk.Button(rotation_tab, text="Horizontal Mirror", command=lambda: mirror_image("horizontal"), 
                                     font=("Arial", 10), bg="#2196F3", fg="white", state=tk.NORMAL)
mirror_horizontal_button.pack(pady=5)

mirror_vertical_button = tk.Button(rotation_tab, text="Vertical Mirror", command=lambda: mirror_image("vertical"), 
                                   font=("Arial", 10), bg="#2196F3", fg="white", state=tk.NORMAL)
mirror_vertical_button.pack(pady=5)

# Translation Tab
translation_tab = tk.Frame(adjustment_tabs, bg="#f0f0f0", padx=10, pady=10)
adjustment_tabs.add(translation_tab, text="Translation")

# X Translation Slider
x_translation_slider = tk.Scale(translation_tab, from_=-200, to=200, orient=tk.HORIZONTAL, 
                                 label="X Translation (pixels)", length=300, bg="#f0f0f0", state=tk.NORMAL)
x_translation_slider.pack(pady=5)

# Y Translation Slider
y_translation_slider = tk.Scale(translation_tab, from_=-200, to=200, orient=tk.HORIZONTAL, 
                                 label="Y Translation (pixels)", length=300, bg="#f0f0f0", state=tk.NORMAL)
y_translation_slider.pack(pady=5)

# Translate Button
translate_button = tk.Button(translation_tab, text="Translate Image", command=translate_image, 
                              font=("Arial", 10), bg="#2196F3", fg="white", state=tk.NORMAL)
translate_button.pack(pady=5)

# Add Convolution Tab
convolution_tab = tk.Frame(adjustment_tabs, bg="#f0f0f0", padx=10, pady=10)
adjustment_tabs.add(convolution_tab, text="Convolution")

# Convolution Title
convolution_title = tk.Label(convolution_tab, text="Convolution Filters", font=("Arial", 14, "bold"), bg="#f0f0f0")
convolution_title.pack(pady=5)

# Convolution Buttons
convolution_filters = [
    ("Smooth", [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]),
    ("Gaussian Blur", [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]),
    ("Mean Removal", [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
    ("Sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    ("Emboss", [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
]

def apply_convolution(kernel):
    """Apply a convolution filter to the image."""
    global edited_img, current_img
    if original_img:
        # Save current state before modification
        current_img = edited_img.copy()
        
        # Convert the image to a NumPy array
        img_array = np.array(edited_img)
        
        # Ensure the image is in the correct format (RGB or Grayscale)
        if len(img_array.shape) == 3:  # RGB image
            # Apply the kernel to each channel separately
            channels = cv2.split(img_array)
            filtered_channels = [cv2.filter2D(channel, -1, np.array(kernel, dtype=np.float32)) for channel in channels]
            filtered_array = cv2.merge(filtered_channels)
        else:  # Grayscale image
            filtered_array = cv2.filter2D(img_array, -1, np.array(kernel, dtype=np.float32))
        
        # Convert the filtered array back to a PIL image
        filtered_img = Image.fromarray(np.clip(filtered_array, 0, 255).astype(np.uint8))
        
        # Update the edited image and display it
        edited_img = filtered_img
        update_display()

# Create buttons for each convolution filter
for filter_name, kernel in convolution_filters:
    btn = tk.Button(convolution_tab, text=filter_name, command=lambda k=kernel: apply_convolution(k),
                    font=("Arial", 10), bg="#2196F3", fg="white", width=15)
    btn.pack(pady=5)

# Kernel Operations Section
kernel_section = tk.Frame(content_frame, bg="#f0f0f0")
kernel_section.pack(fill="x", pady=10)

kernel_title = tk.Label(kernel_section, text="Kernel Operations", font=("Arial", 14, "bold"), bg="#f0f0f0")
kernel_title.pack(pady=5)

kernel_size_var = tk.StringVar(value="3")
kernel_type_var = tk.StringVar(value="average")
custom_kernel_var = tk.StringVar(value="")

kernel_size_label = tk.Label(kernel_section, text="Kernel Size:", bg="#f0f0f0")
kernel_size_label.pack(side="left", padx=5)
kernel_size_entry = tk.Entry(kernel_section, textvariable=kernel_size_var, width=5)
kernel_size_entry.pack(side="left", padx=5)

kernel_type_label = tk.Label(kernel_section, text="Kernel Type:", bg="#f0f0f0")
kernel_type_label.pack(side="left", padx=5)
kernel_type_menu = ttk.Combobox(kernel_section, textvariable=kernel_type_var, values=["average", "gaussian", "custom"], state="readonly")
kernel_type_menu.pack(side="left", padx=5)

custom_kernel_label = tk.Label(kernel_section, text="Custom Kernel (comma-separated):", bg="#f0f0f0")
custom_kernel_label.pack(side="left", padx=5)
custom_kernel_entry = tk.Entry(kernel_section, textvariable=custom_kernel_var, width=20)
custom_kernel_entry.pack(side="left", padx=5)

apply_kernel_button = tk.Button(kernel_section, text="Apply Kernel", command=apply_kernel, font=("Arial", 10), bg="#2196F3", fg="white", state=tk.DISABLED)
apply_kernel_button.pack(side="left", padx=10)

# Section divider
ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

# Grayscale and Binary Section
grayscale_binary_section = tk.Frame(content_frame, bg="#f0f0f0")
grayscale_binary_section.pack(fill="x", pady=10)

grayscale_binary_title = tk.Label(grayscale_binary_section, text="Grayscale & Binary", font=("Arial", 14, "bold"), bg="#f0f0f0")
grayscale_binary_title.pack(pady=5)

# Grayscale and Binary Buttons
grayscale_button = tk.Button(grayscale_binary_section, text="Grayscale", command=lambda: apply_filter("Grayscale"), 
                              font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.DISABLED)
grayscale_button.pack(side="left", padx=10, pady=5, expand=True)

binary_button = tk.Button(grayscale_binary_section, text="Black & White (Binary)", command=apply_threshold, 
                           font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.DISABLED)
binary_button.pack(side="left", padx=10, pady=5, expand=True)

# Add Histogram Button
histogram_button = tk.Button(grayscale_binary_section, text="Show Histogram", command=show_histogram, 
                              font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.NORMAL)
histogram_button.pack(side="left", padx=10, pady=5, expand=True)

binary_projection_button = tk.Button(grayscale_binary_section, text="Binary Projection", 
                                      command=binary_image_projection, 
                                      font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.NORMAL)
binary_projection_button.pack(side="left", padx=10, pady=5, expand=True)

# Section divider
ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

# Bottom Controls
bottom_frame = tk.Frame(content_frame, bg="#f0f0f0")
bottom_frame.pack(fill="x", pady=10)

# Action buttons
undo_button = tk.Button(bottom_frame, text="Undo", command=undo_changes, 
                       font=("Arial", 12), bg="#F44336", fg="white", width=10, state=tk.DISABLED)
undo_button.pack(side="left", padx=10)

reset_button = tk.Button(bottom_frame, text="Reset", command=reset_image, 
                        font=("Arial", 12), bg="#9C27B0", fg="white", width=10, state=tk.DISABLED)
reset_button.pack(side="left", padx=10)

save_button = tk.Button(bottom_frame, text="Save Image", command=save_image, font=("Arial", 12), bg="#4CAF50", fg="white")
save_button.pack(side="left", padx=5)

# Add the canvas and scrollbar to the main window
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Run the Application
root.mainloop()
