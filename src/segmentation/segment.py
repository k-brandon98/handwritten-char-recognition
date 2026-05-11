import os
import cv2
import numpy as np


def load_image(image_path):
  """
  Load an image from disk.
  """
  image = cv2.imread(image_path)

  if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

  return image


def preprocess_for_segmentation(image):
  """
  Convert image to grayscale and threshold it so handwriting becomes white
  on a black background.
  """
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Light blur helps reduce small noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # Otsu threshold automatically chooses threshold value
  _, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  return binary


def find_character_boxes(binary_image):
  """
  Find bounding boxes around connected handwritten components.
  """
  contours, _ = cv2.findContours(
    binary_image,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
  )

  boxes = []

  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    boxes.append((x, y, w, h))

  return boxes


def filter_boxes(boxes, min_width=5, min_height=10, min_area=30):
  """
  Remove tiny boxes that are likely noise.
  """
  filtered = []

  for x, y, w, h in boxes:
    area = w * h

    if w >= min_width and h >= min_height and area >= min_area:
      filtered.append((x, y, w, h))

  return filtered


def sort_boxes_left_to_right(boxes):
  """
  Sort detected character boxes from left to right.
  """
  return sorted(boxes, key=lambda box: box[0])


def crop_characters(binary_image, boxes):
  """
  Crop each detected character from the binary image.
  """
  crops = []

  for x, y, w, h in boxes:
    crop = binary_image[y:y + h, x:x + w]
    crops.append(crop)

  return crops


def resize_character_crop(crop, output_size=28, padding=4):
    """
    Resize a cropped character to fit inside a 28x28 image while preserving aspect ratio.
    Output is white handwriting on black background.
    """
    h, w = crop.shape

    if h == 0 or w == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    # Create square canvas around character first
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = crop

    # Resize to leave padding around character
    inner_size = output_size - 2 * padding
    resized = cv2.resize(square, (inner_size, inner_size), interpolation=cv2.INTER_AREA)

    final_image = np.zeros((output_size, output_size), dtype=np.uint8)
    final_image[padding:padding + inner_size, padding:padding + inner_size] = resized

    return final_image


def segment_word(image_path, output_size=28):
    """
    Full segmentation pipeline.

    Returns:
        processed_characters: list of 28x28 character images
        boxes: sorted bounding boxes
        original_image: original loaded image
        binary_image: thresholded image
    """
    original_image = load_image(image_path)
    binary_image = preprocess_for_segmentation(original_image)

    boxes = find_character_boxes(binary_image)
    boxes = filter_boxes(boxes)
    boxes = sort_boxes_left_to_right(boxes)

    crops = crop_characters(binary_image, boxes)
    processed_characters = [
        resize_character_crop(crop, output_size=output_size)
        for crop in crops
    ]

    return processed_characters, boxes, original_image, binary_image


def save_segmented_characters(characters, output_dir="outputs/segments"):
    """
    Save each segmented character image.
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    for i, char_image in enumerate(characters):
        output_path = os.path.join(output_dir, f"char_{i}.png")
        cv2.imwrite(output_path, char_image)
        saved_paths.append(output_path)

    return saved_paths


def draw_bounding_boxes(image, boxes, output_path="outputs/segmentation_debug.png"):
    """
    Save a copy of the image with bounding boxes drawn around detected characters.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    debug_image = image.copy()

    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, debug_image)
    return output_path


def main():
    """
    Example usage:
        python src/segment.py data/custom_words/cat.png
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/segment.py <image_path>")
        return

    image_path = sys.argv[1]

    characters, boxes, original_image, _ = segment_word(image_path)

    print(f"Detected {len(characters)} characters.")
    print("Bounding boxes:", boxes)

    save_segmented_characters(characters)
    draw_bounding_boxes(original_image, boxes)

    print("Saved segmented characters to outputs/segments/")
    print("Saved debug image to outputs/segmentation_debug.png")


if __name__ == "__main__":
    main()