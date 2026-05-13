from src.segmentation.segment import (
    segment_word,
    save_segmented_characters,
    draw_bounding_boxes
)

image_path = "data/custom_words/cat.jpeg"

characters, boxes, original_image, _ = segment_word(image_path)

print(f"Detected {len(characters)} characters.")
print("Bounding boxes:", boxes)

save_segmented_characters(characters)
draw_bounding_boxes(original_image, boxes)