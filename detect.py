import os
from ultralytics import YOLO
import rasterio
import numpy as np
import json
import cv2

def pixel_to_utm(x_pixel, y_pixel, bounds, GSD_x, GSD_y):
    x_utm = bounds.left + x_pixel * GSD_x
    y_utm = bounds.top + y_pixel * GSD_y
    return x_utm, y_utm

def process_images_with_yolo(model_path, folder_path, output_geojson_path):
    model = YOLO(model_path)
    all_detections = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            tiff_path = os.path.join(folder_path, filename)

            try:
                with rasterio.open(tiff_path) as src:
                    img = src.read([1, 2, 3])
                    img = np.moveaxis(img, 0, -1)

                    bounds = src.bounds
                    transform = src.transform
                    GSD_x, GSD_y = transform[0], transform[4]
                
                results = model(img)
                names = model.names

                for result in results:
                    if result.masks is not None:
                        for i, mask in enumerate(result.masks.cpu().numpy()):
                            mask_2d = mask.data.squeeze().astype(np.uint8)
                            contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for contour in contours:
                                contour_utm = []
                                for point in contour:
                                    x_pixel, y_pixel = point[0]
                                    x_utm, y_utm = pixel_to_utm(x_pixel, y_pixel, bounds, GSD_x, GSD_y)
                                    contour_utm.append([x_utm, y_utm])
                                if contour_utm[0] != contour_utm[-1]:
                                    contour_utm.append(contour_utm[0])
                                class_id = int(result.boxes.cls[i].cpu().numpy())
                                class_name = names[class_id]
                                all_detections.append({
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [contour_utm]
                                    },
                                    "properties": {
                                        "class_id": class_id,
                                        "class_name": class_name,
                                        "source_image": filename
                                    }
                                })

                print(f"Processed {filename}.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    if all_detections:
        geojson_data = {
            "type": "FeatureCollection",
            "features": all_detections
        }
        with open(output_geojson_path, "w") as f:
            json.dump(geojson_data, f, indent=4)

        print(f"Combined GeoJSON with segmentation masks saved to {output_geojson_path}")
    else:
        print("Tidak ada tebu yang terdeteksi.")
