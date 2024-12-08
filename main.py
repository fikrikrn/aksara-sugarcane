import os
import shutil
from tile import tile_raster
from detect import process_images_with_yolo

input_raster = './input/new_tile_tif.tif'
input_dataset = './dataset/best.pt'
output_geojson = './output/output.geojson'

#temporary folder 
temp = './temp'
output_tile = os.path.join(temp,'tile')

def create_temp(temp):
    if not os.path.exists(temp):
        os.makedirs(temp)
        print(f"Folder temporary berhasil dibuat")

def clear_temp(temp):
    if os.path.exists(temp):
        shutil.rmtree(temp)
        print(f"Folder temporary berhasil dihapus")

def main():
    create_temp(temp)
    try:
        tile_raster(input_raster, output_tile, 640)
        process_images_with_yolo(input_dataset, output_tile, output_geojson)
    finally:
        clear_temp(temp)

if __name__ == "__main__":
    main()
    
