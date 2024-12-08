import os
from PIL import Image
import numpy as np
import rasterio
from rasterio.windows import Window
import random

tif_file = "input.tif"
output_dir = "prep_dataset"
tile_pixel_size = 640
coverage_percentage = 0.3

def create_random_tiles(tif_path, output_folder, tile_size, coverage):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        print(f"Ukuran gambar: {width}x{height}")
        tiles_x = (width + tile_size - 1) // tile_size  
        tiles_y = (height + tile_size - 1) // tile_size
        total_tiles = tiles_x * tiles_y
        selected_tiles = int(total_tiles * coverage)
        print(f"Total tile: {total_tiles}, Tile terpilih: {selected_tiles}")
        all_indices = [(i, j) for i in range(tiles_y) for j in range(tiles_x)]
        selected_indices = random.sample(all_indices, selected_tiles)

        for i, j in selected_indices:
            window = Window(j * tile_size, i * tile_size, tile_size, tile_size)
            tile = src.read(window=window)
            
            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                padded_tile = np.zeros((tile.shape[0], tile_size, tile_size), dtype=tile.dtype)
                padded_tile[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = padded_tile
            
            rgb_tile = tile[:3, :, :].transpose(1, 2, 0)
            img = Image.fromarray(rgb_tile)

            tile_name = f"tile_{i}_{j}.jpg"
            img_path = os.path.join(output_folder, tile_name)
            img.save(img_path, "JPEG")
            print(f"Tile disimpan: {img_path}")
            
create_random_tiles(tif_file, output_dir, tile_pixel_size, coverage_percentage)