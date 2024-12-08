import os
import numpy as np
import rasterio
from rasterio.windows import Window

def tile_raster(input_tif, output_dir, tile_size):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        num_bands = src.count 
        print(f"Processing {input_tif} with {num_bands} band(s).")

        num_tiles_x = (width + tile_size - 1) // tile_size
        num_tiles_y = (height + tile_size - 1) // tile_size

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                x_start = i * tile_size
                y_start = j * tile_size
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)

                window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
                transform = src.window_transform(window)
                tile_data = src.read(window=window)  
                tile_data = np.clip(tile_data, 0, 255).astype(np.uint8)
                tile_filename = os.path.join(output_dir, f"tile_{i}_{j}.tif")

                with rasterio.open(
                    tile_filename,
                    'w',
                    driver='GTiff',
                    height=tile_data.shape[1],
                    width=tile_data.shape[2],
                    count=num_bands, 
                    dtype='uint8',  
                    crs=src.crs,
                    transform=transform
                ) as dst:
                    dst.write(tile_data)

                print(f"Tile saved: {tile_filename}")