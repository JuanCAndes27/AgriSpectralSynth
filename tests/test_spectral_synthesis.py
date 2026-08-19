import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio
from affine import Affine

def main():
    print("🚀 Iniciando procesamiento masivo de las 1,199 imágenes...")

    # Rutas absolutas explícitas
    base_dir = Path("D:/AgriSpectralSynth")
    raw_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "data" / "processed"

    nir_dir = output_dir / "nir"
    ndvi_raw_dir = output_dir / "ndvi_raw"
    ndvi_visual_dir = output_dir / "ndvi_visual"
    canopy_dir = output_dir / "canopy_mask"

    for folder in [nir_dir, ndvi_raw_dir, ndvi_visual_dir, canopy_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # Buscar todas las imágenes válidas
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    image_paths = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]

    if not image_paths:
        print(f"❌ ERROR: No se encontraron imágenes en {raw_dir}")
        return

    total = len(image_paths)
    print(f"📁 Total de imágenes detectadas: {total}")
    cmap = plt.get_cmap("jet")
    success_count = 0

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            raw_filename = img_path.stem
            base_name = raw_filename.replace("_NDVI", "").replace("_NIR", "").replace("_canopy", "")
            
            # Cargar imagen RGB
            img = Image.open(img_path).convert("RGB")
            rgb_array = np.array(img, dtype=np.float32) / 255.0

            red_band = rgb_array[:, :, 0]
            green_band = rgb_array[:, :, 1]

            # Procesamiento espectral
            nir_band = np.clip(green_band * 1.6 - red_band * 0.4 + 0.1, 0.0, 1.0)
            ndvi_map = (nir_band - red_band) / (nir_band + red_band + 1e-8)

            mask_canopy = (ndvi_map > 0.35) & (green_band > red_band)
            canopy_img_array = (mask_canopy.astype(np.uint8)) * 255

            # 1. Guardar NIR
            Image.fromarray((nir_band * 255).astype(np.uint8)).save(nir_dir / f"{base_name}_NIR.png")

            # 2. Guardar Canopy
            Image.fromarray(canopy_img_array).save(canopy_dir / f"{base_name}_canopy.png")

            # 3. Guardar NDVI TIF (.tif)
            with rasterio.open(
                ndvi_raw_dir / f"{base_name}_NDVI.tif",
                'w', driver='GTiff',
                height=ndvi_map.shape[0], width=ndvi_map.shape[1],
                count=1, dtype='float32',
                transform=Affine.identity()
            ) as dst:
                dst.write(ndvi_map.astype(np.float32), 1)

            # 4. Guardar NDVI Visual con paleta Jet (.png)
            ndvi_clean = np.nan_to_num(ndvi_map, nan=0.0)
            ndvi_normalized = np.clip((ndvi_clean + 0.2) / 1.2, 0.0, 1.0)
            ndvi_rgba = cmap(ndvi_normalized)
            ndvi_rgb_8bit = (ndvi_rgba[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(ndvi_rgb_8bit).save(ndvi_visual_dir / f"{base_name}_NDVI.png")

            success_count += 1
            
            if idx % 50 == 0 or idx == total:
                print(f"[{idx}/{total}] Procesadas correctamente...")

        except Exception as e:
            print(f"❌ Error en {img_path.name}: {e}")

    print(f"\n🎉 ¡Procesamiento masivo terminado! {success_count}/{total} imágenes actualizadas en la ruta oficial.")

if __name__ == "__main__":
    main()