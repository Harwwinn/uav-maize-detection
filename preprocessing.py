import os
from PIL import Image
import shutil
import numpy as np


class ImagePatcher:
    def __init__(self, input_dir, output_dir,
                 patch_size=224, overlap=0, crop_size=512):
        """
        patch_size: tamaño final (224x224)
        crop_size: tamaño real del recorte antes de hacer resize
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.patch_size = patch_size      # tamaño final (224)
        self.crop_size = crop_size        # tamaño inicial (ej: 512)
        self.step = crop_size - overlap   # el paso depende del tamaño grande

        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def clean_output_folder(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        print(f"Carpeta limpia creada: {self.output_dir}")

    def process_folder(self):
        if not os.path.exists(self.input_dir):
            print(f"Error: La carpeta de entrada '{self.input_dir}' no existe.")
            return

        self.clean_output_folder()
        files = os.listdir(self.input_dir)
        total_patches = 0
        processed_images = 0

        print(f"Iniciando procesamiento de {len(files)} archivos...")

        for filename in files:
            if os.path.splitext(filename)[1].lower() not in self.valid_extensions:
                continue

            img_path = os.path.join(self.input_dir, filename)
            patches_count = self._slice_image(img_path, filename)

            if patches_count > 0:
                processed_images += 1
                total_patches += patches_count
                print(f"Procesada: {filename} -> {patches_count} recuadros.")

        print("-" * 30)
        print(f"RESUMEN:")
        print(f"Imágenes procesadas: {processed_images}")
        print(f"Total de recuadros generados: {total_patches}")
        print(f"Guardados en: {self.output_dir}")

    def _slice_image(self, img_path, filename):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_w, img_h = img.size

                base_name = os.path.splitext(filename)[0]
                count = 0

                for top in range(0, img_h, self.step):
                    for left in range(0, img_w, self.step):

                        right = left + self.crop_size
                        bottom = top + self.crop_size

                        if right > img_w or bottom > img_h:
                            continue

                        # recorte grande
                        patch_big = img.crop((left, top, right, bottom))

                        # filtro sobre el patch grande
                        if not self._is_leaf_patch(patch_big, green_threshold=0.05):
                            continue

                        # resize a 224x224
                        patch = patch_big.resize(
                            (self.patch_size, self.patch_size),
                            Image.LANCZOS
                        )

        
                        save_name = f"{base_name}_patch_{top}_{left}.jpg"
                        save_path = os.path.join(self.output_dir, save_name)
                        patch.save(save_path, quality=95)
                        count += 1

                return count

        except Exception as e:
            print(f"Error al procesar {filename}: {e}")
            return 0

    def _is_leaf_patch(self, patch, green_threshold=0.05):
        arr = np.array(patch)
        R = arr[:, :, 0].astype(np.float32)
        G = arr[:, :, 1].astype(np.float32)
        B = arr[:, :, 2].astype(np.float32)

        green_pixels = (G > R + 15) & (G > B + 15)
        proportion = np.mean(green_pixels)
        return proportion >= green_threshold


# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    INPUT_FOLDER = "./fotos_path"
    OUTPUT_FOLDER = "./fotos_tiles"

    # crop_size define EL TAMAÑO REAL del tile (ej: 512x512)
    patcher = ImagePatcher(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        patch_size=224,
        overlap=0,
        crop_size=1024  # <--- tamaño grande inicial
    )

    patcher.process_folder()
