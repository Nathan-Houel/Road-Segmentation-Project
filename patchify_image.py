import cv2
import os

# Configuration
IMAGE_PATH = "Test_image.jpg"
PATCH_SIZE = 256
OUTPUT_FOLDER = "patches"

# Créer dossier de sortie
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Charger l'image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"ERREUR : Je ne trouve pas l'image '{IMAGE_PATH}'. Vérifiez le nom !")
    exit()


height, width, _ = image.shape
print(f"Image chargée : {width}x{height} pixels.")

# Découpage
count = 0
for y in range(0, height, PATCH_SIZE):
    for x in range(0, width, PATCH_SIZE):

        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
            continue

        #Sauvegarde
        cv2.imwrite(f"{OUTPUT_FOLDER}/patch_{count}.jpg", patch)
        count += 1

print(f"Succès ! {count} patches créés dans le dossier '{OUTPUT_FOLDER}'.")