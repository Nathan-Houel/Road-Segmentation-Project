import cv2
import matplotlib.pyplot as plt

# Charger l'image
image_path = "Test_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Afficher l'image
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.title("Image satellite chargée")
plt.axis('off')
plt.show()

print(f"Dimensions de l'image: {image.shape}")