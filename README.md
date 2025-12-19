# 🛣️ Road Segmentation with U-Net

Ce projet implémente un réseau de neurones U-Net avec PyTorch pour effectuer de la segmentation sémantique de routes sur des images satellites.

L'objectif est de détecter automatiquement les pixels appartenant à une route (affichés en couleur superposée) à partir d'une image aérienne.

## ✨ Fonctionnalités Clés
- **Architecture U-Net** : Modèle encoder-decoder performant pour la segmentation biomédicale et satellite.

- **Custom Loss (Dice + BCE)** : Combinaison de Binary Cross Entropy et de Dice Loss pour gérer le déséquilibre des classes (les routes occupent peu de place sur l'image).

- **Data Augmentation (Albumentations)** : Utilisation de rotations et de miroirs (flips) pour rendre le modèle robuste aux changements d'orientation.

- **Robustesse des données** : Binarisation automatique des masques (seuil à 0.5) pour éviter les erreurs d'interpolation.


## 🧬 Origine des Données et du Modèle
Ce projet se distingue par une approche entièrement artisanale ("from scratch"), de la collecte des données jusqu'à l'architecture du réseau.

### 🗺️ Dataset "Fait Maison"
Contrairement aux projets classiques utilisant des bases de données massives (comme Kaggle ou Cityscapes), le dataset a été **construit manuellement** :

- **Source** : Captures d'écran satellites (Google Earth).

- **Annotation** : Création manuelle des masques de segmentation (pixel-perfect) via un logiciel de retouche.

- **Taille du Dataset** : Le modèle a atteint ses performances avec un jeu de données extrêmement réduit de seulement 20 images.

- **Note** : Cela démontre l'efficacité de la stratégie de Data Augmentation mise en place pour compenser le manque de volume.

### 🧠 Modèle U-Net "Custom"
Le réseau de neurones n'est pas un import de librairie pré-existante.

- L'architecture **U-Net** a été codée couche par couche en **PyTorch**.

- L'implémentation comprend la construction explicite de l'encodeur (contraction path), du goulot d'étranglement (bottleneck) et du décodeur (expansive path) avec les skip connections.


## 🛠️ Installation
1. Cloner le projet (ou télécharger les fichiers).

2. Installer les dépendances via `requirements.txt` :


## 📂 Structure du Projet
```bash
.
├── dataset/
│   ├── images/         # Images satellites d'entraînement (.jpg)
│   └── masks/          # Masques binaires correspondants (.gif/.png)
├── model.py            # Architecture du réseau U-Net
├── my_dataset.py       # Chargement des données + Data Augmentation
├── train.py            # Script d'entraînement (Training Loop)
├── predict.py          # Script de test/prédiction sur une image
└── mon_UNET.pth        # Poids du modèle sauvegardés (après entraînement)
```


## 🚀 Utilisation
### 1. Entraînement du modèle
Pour lancer l'entraînement sur votre dataset :
```bash
python train.py
```

- **Configuration** : Vous pouvez modifier les hyperparamètres (Epochs, Learning Rate, Batch Size) directement au début du fichier `train.py`.

- **Suivi** : Une courbe de Loss (`courbe_loss.png`) est générée et mise à jour en temps réel à chaque époque.

- **Sauvegarde** : Le modèle final est sauvegardé sous `mon_UNET.pth`.

### 2. Prédiction (Inférence)
Pour tester le modèle sur une nouvelle image (ex: `Test_image.jpg`) :

1. Assurez-vous que le fichier `mon_UNET.pth` existe.

2. Modifiez le chemin de l'image cible dans `predict.py`.

3. Lancez la commande :

```bash
python predict.py
```

Le script affichera l'image avec la route détectée en superposition (rouge/rose).


## ⚙️ Détails Techniques
- **Input** : Images RGB redimensionnées ou croppées (256x256).

- **Output** : Masque binaire (0 = Fond, 1 = Route).

- **Optimiseur** : Adam (`lr=1e-3`).

- **Seuil de décision** : Les pixels sont considérés comme "Route" si la probabilité dépasse 20% (configurable).