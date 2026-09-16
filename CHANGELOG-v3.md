# Wind Farm Energy Assessment — v3 (uploads + pertes détaillées)

## Installation
```bash
pip install -r requirements.txt
```

## Lancement
```bash
streamlit run app.py
```

## Ce qui a changé par rapport à la v2

**1. Données de vent** — 4 modes désormais :
- Simple (direction dominante)
- Tableau manuel (12 secteurs)
- **Upload CSV — rose des vents** (colonnes : direction, fréquence, Weibull k, Weibull A) — voir `wind_rose_example.csv`
- **Upload CSV — mesures brutes** (colonnes : direction, vitesse, une ligne par relevé) — la rose est calculée
  automatiquement (fréquence par secteur + ajustement Weibull par la méthode des moments de Justus) — voir
  `wind_measurements_example.csv`

**2. Rugosité** — ajout d'une option **Upload CSV** (colonne `z0`), en plus des presets et de la saisie
manuelle — voir `roughness_example.csv`.

**3. Layout** — inchangé (grille automatique ou upload CSV x,y — voir `layout_example.csv`).

**4. Courbe de puissance** — ajout d'un mode **Upload CSV** (colonnes vitesse/puissance, interpolation
linéaire), en plus du mode paramétrique (cut-in/nominale/cut-out) — voir `power_curve_example.csv`.

**5. Pertes post-sillage** — retour aux **6 postes individuellement réglables** (shadow flicker,
turbine/flow effect, disponibilité, électrique, performance turbine, environnement), au lieu du pourcentage
unique simplifié de la v2. Chaque poste est appliqué multiplicativement, dans l'ordre.

**6. Résultats** — ajout de la métrique **Perte de sillage totale [%]**, calculée comme
`(1 − énergie après sillage / énergie brute) × 100`, avec le détail de la formule dans un expander.

## Fichiers

- `app.py` — interface Streamlit (toutes les entrées ci-dessus).
- `wind_resource.py` — `WindRose` avec `from_table()` (rose CSV) et `from_measurements()` (mesures brutes).
- `power_curve.py` — `PowerCurve` avec `from_table()` (courbe CSV, interpolation linéaire).
- `energy_yield.py` — `LossCascade` (6 postes) + calcul gross/wake/net + Cf + perte de sillage totale.
- `turbine.py`, `wind_farm.py`, `wake_model.py`, `visualization.py` — inchangés.
- `*_example.csv` — modèles pour chaque type d'upload.

## Limites toujours d'actualité

- Pas de correction orographique liée au relief (seule la rugosité/hauteur est prise en compte).
- Le coefficient de poussée (Ct) des modèles de sillage reste fixe (0,8 par défaut).
- L'ajustement Weibull sur mesures brutes utilise une approximation par moments (Justus), pas un maximum
  de vraisemblance — suffisant pour l'estimation mais moins précis qu'un outil dédié (ex: WAsP).
