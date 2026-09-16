"""
app.py
======
Interactive wind farm energy assessment, structured to mirror the classic
WAsP inputs -> modelling -> outputs diagram:

    ENTREES                  MODELISATION            SORTIES
    - Donnees de vent /                                - PEA (production annuelle)
      rose des vents (upload possible)                 - Rose des vents
    - Rugosite du sol (upload possible)  -->  outil  -->  - Pertes de sillage (%)
    - Layout (upload possible)               Python      - Pertes post-sillage (6 postes)
    - Courbe de puissance (upload possible)              - Resultats (tableau) + Cf

Run with:
    streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from turbine import Turbine
from wind_farm import WindFarm
from wake_model import JensenWakeModel, EddyViscosityWakeModel, SimpleWakeModel
from power_curve import PowerCurve
from wind_resource import WindRose
from energy_yield import LossCascade, compute_energy_table

st.set_page_config(page_title="Wind Farm Energy Assessment", layout="wide")
st.title("🌬️ Wind Farm Energy Assessment")
st.caption(
    "Entrées (vent, disposition, rugosité, courbe de puissance — chacune éditable manuellement "
    "ou téléversée en CSV) → modélisation du sillage → sorties (PEA, pertes détaillées, Cf)."
)


def _find_col(df, *keywords):
    """Loosely match a column name containing all given keywords (case-insensitive)."""
    for col in df.columns:
        lname = col.lower().strip()
        if all(kw in lname for kw in keywords):
            return col
    return None


# ======================================================================
# 1. DONNEES DE VENT / ROSE DES VENTS
# ======================================================================
st.header("1. Données de vent — rose des vents")
col_wind, col_rose = st.columns([1, 1])

with col_wind:
    wind_mode = st.radio(
        "Source des données de vent",
        ["Simple (direction dominante)", "Tableau manuel (12 secteurs)",
         "Upload CSV — rose des vents", "Upload CSV — mesures brutes (direction, vitesse)"],
        key="wind_mode",
    )
    reference_height = st.number_input(
        "Hauteur de référence des données de vent [m]", value=10.0, step=1.0,
        help="Hauteur du mât de mesure (ou de la table/rose fournie). "
             "Le vent sera ensuite extrapolé vers la hauteur de moyeu via la rugosité (section 2).",
    )
    wind_rose_raw = None

    if wind_mode == "Simple (direction dominante)":
        prevailing_deg = st.slider("Direction dominante [° / Nord]", 0.0, 360.0, 270.0, step=5.0)
        prevailing_speed = st.slider("Vitesse moyenne du secteur dominant [m/s]", 3.0, 15.0, 7.0, step=0.5)
        spread_deg = st.slider("Étalement directionnel [°]", 10.0, 180.0, 60.0, step=5.0)
        weibull_k = st.slider("Paramètre de forme de Weibull k", 1.2, 3.5, 2.0, step=0.1)
        wind_rose_raw = WindRose.default_12_sector(
            prevailing_deg=prevailing_deg, prevailing_speed=prevailing_speed,
            spread_deg=spread_deg, k=weibull_k,
        )

    elif wind_mode == "Tableau manuel (12 secteurs)":
        st.caption("Fréquence [%], k et A (à la hauteur de référence) par secteur de 30°.")
        default_rose = WindRose.default_12_sector()
        default_df = pd.DataFrame({
            "Direction [deg]": default_rose.sector_centers_deg,
            "Frequence [%]": default_rose.frequencies * 100.0,
            "Weibull k": default_rose.weibull_k,
            "Weibull A [m/s]": default_rose.weibull_a,
        })
        edited_df = st.data_editor(default_df, num_rows="fixed", key="rose_editor")
        wind_rose_raw = WindRose(
            edited_df["Direction [deg]"].values,
            edited_df["Frequence [%]"].values / 100.0,
            edited_df["Weibull k"].values,
            edited_df["Weibull A [m/s]"].values,
        )

    elif wind_mode == "Upload CSV — rose des vents":
        st.caption(
            "Colonnes attendues : direction [°], fréquence [%], Weibull k, Weibull A [m/s] — "
            "une ligne par secteur. Noms de colonnes flexibles (ex: 'Direction [deg]', 'Frequence [%]', "
            "'Weibull k', 'Weibull A [m/s]')."
        )
        rose_file = st.file_uploader("CSV de rose des vents", type=["csv"], key="rose_csv")
        if rose_file is not None:
            try:
                rose_df = pd.read_csv(rose_file)
                wind_rose_raw = WindRose.from_table(rose_df)
                st.success(f"Rose des vents chargée : {wind_rose_raw.n_sectors()} secteurs.")
            except Exception as e:
                st.error(f"Erreur de lecture du CSV : {e}")
        else:
            st.info("Téléverse un CSV, ou une rose par défaut sera utilisée en attendant.")
            wind_rose_raw = WindRose.default_12_sector()

    else:  # Upload CSV - mesures brutes
        st.caption(
            "Colonnes attendues : direction [°] et vitesse [m/s] — une ligne par mesure "
            "(ex: relevé 10-minutes d'un mât de mesure). La rose des vents (fréquence par secteur "
            "+ ajustement Weibull) est calculée automatiquement à partir de ces mesures."
        )
        n_sectors_upload = st.number_input("Nombre de secteurs", min_value=4, max_value=36, value=12, step=1)
        meas_file = st.file_uploader("CSV de mesures brutes", type=["csv"], key="meas_csv")
        if meas_file is not None:
            try:
                meas_df = pd.read_csv(meas_file)
                dir_col = _find_col(meas_df, "dir")
                speed_col = _find_col(meas_df, "speed") or _find_col(meas_df, "vitesse")
                if dir_col is None or speed_col is None:
                    st.error(f"Colonnes non reconnues. Colonnes trouvées : {list(meas_df.columns)}")
                else:
                    wind_rose_raw = WindRose.from_measurements(
                        meas_df[dir_col].values, meas_df[speed_col].values, n_sectors=int(n_sectors_upload)
                    )
                    st.success(f"{len(meas_df)} mesures chargées, rose calculée sur {n_sectors_upload} secteurs.")
            except Exception as e:
                st.error(f"Erreur de lecture du CSV : {e}")
        else:
            st.info("Téléverse un CSV, ou une rose par défaut sera utilisée en attendant.")
            wind_rose_raw = WindRose.default_12_sector()

with col_rose:
    fig_rose = plt.figure(figsize=(4.3, 4.3))
    ax_rose = fig_rose.add_subplot(111, projection="polar")
    wind_rose_raw.plot_wind_rose(ax=ax_rose)
    st.pyplot(fig_rose)
    plt.close(fig_rose)
    st.metric("Vitesse moyenne (hauteur de référence)", f"{wind_rose_raw.overall_mean_speed():.2f} m/s")

# ======================================================================
# 2. RUGOSITE DU SOL
# ======================================================================
st.header("2. Rugosité du sol")
st.caption(
    "Utilisée pour extrapoler le vent de la hauteur de référence vers la hauteur de moyeu "
    "(profil logarithmique), en lieu et place de la correction de terrain complète de WAsP."
)

ROUGHNESS_PRESETS = {
    "Mer calme (z0 = 0,0002 m)": 0.0002,
    "Prairie (z0 = 0,03 m)": 0.03,
    "Cultures (z0 = 0,1 m)": 0.1,
    "Forêt (z0 = 1 m)": 1.0,
    "Ville dense (z0 = 2 m)": 2.0,
    "Personnalisé": None,
    "Upload CSV": None,
}
col_r1, col_r2 = st.columns([1, 1])
with col_r1:
    roughness_choice = st.selectbox("Classe de rugosité", list(ROUGHNESS_PRESETS.keys()), index=1)
    if roughness_choice == "Personnalisé":
        z0 = st.number_input("Longueur de rugosité z0 [m]", value=0.03, min_value=0.0001, step=0.01, format="%.4f")
    elif roughness_choice == "Upload CSV":
        st.caption("Colonne attendue : z0 [m] (une valeur, ou la moyenne sera prise si plusieurs lignes).")
        z0_file = st.file_uploader("CSV de rugosité", type=["csv"], key="z0_csv")
        if z0_file is not None:
            try:
                z0_df = pd.read_csv(z0_file)
                z0_col = _find_col(z0_df, "z0") or _find_col(z0_df, "roughness") or _find_col(z0_df, "rugos")
                if z0_col is None:
                    st.error(f"Colonne z0 non trouvée. Colonnes : {list(z0_df.columns)}")
                    z0 = 0.03
                else:
                    z0 = float(z0_df[z0_col].astype(float).mean())
                    st.success(f"z0 = {z0:.4f} m (moyenne de {len(z0_df)} valeur(s)).")
            except Exception as e:
                st.error(f"Erreur de lecture du CSV : {e}")
                z0 = 0.03
        else:
            st.info("Téléverse un CSV, ou z0 = 0,03 m sera utilisé en attendant.")
            z0 = 0.03
    else:
        z0 = ROUGHNESS_PRESETS[roughness_choice]
        st.caption(f"z0 = {z0} m")
with col_r2:
    HUB_HEIGHT_M = 120.0
    st.metric("Hauteur de moyeu (fixe)", f"{HUB_HEIGHT_M:.0f} m")
    speed_factor = np.log(HUB_HEIGHT_M / z0) / np.log(reference_height / z0)
    st.metric("Facteur d'extrapolation du vent", f"×{speed_factor:.2f}")

wind_rose = wind_rose_raw.rescaled_to_height(HUB_HEIGHT_M, z0, reference_height=reference_height)
st.caption(f"Vitesse moyenne à la hauteur de moyeu ({HUB_HEIGHT_M:.0f} m) après extrapolation : "
          f"**{wind_rose.overall_mean_speed():.2f} m/s**.")

# ======================================================================
# 3. LAYOUT (DISPOSITION DU PARC)
# ======================================================================
st.header("3. Layout — disposition du parc")
ROTOR_DIAMETER_M = 170.0
layout_mode = st.radio("Source de la disposition", ["Grille automatique", "Upload CSV (x, y)"], horizontal=True)

turbines = []
if layout_mode == "Grille automatique":
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        rows = st.slider("Rangées", 1, 6, 2)
    with col_l2:
        cols = st.slider("Colonnes", 1, 6, 3)
    with col_l3:
        spacing = st.slider("Espacement [m]", 150.0, 1200.0, 5 * ROTOR_DIAMETER_M, step=10.0)
    tid = 1
    for r in range(rows):
        for c in range(cols):
            turbines.append(Turbine(id=f"T{tid:02d}", x=c * spacing, y=r * spacing,
                                    diameter=ROTOR_DIAMETER_M, hub_height=HUB_HEIGHT_M))
            tid += 1
else:
    st.caption("Colonnes attendues : x, y [m] (et optionnellement 'id'). Voir layout_example.csv.")
    csv_file = st.file_uploader("CSV avec colonnes x, y (et optionnellement 'id')", type=["csv"])
    if csv_file is not None:
        layout_df = pd.read_csv(csv_file)
        layout_df.columns = [c.strip().lower() for c in layout_df.columns]
        for i, row in layout_df.iterrows():
            tid = str(row["id"]) if "id" in layout_df.columns else f"T{i+1:02d}"
            turbines.append(Turbine(id=tid, x=float(row["x"]), y=float(row["y"]),
                                    diameter=ROTOR_DIAMETER_M, hub_height=HUB_HEIGHT_M))
    else:
        st.info("Téléverse un CSV avec des colonnes x, y pour définir la disposition.")

st.caption(f"Diamètre de rotor (fixe) : {ROTOR_DIAMETER_M:.0f} m — Hauteur de moyeu (fixe) : {HUB_HEIGHT_M:.0f} m.")

wake_choice = st.selectbox("Modèle de sillage", ["Jensen", "Eddy viscosity (Ainslie)", "Simple"])
wake_models = {
    "Jensen": JensenWakeModel(wake_decay=0.05),
    "Eddy viscosity (Ainslie)": EddyViscosityWakeModel(ambient_turbulence=0.10, thrust_coeff=0.8),
    "Simple": SimpleWakeModel(alpha=0.05, beta=0.5),
}
wake_model = wake_models[wake_choice]

if turbines:
    farm = WindFarm(name="Parc évalué")
    farm.add_turbines(turbines)
    fig_layout, ax_layout = plt.subplots(figsize=(6, 4.5))
    for t in farm.turbines:
        ax_layout.add_patch(plt.Circle((t.x, t.y), t.diameter / 2.0, facecolor="steelblue",
                                        edgecolor="black", alpha=0.9, zorder=3))
        ax_layout.annotate(t.id, (t.x, t.y), xytext=(0, 12), textcoords="offset points",
                           ha="center", fontsize=8, zorder=4)
    xs = [t.x for t in farm.turbines]; ys = [t.y for t in farm.turbines]
    pad = max(ROTOR_DIAMETER_M * 3, 100)
    ax_layout.set_xlim(min(xs) - pad, max(xs) + pad)
    ax_layout.set_ylim(min(ys) - pad, max(ys) + pad)
    ax_layout.set_aspect("equal", adjustable="box")
    ax_layout.set_xlabel("Easting [m]"); ax_layout.set_ylabel("Northing [m]")
    ax_layout.set_title(f"{farm.name} : {len(farm.turbines)} éoliennes")
    ax_layout.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig_layout)
    plt.close(fig_layout)
else:
    farm = None

# ======================================================================
# 4. COURBE DE PUISSANCE
# ======================================================================
st.header("4. Courbe de puissance")
power_mode = st.radio("Source de la courbe de puissance", ["Paramétrique", "Upload CSV (courbe fabricant)"],
                       horizontal=True)

if power_mode == "Paramétrique":
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        rated_power = st.number_input("Puissance nominale [kW]", value=6200.0, step=100.0)
    with col_p2:
        cut_in = st.number_input("Vitesse de démarrage (cut-in) [m/s]", value=3.0, step=0.5)
    with col_p3:
        rated_speed = st.number_input("Vitesse nominale [m/s]", value=10.5, step=0.5)
    with col_p4:
        cut_out = st.number_input("Vitesse d'arrêt (cut-out) [m/s]", value=25.0, step=1.0)
    power_curve = PowerCurve(cut_in=cut_in, rated_speed=rated_speed, cut_out=cut_out, rated_power=rated_power)
else:
    st.caption(
        "Colonnes attendues : vitesse de vent [m/s] et puissance [kW] — une ligne par point de la "
        "courbe fabricant. Interpolation linéaire entre les points ; puissance nulle en dehors."
    )
    pc_file = st.file_uploader("CSV de courbe de puissance", type=["csv"], key="power_csv")
    if pc_file is not None:
        try:
            pc_df = pd.read_csv(pc_file)
            power_curve = PowerCurve.from_table(pc_df)
            rated_power = power_curve.rated_power
            st.success(f"Courbe chargée : {len(power_curve.speed_table)} points, "
                      f"P nominale = {rated_power:.0f} kW.")
            fig_pc, ax_pc = plt.subplots(figsize=(5, 2.8))
            ax_pc.plot(power_curve.speed_table, power_curve.power_table, marker="o", color="steelblue")
            ax_pc.set_xlabel("Vitesse de vent [m/s]"); ax_pc.set_ylabel("Puissance [kW]")
            ax_pc.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig_pc)
            plt.close(fig_pc)
        except Exception as e:
            st.error(f"Erreur de lecture du CSV : {e}")
            power_curve = PowerCurve()
            rated_power = power_curve.rated_power
    else:
        st.info("Téléverse un CSV, ou une courbe paramétrique par défaut sera utilisée en attendant.")
        power_curve = PowerCurve()
        rated_power = power_curve.rated_power

st.caption(
    "💡 Le facteur de charge (Cf) ne dépend **pas** de la puissance nominale seule (mode paramétrique) : "
    "augmenter la puissance nominale sans changer la vitesse nominale multiplie l'énergie et la puissance "
    "installée dans les mêmes proportions, donc le Cf reste inchangé. C'est la **vitesse nominale** "
    "(ou la forme de la courbe fabricant) qui détermine vraiment le Cf."
)

# ======================================================================
# 5. PERTES POST-SILLAGE (INDIVIDUELLEMENT ÉDITABLES)
# ======================================================================
st.header("5. Pertes post-sillage")
st.caption("Chaque poste de perte est réglable séparément, appliqué multiplicativement après le sillage.")
loss_cols = st.columns(6)
labels_defaults = [
    ("Shadow flicker [%]", 1.6), ("Turbine/flow effect [%]", 1.08),
    ("Availability [%]", 3.0), ("Electrical [%]", 2.0),
    ("Turbine performance [%]", 3.6), ("Environment [%]", 1.0),
]
loss_values = []
for col, (label, default) in zip(loss_cols, labels_defaults):
    with col:
        loss_values.append(st.number_input(label, value=default, step=0.1, min_value=0.0, max_value=50.0))

losses = LossCascade(
    shadow_flicker=loss_values[0] / 100.0,
    turbine_flow_effect=loss_values[1] / 100.0,
    availability=loss_values[2] / 100.0,
    electrical=loss_values[3] / 100.0,
    turbine_performance=loss_values[4] / 100.0,
    environment=loss_values[5] / 100.0,
)
st.caption(f"Facteur combiné de pertes post-sillage : {losses.combined_factor() * 100:.2f}% retenu "
          f"({(1 - losses.combined_factor()) * 100:.2f}% de perte totale post-sillage).")

# ======================================================================
# 6. RESULTATS
# ======================================================================
st.header("6. Résultats — PEA et facteur de charge")

if farm is not None and st.button("▶ Lancer le calcul", type="primary"):
    with st.spinner("Intégration de la rose des vents sur le modèle de sillage..."):
        rows_data, totals = compute_energy_table(
            farm, wake_model, power_curve, wind_rose, losses, rated_power_kw=rated_power,
        )
    df = pd.DataFrame(rows_data)
    numeric_cols = df.columns.drop("Turbine")
    st.dataframe(df.style.format({c: "{:.1f}" for c in numeric_cols}), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Puissance installée", f"{totals['Total rated power [kW]'] / 1000:.2f} MW")
    m2.metric("PEA brute", f"{totals['Total gross energy [MWh/yr]']:.0f} MWh/an")
    m3.metric("PEA nette", f"{totals['Total net energy [MWh/yr]']:.0f} MWh/an")
    m4.metric("Facteur de charge Cf", f"{totals['Capacity factor Cf'] * 100:.1f} %")

    m5, m6, m7 = st.columns(3)
    m5.metric("Perte de sillage totale", f"{totals['Total wake loss [%]']:.2f} %")
    m6.metric("Pertes post-sillage combinées", f"{totals['Combined post-wake loss [%]']:.2f} %")
    m7.metric("PEA après sillage (avant pertes post-sillage)",
              f"{totals['Total energy after wake effect [MWh/yr]']:.0f} MWh/an")

    with st.expander("Formule du Cf"):
        st.latex(
            r"C_f = \dfrac{\text{Énergie nette totale [kWh/an]}}"
            r"{\sum_i P_{nominale,i} \times 8760\ \text{h/an}}"
        )
        st.write(
            f"= {totals['Total net energy [MWh/yr]']*1000:,.0f} kWh "
            f"/ ({totals['Total rated power [kW]']:,.0f} kW × 8760 h) "
            f"= {totals['Capacity factor Cf']*100:.2f} %"
        )
    with st.expander("D'où vient la perte de sillage totale ?"):
        st.latex(
            r"\text{Perte de sillage totale} = \left(1 - \dfrac{\text{Énergie après sillage}}"
            r"{\text{Énergie brute}}\right) \times 100"
        )
        st.write(
            f"= (1 − {totals['Total energy after wake effect [MWh/yr]']:.0f} / "
            f"{totals['Total gross energy [MWh/yr]']:.0f}) × 100 "
            f"= {totals['Total wake loss [%]']:.2f} %"
        )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("⬇ Télécharger le tableau (CSV)", csv_buffer.getvalue(),
                       file_name="wind_farm_energy_table.csv", mime="text/csv")
elif farm is None:
    st.info("Définis d'abord une disposition de parc (section 3) pour lancer le calcul.")
else:
    st.info("Règle tes hypothèses ci-dessus, puis clique sur **Lancer le calcul**.")
