# ================================
# CARGA DEL MODELO Y DATOS
# ================================
# Reemplaza la URL con la ruta real a tu archivo en GitHub
MODEL_URL = "https://raw.githubusercontent.com/tu_usuario/tu_repo/rama/FFNN_Total.pkl"
DATA_URL = "https://raw.githubusercontent.com/tu_usuario/tu_repo/rama/data_paneles.xlsx"


@st.cache_data
def cargar_modelo():
    response = requests.get(MODEL_URL)
    with open("modelo.pkl", "wb") as f:
        f.write(response.content)
    model = joblib.load("modelo.pkl")
    return model

@st.cache_data
def cargar_data_referencia():
    content = requests.get(DATA_URL).content
    df = pd.read_excel(io.BytesIO(content))
    return df

model_total = cargar_modelo()
df_ref = cargar_data_referencia()

# ================================
# INTERFAZ DE ENTRADA
# ================================
st.title("Dashboard Predictivo de Productividad de Paneles Prefabricados")
st.subheader("Ingreso de datos técnicos del panel")

input_features = [
    'Area', 'Concrete_Quantity_Panels', 'Windows&Doors', 'Ribs', 'Plywood_Area',
    'Total_Rebar_Length', 'Rebar_Cuts', 'Rebar_Chairs', 'Area_Wire_Mesh', 'Mesh_Chairs',
    'Styrofoam_Ribs', 'Styrofoam_Cuts', 'Styrofoam_Boards', 'Conduits', 'Anchors',
    'Preparation (Worker)', 'Formwork (Worker)', 'Assembly (Worker)', 'Concreting (Worker)',
    'Lifting (Worker)', 'Finishing  (Worker)'
]

user_input = {}
for var in input_features:
    default_val = df_ref[var].median() if var in df_ref.columns else 0
    user_input[var] = st.number_input(var, min_value=0.0, value=float(default_val), format="%f")

X = pd.DataFrame([user_input])

# ================================
# PREDICCIÓN TOTAL
# ================================
total_time_min = model_total.predict(X)[0]  # en minutos
area_m2 = user_input['Area'] * 0.092903  # conversión de sqft a m²
productividad = (total_time_min / 60) / area_m2  # hh/m²

# ================================
# KPIS
# ================================
st.header("Indicadores Clave (KPIs)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Productividad Estimada (hh/m²)", f"{productividad:.2f}")
with col2:
    st.metric("Área del Panel (m²)", f"{area_m2:.2f}")
with col3:
    st.metric("Volumen de Concreto (m³)", f"{user_input['Concrete_Quantity_Panels']:.2f}")

# ================================
# GRAFICO DE BARRAS POR COMPONENTES
# ================================
st.header("Carga Operativa por Componente")
carga_vars = ['Rebar_Cuts', 'Styrofoam_Cuts', 'Mesh_Chairs', 'Rebar_Chairs', 'Conduits']
carga = X[carga_vars].T.rename(columns={0: 'Cantidad'})
carga['Componente'] = carga.index
fig_carga = px.bar(carga, x='Componente', y='Cantidad', title="Carga Operativa")
st.plotly_chart(fig_carga)

# ================================
# SHAP EXPLICABILIDAD
# ================================
st.header("Importancia de Variables (SHAP)")
explainer = shap.Explainer(model_total)
shap_values = explainer(X)
st_shap = shap.plots.bar(shap_values[0], show=False)
st.pyplot(bbox_inches='tight')

# ================================
# RELACIONES CLAVE
# ================================
st.header("Relaciones técnicas con la productividad")
def scatter_plot(var):
    fig = px.scatter(df_ref, x=var, y=(df_ref['Preparation'] + df_ref['Formwork'] + df_ref['Assembly'] + df_ref['Concreting'] + df_ref['Lifting'] + df_ref['Finishing'])/60/ (df_ref['Area'] * 0.092903),
                     labels={'x': var, 'y': 'hh/m²'}, title=f"{var} vs Productividad")
    st.plotly_chart(fig)

for var in ['Area', 'Concrete_Quantity_Panels', 'Total_Rebar_Length']:
    if var in df_ref.columns:
        scatter_plot(var)

# ================================
# RECOMENDACIONES BÁSICAS
# ================================
st.header("Recomendaciones Automáticas")
recom = []
if user_input['Rebar_Cuts'] > df_ref['Rebar_Cuts'].quantile(0.75):
    recom.append("🔧 Considera reducir los cortes de acero para agilizar la producción.")
if user_input['Styrofoam_Cuts'] > df_ref['Styrofoam_Cuts'].quantile(0.75):
    recom.append("🔧 Revisa si la geometría de los bloques de poliestireno puede simplificarse.")
if user_input['Conduits'] > df_ref['Conduits'].quantile(0.75):
    recom.append("🔧 Evalúa si se pueden integrar menos ductos sin comprometer funcionalidad.")

if recom:
    for r in recom:
        st.write(r)
else:
    st.success("✅ Diseño técnico sin alertas destacadas. Buen trabajo.")
