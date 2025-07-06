import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Panel Production Dashboard", page_icon=":hammer_and_wrench:", layout="wide")

st.title("Panel Production Dashboard")
st.markdown("_Análisis de Productividad y Tiempo de Producción_")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Sube tu archivo Excel con datos de paneles")

# --- Start of the main content block, only runs if a file is uploaded ---
if uploaded_file is None:
    st.info("⬆️ Por favor, sube un archivo a través de la configuración para comenzar.", icon="ℹ️")
    st.stop() # Stop execution here if no file is uploaded
else: # This 'else' block ensures all the processing happens only when a file exists
    #######################################
    # DATA LOADING & SIMULATION
    #######################################

    @st.cache_data
    def load_and_process_data(path): # Type hint for clarity
        df = pd.read_excel(path)

        # Convertir nombres de columnas a un formato más amigable para Python
        df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('&', 'and').str.replace('.', '').str.replace('-', '_').str.replace('+', '').str.replace('__', '_')

        # Renombrar columnas específicas para que coincidan con el input del modelo simulado
        df = df.rename(columns={
            'Total_Rebar_Length': 'Total_Rebar_Length_ft',
            'Plywood_Area': 'Plywood_Area_sqft',
            'Ribs': 'No_of_Ribs' # Asumiendo que 'Ribs' en el excel es 'No_of_Ribs' para el modelo
        })

        # Simulación de duraciones y tiempos de producción total
        num_rows = len(df)
        df['Preparation_min'] = np.random.randint(60, 240, num_rows)
        df['Formwork_min'] = np.random.randint(90, 300, num_rows)
        df['Assembly_min'] = np.random.randint(120, 480, num_rows)
        df['Concreting_min'] = np.random.randint(45, 180, num_rows)
        df['Lifting_min'] = np.random.randint(30, 120, num_rows)
        df['Finishing_min'] = np.random.randint(60, 240, num_rows)
        df['Prod_Time_min'] = df[['Preparation_min', 'Formwork_min', 'Assembly_min', 'Concreting_min', 'Lifting_min', 'Finishing_min']].sum(axis=1)

        # Simulación de columnas de trabajador si no existen en el input
        # Estas son necesarias para el cálculo de hh/m2
        if 'Formwork_Worker' not in df.columns:
            df['Formwork_Worker'] = np.random.randint(1, 5, num_rows)
        if 'Concreting_Worker' not in df.columns:
            df['Concreting_Worker'] = np.random.randint(1, 4, num_rows)
        if 'Lifting_Worker' not in df.columns:
            df['Lifting_Worker'] = np.random.randint(1, 3, num_rows)
        if 'Finishing_Worker' not in df.columns:
            df['Finishing_Worker'] = np.random.randint(1, 5, num_rows)
        if 'Preparation_Worker' not in df.columns:
            df['Preparation_Worker'] = np.random.randint(1, 5, num_rows)
        if 'Assembly_Worker' not in df.columns:
            df['Assembly_Worker'] = np.random.randint(1, 5, num_rows)


        df['Total_Worker_Hours'] = (df['Preparation_Worker'] + df['Formwork_Worker'] + df['Assembly_Worker'] +
                                    df['Concreting_Worker'] + df['Lifting_Worker'] + df['Finishing_Worker'])

        # Asegúrate de que 'Area' esté en metros cuadrados y no sea cero para evitar divisiones por cero
        df['Area'] = df['Area'].replace(0, np.nan) # Replace 0 with NaN
        df['Estimated_Productivity_hh_m2'] = df['Total_Worker_Hours'] / df['Area']
        df['Productivity_Prod_Time_min_per_m2'] = df['Prod_Time_min'] / df['Area']


        return df

    # Now this line will only be reached if uploaded_file is NOT None
    df = load_and_process_data(uploaded_file)

    with st.expander("Vista Previa de los Datos Procesados"):
        st.dataframe(df)

    #######################################
    # VISUALIZATION METHODS (Removed functions for excluded plots)
    #######################################

    def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
        fig = go.Figure()

        fig.add_trace(
            go.Indicator(
                value=value,
                gauge={"axis": {"visible": False}},
                number={
                    "prefix": prefix,
                    "suffix": suffix,
                    "font.size": 28,
                },
                title={
                    "text": label,
                    "font": {"size": 20},
                },
            )
        )

        if show_graph:
            fig.add_trace(
                go.Scatter(
                    y=random.sample(range(0, 101), 30),
                    hoverinfo="skip",
                    fill="tozeroy",
                    fillcolor=color_graph,
                    line={
                        "color": color_graph,
                    },
                )
            )

        fig.update_xaxes(visible=False, fixedrange=True)
        fig.update_yaxes(visible=False, fixedrange=True)
        fig.update_layout(
            margin=dict(t=30, b=0),
            showlegend=False,
            plot_bgcolor="white",
            height=100,
        )

        st.plotly_chart(fig, use_container_width=True)


    def plot_gauge(
        indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
    ):
        fig = go.Figure(
            go.Indicator(
                value=indicator_number,
                mode="gauge+number",
                domain={"x": [0, 1], "y": [0, 1]},
                number={
                    "suffix": indicator_suffix,
                    "font.size": 26,
                },
                gauge={
                    "axis": {"range": [0, max_bound], "tickwidth": 1},
                    "bar": {"color": indicator_color},
                },
                title={
                    "text": indicator_title,
                    "font": {"size": 24},
                },
            )
        )
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=50, b=10, pad=8),
        )
        st.plotly_chart(fig, use_container_width=True)

    #######################################
    # CUSTOM VISUALIZATIONS (Removed functions for excluded plots, SHAP restored)
    #######################################

    def plot_total_production_time_gauge(df_data):
        avg_prod_time = df_data['Prod_Time_min'].mean()
        max_prod_time = df_data['Prod_Time_min'].max()
        min_prod_time = df_data['Prod_Time_min'].min()

        st.subheader("Tiempo Total de Producción Estimado")
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_gauge(avg_prod_time, "#0068C9", " min", "Promedio", max_prod_time * 1.2 if max_prod_time > 0 else 1)
        with col2:
            plot_gauge(min_prod_time, "#29B09D", " min", "Mínimo", max_prod_time * 1.2 if max_prod_time > 0 else 1)
        with col3:
            plot_gauge(max_prod_time, "#FF2B2B", " min", "Máximo", max_prod_time * 1.2 if max_prod_time > 0 else 1)


    # Removed: plot_stage_production_time_gauges (Tiempos Estimados por Etapa)
    # def plot_stage_production_time_gauges(df_data):
    #     st.subheader("Tiempos Estimados por Etapa")
    #     stages = ["Preparation", "Formwork", "Assembly", "Concreting", "Lifting", "Finishing"]
    #     cols = st.columns(len(stages))
    #     for i, stage in enumerate(stages):
    #         with cols[i]:
    #             avg_time = df_data[f'{stage}_min'].mean()
    #             max_time = df_data[f'{stage}_min'].max()
    #             plot_gauge(avg_time, "#F63366", " min", f"{stage} Prom.", max_time * 1.2 if max_time > 0 else 1)


    # Removed: plot_productivity_ratio_per_stage (Análisis de Productividad y Relaciones)
    # def plot_productivity_ratio_per_stage(df_data):
    #     st.subheader("Ratio de Productividad (min/m²) por Etapa")
    #     stages = ["Preparation", "Formwork", "Assembly", "Concreting", "Lifting", "Finishing"]
    #     productivity_data = []
    #     for stage in stages:
    #         avg_productivity = (df_data[f'{stage}_min'] / df_data['Area']).replace([np.inf, -np.inf], np.nan).mean()
    #         productivity_data.append({"Etapa": stage, "Productividad (min/m²)": avg_productivity})

    #     df_productivity = pd.DataFrame(productivity_data)
    #     df_productivity = df_productivity.dropna(subset=["Productividad (min/m²)"])

    #     if not df_productivity.empty:
    #         fig = px.bar(df_productivity,
    #                     x="Etapa",
    #                     y="Productividad (min/m²)",
    #                     title="Productividad Promedio por Etapa",
    #                     color="Productividad (min/m²)",
    #                     color_continuous_scale=px.colors.sequential.Viridis,
    #                     text_auto=".2f")
    #         fig.update_traces(textposition="outside")
    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.info("No hay datos de productividad para mostrar. Verifica la columna 'Area' en tu archivo.")


    #     total_avg_productivity = (df_data['Prod_Time_min'] / df_data['Area']).replace([np.inf, -np.inf], np.nan).mean()
    #     if not pd.isna(total_avg_productivity):
    #         st.markdown(f"**Productividad Total Promedio:** {total_avg_productivity:.2f} min/m²")
    #     else:
    #         st.markdown("**Productividad Total Promedio:** No calculable (verifica la columna 'Area').")


    # Removed: plot_scatter_productivity (Relación entre Variables Técnicas y Productividad)
    # def plot_scatter_productivity(df_data):
    #     st.subheader("Relación entre Variables Técnicas y Productividad")

    #     options = {
    #         'Área (m²)' : 'Area',
    #         'Volumen de Concreto (m³)' : 'Concrete_Quantity_Panels',
    #         'Longitud Total de Armaduras (ft)' : 'Total_Rebar_Length_ft',
    #         'Cantidad de Costillas' : 'No_of_Ribs',
    #         'Área de Malla de Alambre (m²)' : 'Area_Wire_Mesh',
    #         'Cortes de Armadura' : 'Rebar_Cuts',
    #         'Sillas de Armadura' : 'Rebar_Chairs',
    #         'Cortes de Styrofoam' : 'Styrofoam_Cuts',
    #         'Conduits' : 'Conduits',
    #         'Anchors' : 'Anchors',
    #         'Windows & Doors' : 'WindowsandDoors',
    #         'Plywood Area (sqft)' : 'Plywood_Area_sqft'
    #     }

    #     available_options = {k: v for k, v in options.items() if v in df_data.columns}

    #     if not available_options:
    #         st.warning("No se encontraron columnas técnicas relevantes en tus datos para el scatter plot.")
    #         return

    #     selected_x_var_display = st.selectbox("Selecciona la variable técnica para el Scatter Plot:", list(available_options.keys()))
    #     selected_x_var_column = available_options[selected_x_var_display]

    #     plot_df = df_data.dropna(subset=[selected_x_var_column, "Productivity_Prod_Time_min_per_m2"])

    #     if not plot_df.empty:
    #         fig = px.scatter(plot_df,
    #                         x=selected_x_var_column,
    #                         y="Productivity_Prod_Time_min_per_m2",
    #                         title=f"Productividad (min/m²) vs {selected_x_var_display}",
    #                         trendline="ols",
    #                         color_discrete_sequence=px.colors.qualitative.Pastel)
    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.info("No hay suficientes datos para crear el scatter plot con las selecciones actuales.")


    # Restored: plot_feature_importance (Importancia de las Variables y Recomendaciones)
    def plot_feature_importance():
        st.subheader("Importancia de las Variables (Simulado - SHAP)")
        st.info("💡 **Nota:** Este gráfico es una simulación. Cuando integres tu modelo real, aquí se mostrará la importancia real de las variables usando SHAP u otra técnica.", icon="ℹ️")

        simulated_importance = {
            'Preparation_Worker': 0.18,
            'Total_Rebar_Length_ft': 0.15,
            'Assembly_Worker': 0.12,
            'Area': 0.10,
            'Concrete_Quantity_Panels': 0.09,
            'Rebar_Cuts': 0.08,
            'No_of_Ribs': 0.07,
            'WindowsandDoors': 0.06,
            'Styrofoam_Cuts': 0.05,
            'Area_Wire_Mesh': 0.04,
            'Mesh_Chairs': 0.03,
            'Rebar_Chairs': 0.02,
            'Conduits': 0.01,
            'Anchors': 0.005,
            'Plywood_Area_sqft': 0.005,
        }
        df_importance = pd.DataFrame(list(simulated_importance.items()), columns=['Feature', 'Importance'])
        df_importance = df_importance.sort_values(by='Importance', ascending=False)

        fig = px.bar(df_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Importancia Relativa de las Variables en la Predicción',
                    color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Recomendaciones Basadas en la Importancia de Variables (Simulado):")
        st.markdown("""
        - **Optimización de Mano de Obra en Preparación y Ensamblaje:** Dado que `Preparation (Worker)` y `Assembly (Worker)` son influyentes, revisar y optimizar los procesos en estas etapas puede generar mejoras significativas en el tiempo total de producción.
        - **Gestión de Armaduras:** La `Total_Rebar_Length_ft` y `Rebar_Cuts` son importantes. Esto sugiere que la complejidad y la manipulación del acero son factores clave. Buscar prefabricación o métodos que reduzcan cortes puede ser beneficioso.
        - **Diseño del Panel (Área y Concreto):** `Area` y `Concrete_Quantity_Panels` también son relevantes. Esto indica que el tamaño y el volumen de concreto impactan directamente en la duración.
        """)


    # Removed: plot_operational_load_heatmap (Mapa de Calor y Análisis de Complejidad)
    # def plot_operational_load_heatmap(df_data):
    #     st.subheader("Mapa de Calor de Carga Operativa (Contribución al Tiempo)")
    #     st.info("Este mapa de calor simula cuánto 'aportan' ciertos elementos (basado en sus cantidades) al tiempo total de producción.", icon="ℹ️")

    #     load_columns = [
    #         'Concrete_Quantity_Panels', 'WindowsandDoors', 'No_of_Ribs', 'Plywood_Area_sqft',
    #         'Total_Rebar_Length_ft', 'Rebar_Cuts', 'Rebar_Chairs', 'Area_Wire_Mesh', 'Mesh_Chairs',
    #         'Styrofoam_Ribs', 'Styrofoam_Cuts', 'Styrofoam_Boards', 'Conduits', 'Anchors'
    #     ]
    #     load_columns_present = [col for col in load_columns if col in df_data.columns]

    #     if not load_columns_present:
    #         st.warning("No se encontraron columnas de carga operativa relevantes en tus datos para el mapa de calor.")
    #         return

    #     df_load = df_data[load_columns_present].copy()

    #     for col in df_load.columns:
    #         if df_load[col].max() > df_load[col].min():
    #             df_load[col] = (df_load[col] - df_load[col].min()) / (df_load[col].max() - df_load[col].min())
    #         else:
    #             df_load[col] = 0

    #     df_load['Panel_ID'] = df_data.index.astype(str)

    #     df_melted = df_load.melt(id_vars=['Panel_ID'], var_name='Elemento', value_name='Carga_Normalizada')
    #     df_melted = df_melted.dropna(subset=['Carga_Normalizada'])

    #     if not df_melted.empty:
    #         fig = px.density_heatmap(df_melted,
    #                                 x="Elemento",
    #                                 y="Panel_ID",
    #                                 z="Carga_Normalizada",
    #                                 title="Carga Operativa de Elementos por Panel (Normalizado)",
    #                                 color_continuous_scale="Viridis")
    #         fig.update_layout(xaxis_nticks=len(load_columns_present), yaxis_title="ID del Panel")
    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.info("No hay datos suficientes para crear el mapa de calor de carga operativa.")


    #     st.markdown("---")
    #     st.subheader("Análisis de Complejidad Estructural Estimada")
    #     st.info("Esta es una métrica simplificada que combina el número de elementos y detalles técnicos para dar una idea de la complejidad del panel.", icon="ℹ️")

    #     complexity_factors = {
    #         'No_of_Ribs': 0.5,
    #         'Rebar_Cuts': 0.8,
    #         'Rebar_Chairs': 0.3,
    #         'WindowsandDoors': 1.0,
    #         'Styrofoam_Cuts': 0.4,
    #         'Conduits': 0.2,
    #         'Anchors': 0.1
    #     }

    #     df_data['Complexity_Score'] = 0
    #     for col, weight in complexity_factors.items():
    #         if col in df_data.columns:
    #             df_data['Complexity_Score'] += df_data[col] * weight
    #         else:
    #             st.warning(f"La columna '{col}' necesaria para el cálculo de la complejidad no se encontró en los datos. Se omitirá.")


    #     plot_df_complexity = df_data.dropna(subset=['Complexity_Score', 'Productivity_Prod_Time_min_per_m2', 'Area', 'Concrete_Quantity_Panels'])

    #     if not plot_df_complexity.empty:
    #         fig_complexity_productivity = px.scatter(plot_df_complexity,
    #                                                 x='Complexity_Score',
    #                                                 y='Productivity_Prod_Time_min_per_m2',
    #                                                 title='Complejidad Estructural vs Productividad',
    #                                                 trendline="ols",
    #                                                 color='Prod_Time_min',
    #                                                 hover_data=['Area', 'Concrete_Quantity_Panels'])
    #         st.plotly_chart(fig_complexity_productivity, use_container_width=True)
    #     else:
    #         st.info("No hay suficientes datos para crear el scatter plot de complejidad vs productividad.")


    #######################################
    # STREAMLIT LAYOUT
    #######################################

    st.markdown("---")
    st.header("Resumen General de KPIs")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    with col_kpi1:
        avg_productivity = df['Estimated_Productivity_hh_m2'].mean()
        plot_metric(
            "Productividad Estimada",
            avg_productivity if not pd.isna(avg_productivity) else 0,
            suffix=" hh/m²",
            show_graph=False,
            color_graph="rgba(0, 104, 201, 0.2)",
        )

    with col_kpi2:
        total_concrete_volume = df['Concrete_Quantity_Panels'].sum()
        plot_metric(
            "Volumen Total de Concreto",
            total_concrete_volume,
            suffix=" m³",
            show_graph=False,
            color_graph="rgba(0, 104, 201, 0.2)",
        )

    with col_kpi3:
        total_panel_area = df['Area'].sum()
        plot_metric(
            "Área Total de Paneles",
            total_panel_area,
            suffix=" m²",
            show_graph=False,
            color_graph="rgba(0, 104, 201, 0.2)",
        )

    st.markdown("---")
    st.header("Avance Estimado y Tiempos de Producción")
    col_progress, col_overall_time = st.columns(2)
    with col_progress:
        st.subheader("Porcentaje de Avance Estimado")
        st.info("Esto simula un avance general. En un sistema real, se calcularía por panel o por fase completada.", icon="ℹ️")
        simulated_progress = random.randint(50, 95)
        st.metric(label="Avance General", value=f"{simulated_progress}%", delta=f"{random.randint(-5, 5)}%")

    with col_overall_time:
        plot_total_production_time_gauge(df)

    # Removed: plot_stage_production_time_gauges(df)
    # Removed: st.header("Análisis de Productividad y Relaciones")
    # Removed: plot_productivity_ratio_per_stage(df)
    # Removed: plot_scatter_productivity(df)

    st.markdown("---")
    st.header("Análisis Avanzado y Recomendaciones") # This header is now just for SHAP
    plot_feature_importance() # SHAP and recommendations are called here
    # Removed: plot_operational_load_heatmap(df) # Map of Heat and Complexity Analysis


    # Sección para ingresar datos de un nuevo panel
    st.markdown("---")
    st.header("Estimar Tiempo para un Nuevo Panel")
    st.markdown("Introduce los datos de un nuevo panel para obtener una estimación de su tiempo de producción.")

    with st.form("new_panel_form"):
        st.subheader("Variables de Entrada para el Modelo (Simulado)")

        col_input1, col_input2, col_input3 = st.columns(3)

        model_inputs = {
            'Preparation (Worker)': 'prep_worker',
            'Total_Rebar_Length(ft)': 'rebar_length',
            'Assembly (Worker)': 'assembly_worker',
            'Area': 'area',
            'Area_Wire_Mesh': 'mesh_area',
            'Mesh_Chairs': 'mesh_chairs',
            'Windows&Doors': 'windows_doors',
            'Rebar_Cuts': 'rebar_cuts',
            'Rebar_Chairs': 'rebar_chairs',
            'Styrofoam_Cuts': 'styrofoam_cuts',
            'Concrete_Quantity_Panels': 'concrete_quantity',
            'No_of_Ribs': 'num_ribs',
            'Plywood_Area(sqft)': 'plywood_area',
            'Conduits': 'conduits',
            'Anchors': 'anchors'
        }

        for i, (label, key) in enumerate(model_inputs.items()):
            current_col = [col_input1, col_input2, col_input3][i % 3]
            with current_col:
                if 'Worker' in label or 'Area' in label or 'Length' in label or 'Quantity' in label or 'Plywood' in label:
                    st.number_input(label, min_value=0.0, value=1.0, key=key)
                else:
                    st.number_input(label, min_value=0, value=1, key=key)


        submitted = st.form_submit_button("Estimar Tiempo")

        if submitted:
            sim_prod_time = random.randint(240, 720)
            sim_prep_time = random.randint(60, sim_prod_time // 4)
            sim_form_time = random.randint(60, sim_prod_time // 4)
            sim_assembly_time = random.randint(90, sim_prod_time // 3)
            sim_conc_time = random.randint(45, sim_prod_time // 5)
            sim_lift_time = random.randint(30, sim_prod_time // 6)
            sim_finish_time = random.randint(60, sim_prod_time // 4)

            st.subheader(f"Estimación para el Nuevo Panel:")
            st.metric("Tiempo de Producción Total Estimado:", f"{sim_prod_time} min")

            st.markdown("**Desglose por Etapa (minutos):**")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.write(f"- Preparación: {sim_prep_time} min")
                st.write(f"- Encofrado: {sim_form_time} min")
            with col_res2:
                st.write(f"- Ensamblaje: {sim_assembly_time} min")
                st.write(f"- Hormigonado: {sim_conc_time} min")
            with col_res3:
                st.write(f"- Levantamiento: {sim_lift_time} min")
                st.write(f"- Acabado: {sim_finish_time} min")

            st.markdown("---")
            st.subheader("Factores Clave para esta Predicción (Simulado)")
            st.info("En un modelo real, se usarían técnicas como SHAP para explicar la predicción de este panel específico.", icon="ℹ️")
            st.markdown("- **Área y Cantidad de Concreto:** Generalmente, paneles más grandes o con más concreto toman más tiempo.")
            st.markdown("- **Complejidad de Armaduras:** Más cortes y sillas de armadura pueden aumentar significativamente el tiempo.")
            st.markdown("- **Elementos Adicionales:** Ventanas, puertas, conducciones y anclajes añaden complejidad y tiempo.")
