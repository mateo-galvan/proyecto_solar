# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 20:56:53 2025

@author: SYSTEM-ACTUAL
"""

# -*- coding: utf-8 -*-
"""
Simulador de Energía Solar para Medellín
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

class SolarEnergySimulator:
    def __init__(self):
        # Constantes
        self.SOLAR_CONSTANT = 1361  # W/m²
        self.PANEL_TYPES = {
            'monocristalino': {'efficiency': 0.18, 'temp_coeff': -0.004},
            'policristalino': {'efficiency': 0.15, 'temp_coeff': -0.0045},
            'película_delgada': {'efficiency': 0.10, 'temp_coeff': -0.002}
        }
        
        # Parámetros por defecto para Medellín
        self.latitude = 6.2442  # Medellín
        self.longitude = -75.5812
        self.date = datetime.now().date()
        self.panel_type = 'monocristalino'
        self.panel_area = 1.6  # m²
        self.tilt_angle = 10  # grados (optimizado para Medellín cerca del ecuador)
        self.azimuth_angle = 180  # grados (0=N, 90=E, 180=S, 270=W)
        
    def calculate_solar_position(self, date, hour, lat, lon):
        """Calcula la posición solar (altitud y azimut) para una fecha, hora y ubicación dadas"""
        # Día juliano (día del año)
        day_of_year = date.timetuple().tm_yday
        
        # Conversión de hora a hora solar
        time_offset = (4 * (lon) + 60 * (0)) / 60  # Simplificado
        solar_time = hour + time_offset
        
        # Ángulo horario (grados)
        hour_angle = 15 * (solar_time - 12)
        
        # Declinación solar (grados)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Altitud solar (grados)
        lat_rad = np.radians(lat)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)
        
        sin_altitude = (np.sin(lat_rad) * np.sin(dec_rad) + 
                       np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
        altitude = np.degrees(np.arcsin(sin_altitude))
        
        # Azimut solar (grados)
        cos_azimuth = ((np.sin(dec_rad) * np.cos(lat_rad) - 
                       np.cos(dec_rad) * np.sin(lat_rad) * np.cos(ha_rad)) / 
                      np.cos(np.radians(altitude)))
        azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))
        
        # Ajustar azimut según mañana/tarde
        if hour_angle > 0:
            azimuth = 360 - azimuth
        
        return max(0, altitude), azimuth
    
    def calculate_irradiance(self, altitude, date, cloud_cover=0.5):
        """Calcula la irradiancia solar considerando efectos atmosféricos"""
        # Irradiancia extraterrestre
        day_of_year = date.timetuple().tm_yday
        n = 1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365))
        extraterrestrial_irradiance = self.SOLAR_CONSTANT * n
        
        if altitude <= 0:
            return 0, 0, 0
        
        # Masa de aire
        air_mass = 1 / np.sin(np.radians(altitude))
        
        # Transmitancia atmosférica (modelo simplificado)
        # Aumentamos la nubosidad promedio para Medellín
        atmospheric_transmittance = (0.7 ** (air_mass ** 0.678)) * (1 - cloud_cover * 0.75)
        
        # Irradiancia directa normal (DNI)
        dni = extraterrestrial_irradiance * atmospheric_transmittance
        
        # Irradiancia directa horizontal (DHI)
        dhi = dni * np.sin(np.radians(altitude))
        
        # Irradiancia difusa horizontal (modelo simplificado)
        # Mayor fracción difusa por la nubosidad de Medellín
        diffuse_fraction = 0.2 + 0.6 * cloud_cover
        diffuse_irradiance = dhi * diffuse_fraction
        
        # Irradiancia global horizontal (GHI)
        ghi = dhi + diffuse_irradiance
        
        return ghi, dhi, diffuse_irradiance
    
    def calculate_irradiance_on_tilted_surface(self, ghi, dhi, altitude, azimuth, 
                                             tilt_angle, azimuth_angle):
        """Calcula la irradiancia en una superficie inclinada"""
        # Ángulo de incidencia
        cos_incidence = (np.sin(np.radians(altitude)) * np.cos(np.radians(tilt_angle)) +
                        np.cos(np.radians(altitude)) * np.sin(np.radians(tilt_angle)) *
                        np.cos(np.radians(azimuth - azimuth_angle)))
        
        # Factor de inclinación para radiación directa
        rb = max(0, cos_incidence / max(0.087, np.sin(np.radians(altitude))))
        
        # Radiación directa en superficie inclinada
        direct_tilted = dhi * rb
        
        # Radiación difusa en superficie inclinada (modelo isotrópico)
        diffuse_tilted = dhi * ((1 + np.cos(np.radians(tilt_angle))) / 2)
        
        # Radiación total en superficie inclinada
        total_tilted = direct_tilted + diffuse_tilted
        
        return total_tilted, direct_tilted, diffuse_tilted
    
    def calculate_power_output(self, irradiance, panel_type, panel_area, temperature=25):
        """Calcula la potencia de salida del panel fotovoltaico"""
        panel_params = self.PANEL_TYPES[panel_type]
        efficiency = panel_params['efficiency'] * (1 + panel_params['temp_coeff'] * (temperature - 25))
        power = irradiance * panel_area * efficiency
        return max(0, power)
    
    def get_real_solar_data(self, lat, lon, date):
        """Obtiene datos reales de radiación solar de la API de Open-Meteo"""
        try:
            start_date = date.strftime("%Y-%m-%d")
            end_date = date.strftime("%Y-%m-%d")
            
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=direct_radiation,diffuse_radiation&timezone=auto"
            
            response = requests.get(url)
            data = response.json()
            
            if 'hourly' in data:
                hours = list(range(24))
                direct_rad = data['hourly']['direct_radiation']
                diffuse_rad = data['hourly']['diffuse_radiation']
                return hours, direct_rad, diffuse_rad
            else:
                return None, None, None
        except:
            return None, None, None
    
    def run_simulation(self, latitude, longitude, date, panel_type, panel_area, tilt_angle, azimuth_angle):
        """Ejecuta la simulación completa"""
        # Para Medellín, extendemos el rango horario ya que hay más horas de luz
        hours = np.linspace(5, 19, 15)  # De 5am a 7pm
        altitudes = []
        azimuths = []
        ghi_values = []
        dhi_values = []
        diffuse_values = []
        tilted_values = []
        power_values = []
        
        # Obtener datos reales para comparación
        real_hours, real_direct, real_diffuse = self.get_real_solar_data(latitude, longitude, date)
        
        for hour in hours:
            # Calcular posición solar
            altitude, azimuth = self.calculate_solar_position(date, hour, latitude, longitude)
            altitudes.append(altitude)
            azimuths.append(azimuth)
            
            # Calcular irradiancia (mayor nubosidad para Medellín)
            ghi, dhi, diffuse = self.calculate_irradiance(altitude, date, cloud_cover=0.5)
            ghi_values.append(ghi)
            dhi_values.append(dhi)
            diffuse_values.append(diffuse)
            
            # Calcular irradiancia en superficie inclinada
            tilted_irradiance, _, _ = self.calculate_irradiance_on_tilted_surface(
                ghi, dhi, altitude, azimuth, tilt_angle, azimuth_angle)
            tilted_values.append(tilted_irradiance)
            
            # Calcular potencia de salida (temperatura ambiente más alta para Medellín)
            power = self.calculate_power_output(tilted_irradiance, panel_type, panel_area, temperature=28)
            power_values.append(power)
        
        return {
            'hours': hours,
            'altitudes': altitudes,
            'azimuths': azimuths,
            'ghi': ghi_values,
            'dhi': dhi_values,
            'diffuse': diffuse_values,
            'tilted': tilted_values,
            'power': power_values,
            'real_data': (real_hours, real_direct, real_diffuse) if real_hours else None
        }
    
    def create_interactive_ui(self):
        """Crea una interfaz interactiva para el usuario"""
        # Widgets de entrada con valores por defecto para Medellín
        lat_slider = widgets.FloatSlider(value=6.2442, min=-90, max=90, step=0.1, description='Latitud:')
        lon_slider = widgets.FloatSlider(value=-75.5812, min=-180, max=180, step=0.1, description='Longitud:')
        date_picker = widgets.DatePicker(value=datetime.now().date(), description='Fecha:')
        panel_dropdown = widgets.Dropdown(options=list(self.PANEL_TYPES.keys()), value='monocristalino', description='Tipo panel:')
        area_slider = widgets.FloatSlider(value=1.6, min=0.5, max=5, step=0.1, description='Área (m²):')
        tilt_slider = widgets.IntSlider(value=10, min=0, max=90, step=1, description='Inclinación (°):')
        azimuth_slider = widgets.IntSlider(value=180, min=0, max=360, step=1, description='Orientación (°):')
        
        run_button = widgets.Button(description="Ejecutar Simulación", button_style='success')
        output = widgets.Output()
        
        # Diseño de la interfaz
        left_panel = widgets.VBox([lat_slider, lon_slider, date_picker, panel_dropdown])
        right_panel = widgets.VBox([area_slider, tilt_slider, azimuth_slider, run_button])
        ui = widgets.HBox([left_panel, right_panel])
        
        # Función de callback para el botón
        def on_run_button_clicked(b):
            with output:
                clear_output()
                self.latitude = lat_slider.value
                self.longitude = lon_slider.value
                self.date = date_picker.value
                self.panel_type = panel_dropdown.value
                self.panel_area = area_slider.value
                self.tilt_angle = tilt_slider.value
                self.azimuth_angle = azimuth_slider.value
                
                # Ejecutar simulación
                results = self.run_simulation(
                    self.latitude, self.longitude, self.date, 
                    self.panel_type, self.panel_area, 
                    self.tilt_angle, self.azimuth_angle
                )
                
                # Visualizar resultados
                self.visualize_results(results)
        
        run_button.on_click(on_run_button_clicked)
        
        # Mostrar la interfaz
        display(ui, output)
        
        # Ejecutar simulación inicial
        results = self.run_simulation(
            self.latitude, self.longitude, self.date, 
            self.panel_type, self.panel_area, 
            self.tilt_angle, self.azimuth_angle
        )
        with output:
            self.visualize_results(results)
    
    def visualize_results(self, results):
        """Visualiza los resultados de la simulación"""
        hours = results['hours']
        altitudes = results['altitudes']
        power_values = results['power']
        real_data = results['real_data']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico 1: Altitud solar
        ax1.plot(hours, altitudes, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Hora del día')
        ax1.set_ylabel('Altitud solar (°)')
        ax1.set_title('Altitud Solar a lo Largo del Día - Medellín')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(5, 20, 2))
        
        # Gráfico 2: Potencia de salida
        ax2.plot(hours, power_values, 'r-', linewidth=2, marker='o')
        ax2.set_xlabel('Hora del día')
        ax2.set_ylabel('Potencia (W)')
        ax2.set_title('Producción de Energía del Panel Fotovoltaico - Medellín')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(5, 20, 2))
        
        # Gráfico 3: Comparación de irradiancia
        ax3.plot(hours, results['ghi'], 'g-', label='Global Horizontal', linewidth=2)
        ax3.plot(hours, results['dhi'], 'b-', label='Directa Horizontal', linewidth=2)
        ax3.plot(hours, results['diffuse'], 'orange', label='Difusa', linewidth=2)
        ax3.plot(hours, results['tilted'], 'r-', label='En Panel Inclinado', linewidth=2)
        
        # Añadir datos reales si están disponibles
        if real_data:
            real_hours, real_direct, real_diffuse = real_data
            real_global = [d + df for d, df in zip(real_direct, real_diffuse)]
            ax3.plot(real_hours, real_global[:24], 'g--', label='GHI Real', alpha=0.7)
            ax3.plot(real_hours, real_direct[:24], 'b--', label='DNI Real', alpha=0.7)
            ax3.plot(real_hours, real_diffuse[:24], '--', color='orange', label='Difusa Real', alpha=0.7)
        
        ax3.set_xlabel('Hora del día')
        ax3.set_ylabel('Irradiancia (W/m²)')
        ax3.set_title('Comparación de Irradiancia Solar - Medellín')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 2))
        
        # Gráfico 4: Diagrama de posición solar
        azimuths_rad = np.radians(results['azimuths'])
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(azimuths_rad, results['altitudes'], 'ro-', linewidth=2)
        ax4.set_theta_zero_location('N')
        ax4.set_theta_direction(-1)
        ax4.set_rlabel_position(0)
        ax4.set_ylim(0, 90)
        ax4.set_title('Trayectoria Solar - Medellín (Coordenadas Polares)', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Calcular estadísticas
        total_energy = np.trapz(power_values, hours)  # Wh
        max_power = max(power_values)
        max_irradiance = max(results['tilted'])
        
        # Información específica de Medellín
        print("="*60)
        print("SIMULACIÓN DE ENERGÍA SOLAR PARA MEDELLÍN")
        print("="*60)
        print(f"Ubicación: Latitud {self.latitude}°, Longitud {self.longitude}°")
        print(f"Fecha: {self.date.strftime('%d/%m/%Y')}")
        print(f"Tipo de panel: {self.panel_type}")
        print(f"Área del panel: {self.panel_area} m²")
        print(f"Ángulo de inclinación: {self.tilt_angle}° (recomendado para Medellín: 10-15°)")
        print(f"Orientación (azimut): {self.azimuth_angle}° (Sur es lo óptimo)")
        print("-"*60)
        print(f"Energía total generada: {total_energy:.2f} Wh")
        print(f"Potencia máxima: {max_power:.2f} W")
        print(f"Irradiancia máxima en panel: {max_irradiance:.2f} W/m²")
        
        # Información adicional sobre Medellín
        print("\nCARACTERÍSTICAS DE MEDELLÍN:")
        print("- Ubicación: 6.2442° N, 75.5812° W")
        print("- Clima tropical con temperatura promedio: 22°C")
        print("- Nubosidad media-alta: ~50%")
        print("- Radiación solar promedio: 4.5-5.0 kWh/m²/día")
        print("- Recomendación: Inclinación de 10-15° hacia el Sur")
        
        if real_data:
            print("\nNota: Los datos reales (líneas discontinuas) se obtuvieron de la API de Open-Meteo")
        else:
            print("\nNota: No se pudieron obtener datos reales de la API")

# Ejecutar la aplicación
if __name__ == "__main__":
    simulator = SolarEnergySimulator()
    simulator.create_interactive_ui()