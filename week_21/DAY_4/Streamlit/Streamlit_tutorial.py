# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:39:57 2025

@author: milos
"""

# pip install streamlit
# pip install plotly-express

# Importiere Streamlit und weitere benötigte Bibliotheken
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setze den Titel der App
st.title("Streamlit Lehr-App")

# Begrüßungstext für die Nutzer
st.write("Willkommen zu dieser interaktiven Streamlit-App!")

# --- Abschnitt 1: Text und Markdown ---
st.header("1. Text und Markdown")

# Erklärung zur Textanzeige
st.write("Du kannst Text mit `st.write()` oder `st.markdown()` anzeigen.")

# Anzeige von einfachem Text
st.write("Das ist einfacher Text.")

# Anzeige von Markdown-Text mit Formatierungen
st.markdown("""
Das ist **Markdown**! Du kannst es verwenden, um Text zu formatieren:

- **Fett**
- *Kursiv*
- `Code`
- [Links](https://streamlit.io)
""")

# --- Abschnitt 2: Eingabewidgets ---
st.header("2. Eingabewidgets")

# Erklärung zu Eingabemöglichkeiten
st.write("Streamlit bietet verschiedene Eingabewidgets, um mit Nutzern zu interagieren.")

# Slider-Widget zur Alterseingabe
st.subheader("Slider")
age = st.slider("Wie alt bist du?", 0, 100, 25)
st.write(f"Du bist {age} Jahre alt.")

# Text-Eingabefeld
st.subheader("Texteingabe")
name = st.text_input("Wie heißt du?")
if name:
    st.write(f"Hallo, {name}!")

# Button-Widget
st.subheader("Button")
if st.button("Klick mich"):
    st.write("Du hast den Button geklickt!")

# --- Abschnitt 3: Datenanzeige ---
st.header("3. Datenanzeige")

# Erklärung zur Datenanzeige
st.write("Du kannst Daten in Tabellen, Diagrammen und mehr anzeigen.")

# Anzeige eines DataFrames mit Zufallsdaten
st.subheader("DataFrame")
data = pd.DataFrame({
    "Spalte 1": np.random.randn(10),
    "Spalte 2": np.random.randn(10)
})
st.write("Hier ist ein zufälliger DataFrame:")
st.dataframe(data)

# Anzeige eines Diagramms mit Matplotlib
st.subheader("Diagramm")
st.write("Du kannst Diagramme mit Matplotlib oder den eingebauten Streamlit-Diagrammfunktionen erstellen.")
fig, ax = plt.subplots()
ax.plot(data["Spalte 1"], data["Spalte 2"], "o")
st.pyplot(fig)

# --- Abschnitt 4: Layouts ---
st.header("4. Layouts")

# Erklärung zu Layout-Optionen
st.write("Streamlit bietet Layout-Optionen wie Spalten und Ausklappbereiche.")

# Zwei Spalten nebeneinander
st.subheader("Spalten")
col1, col2 = st.columns(2)
with col1:
    st.write("Das ist Spalte 1.")
with col2:
    st.write("Das ist Spalte 2.")

# Ausklappbereich (Expander)
st.subheader("Ausklappbereich")
with st.expander("Zum Ausklappen klicken"):
    st.write("Dieser Inhalt ist standardmäßig versteckt.")

# --- Abschnitt 5: Datei-Upload ---
st.header("5. Datei-Upload")

# Erklärung zum Datei-Upload
st.write("Du kannst Dateien hochladen und in deiner App verarbeiten.")

# Datei-Upload-Widget für CSV-Dateien
uploaded_file = st.file_uploader("Lade eine CSV-Datei hoch", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Hochgeladener DataFrame:")
    st.dataframe(df)

# --- Abschnitt 6: Interaktive Visualisierungen ---
st.header("6. Interaktive Visualisierungen")

# Erklärung zu interaktiven Diagrammen
st.write("Streamlit unterstützt interaktive Visualisierungen mit Bibliotheken wie Plotly.")

# Interaktives Plotly-Diagramm
st.subheader("Interaktives Plotly-Diagramm")
if st.checkbox("Plotly-Diagramm anzeigen"):
    import plotly.express as px
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    st.plotly_chart(fig)

# --- Abschnitt 7: Erweiterte Funktionen ---
st.header("7. Erweiterte Funktionen")

# Erklärung zu erweiterten Features
st.write("Streamlit unterstützt auch erweiterte Funktionen wie Caching und Session State.")


# Caching-Funktion zur Optimierung
st.subheader("Caching")
@st.cache_data # Decorator
def expensive_computation():
    st.write("Führe eine aufwändige Berechnung durch...")
    return np.random.rand(100)

# Die Funktion wird nur einmal für identische Eingaben ausgeführt wird.
# Bei weiteren Aufrufen mit den gleichen Parametern wird das zwischengespeicherte 
# Ergebnis (Cache) zurückgegeben, anstatt die Funktion erneut auszuführen.

# Aufruf der gecachten Funktion
cached_data = expensive_computation()
st.write("Gecachte Daten:", cached_data)

